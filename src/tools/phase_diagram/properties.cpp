// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "burnman/tools/phase_diagram/properties.hpp"
#include "burnman/core/relaxed_solution.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/tools/phase_diagram/consistency.hpp"
#include "burnman/tools/polytope.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>

namespace burnman::phase_diagrams {
namespace {
using Getter = double (Material::*)() const;
const std::map<std::string, Getter> getters = {
#define PROPERTY(name) {#name, &Material::get_##name}
    PROPERTY(pressure),
    PROPERTY(temperature),
    PROPERTY(molar_mass),
    PROPERTY(molar_internal_energy),
    PROPERTY(molar_gibbs),
    PROPERTY(molar_helmholtz),
    PROPERTY(molar_volume),
    PROPERTY(density),
    PROPERTY(molar_entropy),
    PROPERTY(molar_enthalpy),
    PROPERTY(isothermal_bulk_modulus_reuss),
    PROPERTY(isentropic_bulk_modulus_reuss),
    PROPERTY(isothermal_compressibility_reuss),
    PROPERTY(isentropic_compressibility_reuss),
    PROPERTY(shear_modulus),
    PROPERTY(p_wave_velocity),
    PROPERTY(bulk_sound_velocity),
    PROPERTY(shear_wave_velocity),
    PROPERTY(grueneisen_parameter),
    PROPERTY(thermal_expansivity),
    PROPERTY(molar_heat_capacity_v),
    PROPERTY(molar_heat_capacity_p),
    PROPERTY(isentropic_thermal_gradient)
#undef PROPERTY
};
void validate_properties(const std::vector<std::string> &names) {
  if (names.empty())
    throw std::invalid_argument("Supply at least one thermodynamic property.");
  for (const auto &name : names)
    if (!getters.count(name))
      throw std::invalid_argument("Unknown scalar thermodynamic property: " +
                                  name);
}

struct ReactionCurvature {
  Eigen::MatrixXd basis, scaled;
  Eigen::VectorXd scale, eigenvalues;
  Eigen::MatrixXd eigenvectors;
  explicit ReactionCurvature(const Assemblage &a)
      : basis(a.get_reaction_basis()) {
    if (!basis.rows())
      return;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(basis.cols(), basis.cols());
    const auto counts = a.get_endmembers_per_phase();
    const auto fractions = a.get_molar_fractions();
    Eigen::Index offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      const Eigen::Index n = counts[i];
      if (n > 1) {
        const double f = fractions[static_cast<Eigen::Index>(i)];
        if (f <= 0.)
          throw std::runtime_error("Zero-amount solution in reaction Hessian.");
        // Solution's Hessian differentiates fractions; the extensive Hessian
        // differentiates species amounts. Use a one-current-mole reference.
        h.block(offset, offset, n, n) =
            a.get_phase<Solution>(i)->get_gibbs_hessian() / f;
      }
      offset += n;
    }
    scaled = basis * h * basis.transpose();
    if (!scaled.allFinite())
      throw std::runtime_error("Non-finite reaction Hessian.");
    scale =
        scaled.diagonal().cwiseAbs().cwiseMax(1.).cwiseSqrt().cwiseInverse();
    scaled = scale.asDiagonal() * scaled * scale.asDiagonal();
    scaled = .5 * (scaled + scaled.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(scaled);
    if (eig.info() != Eigen::Success)
      throw std::runtime_error("Reaction Hessian eigensolve failed.");
    eigenvalues = eig.eigenvalues();
    eigenvectors = eig.eigenvectors();
    if (eigenvalues.minCoeff() <
        -1.e-9 * std::max(1., eigenvalues.cwiseAbs().maxCoeff()))
      throw std::runtime_error("Equilibrium has negative Gibbs curvature.");
  }
  double singular_tolerance() const {
    // Resolve soft positive modes down to roundoff, rather than imposing
    // a condition-number cutoff that masks finite low-temperature responses.
    return 64. * std::numeric_limits<double>::epsilon() *
           std::max(1., eigenvalues.cwiseAbs().maxCoeff());
  }
  bool singular() const {
    return basis.rows() && eigenvalues.minCoeff() <= singular_tolerance();
  }
  Eigen::MatrixXd couplings(const Assemblage &a,
                            Eigen::Vector2d *derivative_scale = nullptr) const {
    Eigen::MatrixXd partial(basis.cols(), 2);
    const auto counts = a.get_endmembers_per_phase();
    Eigen::Index offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      const Eigen::Index n = counts[i];
      if (auto s = a.get_phase<Solution>(i)) {
        partial.block(offset, 0, n, 1) = s->get_partial_volumes();
        partial.block(offset, 1, n, 1) = -s->get_partial_entropies();
      } else {
        partial(offset, 0) = a.get_phase(i)->get_molar_volume();
        partial(offset, 1) = -a.get_phase(i)->get_molar_entropy();
      }
      offset += n;
    }
    if (derivative_scale)
      *derivative_scale =
          (scale.asDiagonal() * basis.cwiseAbs() * partial.cwiseAbs())
              .colwise()
              .norm();
    Eigen::MatrixXd coupling =
        eigenvectors.transpose() * scale.asDiagonal() * basis * partial;
    if (!coupling.allFinite())
      throw std::runtime_error("Non-finite reaction derivative coupling.");
    return coupling;
  }
  double divergent_expansivity(const Assemblage &a) const {
    // Approach zero Gibbs curvature from the stable side. A null mode's
    // contribution is -dV*(-dS)/(V*lambda); retain its sign instead of
    // assigning a finite pseudoinverse response. Mixed signs are ambiguous.
    Eigen::Vector2d norms;
    const auto coupling = couplings(a, &norms);
    const double relative = 64. * std::numeric_limits<double>::epsilon();
    int sign = 0;
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
      if (eigenvalues[i] > singular_tolerance())
        continue;
      if (eigenvalues[i] < -singular_tolerance())
        throw std::runtime_error("Equilibrium has negative Gibbs curvature.");
      if (std::abs(coupling(i, 0)) <= relative * norms[0] ||
          std::abs(coupling(i, 1)) <= relative * norms[1])
        throw std::runtime_error("Singular reaction Hessian: expansivity "
                                 "divergence has no unique sign.");
      const int mode_sign =
          (coupling(i, 0) < 0.) != (coupling(i, 1) < 0.) ? 1 : -1;
      if (sign && mode_sign != sign)
        throw std::runtime_error("Singular reaction Hessian: expansivity "
                                 "divergence has no unique sign.");
      sign = mode_sign;
    }
    return sign * std::numeric_limits<double>::infinity();
  }
  Eigen::Matrix2d correction(const Assemblage &a) const {
    if (!basis.rows())
      return Eigen::Matrix2d::Zero();
    if (singular())
      throw std::runtime_error(
          "Singular reaction Hessian: relaxed response is undefined.");
    const auto coupling = couplings(a);
    return coupling.transpose() * eigenvalues.cwiseInverse().asDiagonal() *
           coupling;
  }
};

class Properties {
  const Assemblage &a;
  bool relaxed;
  std::optional<ReactionCurvature> curvature;
  std::optional<Eigen::Vector3d> response;
  const Eigen::Vector3d &thermal() {
    if (!response) {
      if (!curvature)
        curvature.emplace(a);
      const auto c = curvature->correction(a);
      const double v = a.get_molar_volume();
      response = Eigen::Vector3d(
          a.get_thermal_expansivity() - c(0, 1) / v,
          a.get_molar_heat_capacity_p() + a.get_temperature() * c(1, 1),
          a.get_isothermal_compressibility_reuss() + c(0, 0) / v);
    }
    return *response;
  }

public:
  Properties(const Assemblage &input, bool relax) : a(input), relaxed(relax) {}
  void check_stability() { curvature.emplace(a); }
  double value(const std::string &name) {
    if (relaxed && name == "thermal_expansivity") {
      if (!curvature)
        curvature.emplace(a);
      if (curvature->singular())
        return curvature->divergent_expansivity(a);
    }
    double value;
    if (!relaxed && name == "isentropic_thermal_gradient")
      value = a.get_thermal_expansivity() * a.get_molar_volume() *
              a.get_temperature() / a.get_molar_heat_capacity_p();
    else if (!relaxed || name == "pressure" || name == "temperature" ||
             name == "molar_mass" || name == "molar_internal_energy" ||
             name == "molar_gibbs" || name == "molar_helmholtz" ||
             name == "molar_volume" || name == "density" ||
             name == "molar_entropy" || name == "molar_enthalpy" ||
             name == "shear_modulus" || name == "shear_wave_velocity")
      value = (a.*getters.at(name))();
    else {
      const auto &r = thermal();
      const double alpha = r[0], cp = r[1], kt = 1. / r[2];
      const double v = a.get_molar_volume(), t = a.get_temperature();
      const double cv = cp - t * v * alpha * alpha * kt;
      const double ks = kt * cp / cv;
      if (name == "thermal_expansivity")
        value = alpha;
      else if (name == "molar_heat_capacity_p")
        value = cp;
      else if (name == "molar_heat_capacity_v")
        value = cv;
      else if (name == "isothermal_compressibility_reuss")
        value = r[2];
      else if (name == "isothermal_bulk_modulus_reuss")
        value = kt;
      else if (name == "isentropic_bulk_modulus_reuss")
        value = ks;
      else if (name == "isentropic_compressibility_reuss")
        value = 1. / ks;
      else if (name == "grueneisen_parameter")
        value = alpha * kt * v / cv;
      else if (name == "isentropic_thermal_gradient")
        value = alpha * v * t / cp;
      else if (name == "bulk_sound_velocity")
        value = std::sqrt(ks / a.get_density());
      else
        value =
            std::sqrt((ks + 4. / 3. * a.get_shear_modulus()) / a.get_density());
    }
    if (!std::isfinite(value))
      throw std::runtime_error("Non-finite property: " + name);
    return value;
  }
};

int ring_location(const Eigen::MatrixXd &ring, const Eigen::Vector2d &q,
                  const Eigen::Vector2d &origin, const Eigen::Vector2d &range) {
  return detail::ring_location(
      static_cast<std::size_t>(ring.rows()),
      [&](std::size_t i) -> Eigen::Vector2d {
        return (ring.row(static_cast<Eigen::Index>(i)).transpose() - origin)
            .cwiseQuotient(range);
      },
      (q - origin).cwiseQuotient(range), 1.e-9);
}
bool contains(const FieldPolygon &field, const Eigen::Vector2d &q,
              const Eigen::Vector2d &origin, const Eigen::Vector2d &range) {
  return ring_location(field.vertices, q, origin, range) >= 0 &&
         std::none_of(field.holes.begin(), field.holes.end(),
                      [&](const auto &hole) {
                        return ring_location(hole, q, origin, range) >= 0;
                      });
}

// Nearest compatible accepted point beside this column. A different phase set
// cannot initialise this candidate, but can be tried separately as an
// alternative.
const State *neighbour_seed(const std::map<double, State> &row,
                                    const std::vector<int> &ids,
                                    const Eigen::Vector2d &q, DiagramType type,
                                    const Eigen::Vector2d &range) {
  const auto upper = row.lower_bound(q[0]);
  const State *nearest = nullptr;
  double best = std::numeric_limits<double>::infinity();
  auto consider = [&](auto it) {
    if (it == row.end() || it->second.phases.size() != ids.size())
      return;
    for (int id : ids)
      if (detail::find_phase(it->second.phases, id) == it->second.phases.end())
        return;
    const double distance = (diagram_coordinates(it->second, type) - q)
                                .cwiseQuotient(range)
                                .squaredNorm();
    if (distance < best) {
      nearest = &it->second;
      best = distance;
    }
  };
  consider(upper);
  if (upper != row.begin())
    consider(std::prev(upper));
  return nearest;
}

std::shared_ptr<Assemblage>
equilibrate_grid_candidate(detail::Engine &engine, const std::vector<int> &ids,
                           const State &seed, const Eigen::Vector2d &q,
                           bool check_competitors) {
  // Preserve physical P,T as well as compositions in inverse S/V sections.
  auto a = engine.make_at(ids, seed.phases, q,
                          Eigen::Vector2d(seed.pressure, seed.temperature));
  a->set_averaging_scheme(types::AveragingType::VRH);
  auto constraints = engine.state_constraints(q);
  auto solve = engine.solve(*a, constraints, false);
  if (!solve.success)
    throw std::runtime_error("Equilibrium solve failed: " + solve.message);
  Properties(*a, false).check_stability();
  // Only fallback assemblages need a competing-phase search. A converged
  // assigned field is accepted directly, including provisional/open labels.
  if (check_competitors &&
      engine.stability(*a) < -engine.settings.affinity_tolerance)
    throw std::runtime_error(
        "Candidate assemblage has a negative competing-phase affinity.");
  return a;
}

// Saved phase amounts/compositions and physical P,T form a portable Newton
// starting state. Keep only the phases of this proposed assemblage.
std::optional<State> field_seed(const Result &source,
                                        const FieldPolygon &field,
                                        const Eigen::Vector2d &q,
                                        const Eigen::Vector2d &range) {
  std::optional<State> best;
  double distance = std::numeric_limits<double>::infinity();
  auto consider = [&](const auto &state) {
    for (int id : field.phases)
      if (detail::find_phase(state.phases, id) == state.phases.end())
        return;
    const double next = (diagram_coordinates(state, source.section.type) - q)
                            .cwiseQuotient(range)
                            .norm();
    if (next >= distance)
      return;
    State seed;
    seed.pressure = state.pressure;
    seed.temperature = state.temperature;
    seed.entropy = state.entropy;
    seed.volume = state.volume;
    seed.composition_coordinate = state.composition_coordinate;
    for (int id : field.phases)
      seed.phases.push_back(*detail::find_phase(state.phases, id));
    best = std::move(seed);
    distance = next;
  };
  for (const auto &state : source.samples)
    if (state.success && !state.outside_model_domain)
      consider(state);
  for (const auto &line : source.boundaries)
    for (const auto &state : line.points)
      consider(state);
  return best;
}

void validate_candidates(
    const Result &source,
    const std::vector<std::shared_ptr<Material>> &candidates) {
  const int stride = source.settings.max_phase_instances;
  if (source.composition_start.empty() || stride < 1)
    throw std::invalid_argument("Grid requires saved composition_start and a "
                                "positive phase-ID stride.");
  if (source.phase_names.size() !=
      candidates.size() * static_cast<std::size_t>(stride))
    throw std::invalid_argument(
        "Grid candidate names/order must match the saved phase_diagram.");
  for (std::size_t i = 0; i < candidates.size(); ++i)
    if (!candidates[i] ||
        candidates[i]->get_name() !=
            source.phase_names[i * static_cast<std::size_t>(stride)])
      throw std::invalid_argument(
          "Grid candidate names/order must match the saved phase_diagram.");
}

void validate_grid_axis(const Eigen::VectorXd &coordinates,
                        const std::array<double, 2> &bounds) {
  if (!coordinates.size() || !coordinates.allFinite() ||
      coordinates.minCoeff() < bounds[0] || coordinates.maxCoeff() > bounds[1])
    throw std::invalid_argument(
        "Grid coordinates must be finite, nonempty and inside the diagram.");
  for (Eigen::Index i = 1; i < coordinates.size(); ++i)
    if (coordinates[i] <= coordinates[i - 1])
      throw std::invalid_argument(
          "Grid coordinates must be strictly increasing.");
}

GridSolution capture_solution(const Assemblage &a, const detail::Engine &engine,
                              const std::vector<int> &ids) {
  GridSolution solution;
  solution.n_moles = a.get_n_moles();
  solution.gibbs = a.get_n_moles() * a.get_molar_gibbs();
  solution.molar_fractions = a.get_molar_fractions();
  engine.record_coordinates(solution.state, a);
  solution.state.phases = engine.snapshot(a, ids);
  solution.state.mass_balance_error = engine.mass_error(a);
  solution.state.residual = detail::reaction_residual(a);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto phase = a.get_phase(i);
    solution.phase_bases.push_back(engine.composition_basis(
        *phase, solution.state.phases[i].candidate_index));
    if (auto s = std::dynamic_pointer_cast<Solution>(phase))
      solution.phase_compositions.push_back(s->get_molar_fractions().matrix());
    else
      solution.phase_compositions.push_back(Eigen::VectorXd::Ones(1));
  }
  return solution;
}

std::shared_ptr<Assemblage> restore_solution(const GridSolution &solution,
                                             detail::Engine &engine) {
  const auto &state = solution.state;
  if (!std::isfinite(state.pressure) || !std::isfinite(state.temperature) ||
      state.pressure < 0. || state.temperature < 0. || state.phases.empty() ||
      state.phases.size() != solution.phase_bases.size() ||
      state.phases.size() != solution.phase_compositions.size() ||
      solution.molar_fractions.size() !=
          static_cast<Eigen::Index>(state.phases.size()) ||
      !solution.molar_fractions.allFinite() ||
      solution.molar_fractions.minCoeff() < -1.e-12 ||
      std::abs(solution.molar_fractions.sum() - 1.) > 1.e-8 ||
      !std::isfinite(solution.n_moles) || solution.n_moles <= 0.)
    throw std::invalid_argument("Invalid saved grid solution.");
  auto a = std::make_shared<Assemblage>();
  Eigen::ArrayXd amounts(static_cast<Eigen::Index>(state.phases.size()));
  for (std::size_t i = 0; i < state.phases.size(); ++i) {
    const auto &p = state.phases[i];
    const auto &basis = solution.phase_bases[i];
    const auto &composition = solution.phase_compositions[i];
    if (p.id < 0 ||
        p.candidate_index != p.id / engine.settings.max_phase_instances ||
        static_cast<std::size_t>(p.candidate_index) >= engine.phases.size() ||
        !std::isfinite(p.amount) || p.amount < -1.e-12 * solution.n_moles)
      throw std::invalid_argument(
          "Invalid saved grid phase identity or amount.");
    auto phase = detail::clone(
        engine.phases[static_cast<std::size_t>(p.candidate_index)].material);
    const auto n = engine.phases[static_cast<std::size_t>(p.candidate_index)]
                       .vertices.cols();
    if (basis.cols() != n || basis.rows() < 1 || basis.rows() > n ||
        composition.size() != basis.rows() || p.composition.size() != n ||
        !basis.allFinite() || !composition.allFinite() ||
        !p.composition.allFinite() ||
        std::abs(composition.sum() - 1.) > 1.e-8 ||
        (basis.transpose() * composition - p.composition).norm() > 1.e-8)
      throw std::invalid_argument(
          "Invalid saved grid phase basis or composition.");
    if (auto s = std::dynamic_pointer_cast<Solution>(phase)) {
      if (basis.rows() == n && basis.isIdentity(1.e-14))
        s->set_composition(composition.array());
      else {
        if (basis.fullPivLu().rank() != basis.rows())
          throw std::invalid_argument(
              "Saved grid phase basis is rank deficient.");
        phase = polytope::transform_solution_to_new_basis(
            *s, basis, composition.array(), p.name);
      }
    } else if (basis.rows() != 1 || !basis.isIdentity(1.e-14))
      throw std::invalid_argument("Invalid saved pure-phase basis.");
    phase->set_name(p.name);
    a->add_phases({phase});
    amounts[static_cast<Eigen::Index>(i)] = p.amount;
  }
  if (!std::isfinite(amounts.sum()) || amounts.sum() <= 0.)
    throw std::invalid_argument(
        "Saved grid solution has no positive phase amount.");
  if ((amounts - solution.molar_fractions * solution.n_moles).matrix().norm() >
      1.e-10 * solution.n_moles)
    throw std::invalid_argument("Saved grid phase amounts disagree with their "
                                "fractions and total moles.");
  a->set_fractions(solution.molar_fractions);
  a->set_n_moles(solution.n_moles);
  a->set_averaging_scheme(types::AveragingType::VRH);
  a->set_state(state.pressure, state.temperature);
  return a;
}
} // namespace

std::vector<std::string> thermodynamic_property_names() {
  std::vector<std::string> names;
  for (const auto &entry : getters)
    names.push_back(entry.first);
  return names;
}
std::map<std::string, double>
assemblage_properties(const Assemblage &a,
                      const std::vector<std::string> &names, bool relaxed) {
  validate_properties(names);
  for (Eigen::Index i = 0; i < a.get_n_phases(); ++i)
    if (a.get_phase<RelaxedSolution>(static_cast<std::size_t>(i)))
      throw std::invalid_argument(
          "assemblage_properties requires ordinary Solutions with explicit "
          "equilibrated populations; phase_diagram_grid unwraps "
          "RelaxedSolution "
          "candidates automatically.");
  Properties properties(a, relaxed);
  std::map<std::string, double> values;
  for (const auto &name : names)
    values[name] = properties.value(name);
  return values;
}

PropertyGrid PropertyGridder::evaluate(const std::vector<std::string> &names,
                                       const Eigen::VectorXd &x,
                                       const Eigen::VectorXd &y, bool relaxed,
                                       bool store_solutions) {
  const auto &source = source_;
  validate_properties(names);
  PropertyGrid result;
  result.diagram_type = source.section.type;
  result.relaxed = relaxed;
  result.x = x;
  result.y = y;
  result.context.pressure_range = source.pressure_range;
  result.context.temperature_range = source.temperature_range;
  result.context.section = source.section;
  result.context.composition_start = source.composition_start;
  result.context.settings = source.settings;
  result.context.phase_names = source.phase_names;
  if (store_solutions)
    result.solutions.resize(static_cast<std::size_t>(x.size() * y.size()));
  result.field_numbers = Eigen::MatrixXi::Zero(y.size(), x.size());
  result.success = result.field_numbers.cast<bool>().array();
  result.equilibrium_success = result.success;
  for (const auto &name : names)
    result.values[name] = Eigen::MatrixXd::Constant(
        y.size(), x.size(), std::numeric_limits<double>::quiet_NaN());
  result.diagnostics = geometry_.diagnostics;
  result.equilibrium_diagnostics = result.diagnostics;
  calculate(result, false);
  audit(result);
  return result;
}

void PropertyGridder::retry(PropertyGrid &grid) {
  if (grid.solutions.size() !=
      static_cast<std::size_t>(grid.x.size() * grid.y.size()))
    throw std::invalid_argument(
        "Reverse grid retries require store_solutions=True.");
  if (grid.diagram_type != source_.section.type ||
      grid.context.phase_names != source_.phase_names ||
      grid.context.composition_start != source_.composition_start)
    throw std::invalid_argument(
        "Saved grid does not match the source diagram.");
  calculate(grid, true);
  audit(grid, true);
}

void PropertyGridder::audit(PropertyGrid &grid, bool reverse, bool reset) {
  using namespace detail;
  const auto count = static_cast<std::size_t>(grid.x.size() * grid.y.size());
  if (grid.thermodynamic_samples.size() != count) {
    if (grid.solutions.size() != count)
      throw std::invalid_argument(
          "Thermodynamic audit requires retained equilibrium states.");
    grid.thermodynamic_samples.resize(count);
    for (std::size_t k = 0; k < count; ++k)
      if (auto &saved = grid.solutions[k]) {
        if (!std::isfinite(saved->gibbs)) {
          auto a = restore_solution(*saved, *engine_);
          saved->gibbs = a->get_n_moles() * a->get_molar_gibbs();
        }
        grid.thermodynamic_samples[k] = thermodynamic_sample(
            equilibrium_state(saved->state, saved->gibbs), source_.section);
      }
  }
  // Sweep-time messages can become stale when a later point is repaired.
  // Rebuild only consistency diagnostics from the final retained states.
  for (auto *messages : {&grid.diagnostics, &grid.equilibrium_diagnostics})
    erase_if(*messages, [](const auto &message) {
      return message.rfind("Thermodynamic audit: ", 0) == 0 ||
             message.find("Thermodynamic inconsistency with a neighbouring "
                          "equilibrium remains after recovery.") !=
                 std::string::npos;
    });
  if (reset || reverse != audit_reverse_) {
    audit_previous_.clear();
    audit_current_.clear();
    audit_row_.reset();
  }
  audit_reverse_ = reverse;
  auto &coverage = grid.thermodynamic_validation;
  coverage = {"grid_and_joins", 0, 0, 0};
  for (Eigen::Index row = 0; row < grid.y.size(); ++row) {
    const auto j = reverse ? grid.y.size() - 1 - row : row;
    if (!audit_row_ || *audit_row_ != grid.y[j]) {
      audit_previous_.clear();
      if (audit_row_ &&
          (reverse ? grid.y[j] < *audit_row_ : grid.y[j] > *audit_row_))
        audit_previous_ = std::move(audit_current_);
      audit_current_.clear();
      audit_row_ = grid.y[j];
    }
    for (Eigen::Index col = 0; col < grid.x.size(); ++col) {
      const auto i = reverse ? grid.x.size() - 1 - col : col;
      auto sample = grid.thermodynamic_samples[static_cast<std::size_t>(
          j * grid.x.size() + i)];
      sample.available = sample.available && grid.equilibrium_success(j, i);
      auto check = [&](const auto &other, std::size_t axis) {
        if (audit_thermodynamic_pair(sample, other, axis, coverage) > 0.) {
          const auto message =
              "Thermodynamic audit: grid [" + std::to_string(j) + ", " +
              std::to_string(i) +
              "] has an inconsistent final neighbour along axis " +
              std::to_string(axis) + ".";
          grid.diagnostics.push_back(message);
          grid.equilibrium_diagnostics.push_back(message);
        }
      };
      auto along = audit_current_.lower_bound(grid.x[i]);
      if (reverse) {
        if (along != audit_current_.end() && along->first == grid.x[i])
          ++along;
        if (along != audit_current_.end())
          check(along->second, 0);
      } else if (along != audit_current_.begin())
        check(std::prev(along)->second, 0);
      if (const auto above = audit_previous_.find(grid.x[i]);
          above != audit_previous_.end())
        check(above->second, 1);
      audit_current_[grid.x[i]] = std::move(sample);
    }
  }
  grid.resolved = grid.diagnostics.empty();
}

void PropertyGridder::calculate(PropertyGrid &result, bool reverse) {
  const auto &source = source_;
  const auto &geometry = geometry_;
  const auto &x = result.x;
  const auto &y = result.y;
  result.thermodynamic_samples.resize(
      static_cast<std::size_t>(x.size() * y.size()));
  const auto &origin = origin_;
  const auto &range = range_;
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const auto &coordinates = axis == 0 ? x : y;
    validate_grid_axis(coordinates, source.coordinate_ranges()[axis]);
    const auto &complete = grid_axes_[axis];
    if (!std::includes(complete.data(), complete.data() + complete.size(),
                       coordinates.data(),
                       coordinates.data() + coordinates.size()))
      throw std::invalid_argument(
          "Grid batches must use prepared grid coordinates.");
  }
  if (reverse != reverse_) {
    previous_row_.clear();
    current_row_.clear();
    row_coordinate_.reset();
    reverse_ = reverse;
  }
  auto &engine = *engine_;
  // Keep immutable preparation across batches, without carrying additional
  // chemical-potential or physical-coordinate guesses across their joins.
  // The same two rows of accepted equilibria still supply the warm starts.
  engine.potential_seed.resize(0);
  engine.physical_seed = physical_seed_;
  engine.equilibrium_solves = engine.minimization_calls = 0;
  if (!engine.free_vectors.empty())
    engine.set_bulk(0.);
  auto issue = [&](const std::string &message, bool equilibrium = true) {
    result.resolved = false;
    result.diagnostics.push_back(message);
    if (equilibrium)
      result.equilibrium_diagnostics.push_back(message);
  };
  auto within = [&](std::size_t f, const Eigen::Vector2d &q) {
    return (q.array() >= (fields_[f].low - 1.e-9 * range).array()).all() &&
           (q.array() <= (fields_[f].high + 1.e-9 * range).array()).all() &&
           contains(geometry.polygons[f], q, origin, range);
  };
  auto neighbours_at = [&](const Eigen::Vector2d &q) {
    std::vector<std::pair<const State *, std::size_t>> neighbours;
    auto along = current_row_.lower_bound(q[0]);
    if (reverse) {
      if (along != current_row_.end() && along->first == q[0])
        ++along;
      if (along != current_row_.end())
        neighbours.emplace_back(&along->second, 0);
    } else if (along != current_row_.begin())
      neighbours.emplace_back(&std::prev(along)->second, 0);
    if (const auto above = previous_row_.find(q[0]);
        above != previous_row_.end())
      neighbours.emplace_back(&above->second, 1);
    return neighbours;
  };
  for (Eigen::Index row = 0; row < y.size(); ++row) {
    const auto j = reverse ? y.size() - 1 - row : row;
    if (!row_coordinate_ || y[j] != *row_coordinate_) {
      previous_row_.clear();
      if (row_coordinate_ &&
          (reverse ? y[j] < *row_coordinate_ : y[j] > *row_coordinate_))
        previous_row_ = std::move(current_row_);
      current_row_.clear();
      row_coordinate_ = y[j];
    }
    for (Eigen::Index column = 0; column < x.size(); ++column) {
      const auto i = reverse ? x.size() - 1 - column : column;
      const Eigen::Vector2d q(x[i], y[j]);
      const auto neighbours = neighbours_at(q);
      auto consistent = [&](const State &state) {
        return std::none_of(
            neighbours.begin(), neighbours.end(), [&](const auto &n) {
              return detail::thermodynamic_violation(
                         state, *n.first, source.section, n.second) > 0.;
            });
      };
      if (reverse && result.equilibrium_success(j, i)) {
        auto &saved =
            result.solutions[static_cast<std::size_t>(j * x.size() + i)];
        if (saved) {
          // Legacy archives lack energy; replay it once without equilibration.
          if (!std::isfinite(saved->gibbs)) {
            auto a = restore_solution(*saved, engine);
            saved->gibbs = a->get_n_moles() * a->get_molar_gibbs();
          }
          auto state = detail::equilibrium_state(saved->state, saved->gibbs);
          result.thermodynamic_samples[static_cast<std::size_t>(j * x.size() +
                                                                i)] =
              detail::thermodynamic_sample(state, source.section);
          if (consistent(state)) {
            current_row_[q[0]] = std::move(state);
            const auto prefix = "grid [" + std::to_string(j) + ", " +
                                std::to_string(i) + "]: Thermodynamic";
            for (auto *messages :
                 {&result.diagnostics, &result.equilibrium_diagnostics})
              detail::erase_if(*messages, [&](const auto &message) {
                return message.find(prefix) != std::string::npos;
              });
            continue;
          }
        }
        result.equilibrium_success(j, i) = result.success(j, i) = false;
        saved.reset();
        for (auto &[name, values] : result.values)
          values(j, i) = std::numeric_limits<double>::quiet_NaN();
      }
      // Never fill model-domain masks, including their edges, by borrowing an
      // assemblage from a nearby valid field.
      if (std::any_of(source.excluded_regions.begin(),
                      source.excluded_regions.end(), [&](const auto &ring) {
                        return ring_location(ring, q, origin, range) >= 0;
                      }))
        continue;
      int assigned = -1;
      bool excluded = false;
      for (std::size_t f = 0; f < geometry.polygons.size(); ++f)
        if (within(f, q)) {
          const auto &field = geometry.polygons[f];
          excluded = excluded || field.outside_model_domain;
          if (!field.outside_model_domain &&
              (assigned < 0 ||
               field.field_number <
                   geometry.polygons[static_cast<std::size_t>(assigned)]
                       .field_number))
            assigned = static_cast<int>(f);
        }
      if (excluded)
        continue;
      if (assigned >= 0)
        result.field_numbers(j, i) =
            geometry.polygons[static_cast<std::size_t>(assigned)].field_number;
      const auto point =
          "grid [" + std::to_string(j) + ", " + std::to_string(i) + "]: ";
      if (reverse) {
        for (auto *messages :
             {&result.diagnostics, &result.equilibrium_diagnostics})
          messages->erase(std::remove_if(messages->begin(), messages->end(),
                                         [&](const auto &message) {
                                           return message.find(point) !=
                                                  std::string::npos;
                                         }),
                          messages->end());
      }
      std::string failure =
          "No saved equilibrium supplies a candidate assemblage.";
      std::shared_ptr<Assemblage> accepted;
      std::vector<int> accepted_ids;
      int chosen = 0;
      if (!engine.free_vectors.empty())
        engine.set_bulk(q[1]);
      auto attempt = [&](std::size_t f, bool check_competitors) {
        const auto &field = geometry.polygons[f];
        auto &cached = fields_[f];
        if (field.outside_model_domain || field.phases.empty())
          return false;
        std::vector<const State *> starts;
        for (const auto *row : {&current_row_, &previous_row_})
          if (const auto *seed = neighbour_seed(*row, field.phases, q,
                                                source.section.type, range))
            starts.push_back(seed);
        std::sort(starts.begin(), starts.end(),
                  [&](const auto *a, const auto *b) {
                    return (diagram_coordinates(*a, source.section.type) - q)
                               .cwiseQuotient(range)
                               .squaredNorm() <
                           (diagram_coordinates(*b, source.section.type) - q)
                               .cwiseQuotient(range)
                               .squaredNorm();
                  });
        // A nearby equilibrium's P,T matters as well as its compositions in
        // inverse S/V sections; a failed Newton iterate must never replace it.
        auto solve_from = [&](const State &seed) {
          try {
            accepted = equilibrate_grid_candidate(engine, field.phases, seed, q,
                                                  check_competitors);
            chosen = field.field_number;
            accepted_ids = field.phases;
            return true;
          } catch (const std::exception &e) {
            failure = e.what();
            return false;
          }
        };
        for (const auto *seed : starts)
          if (solve_from(*seed))
            return true;
        if (!cached.seed_ready) {
          cached.seed = field_seed(source, field, field.label_position, range);
          cached.seed_ready = true;
        }
        if (cached.seed && solve_from(*cached.seed))
          return true;
        // Saved samples/boundary states can supply another basin when neither
        // neighbouring grid point converges. No phase discovery is performed.
        auto nearby = field_seed(source, field, q, range);
        return nearby && solve_from(*nearby);
      };
      if (assigned >= 0) {
        const auto f = static_cast<std::size_t>(assigned);
        attempt(f, false);
      }
      if (!accepted && assigned >= 0) {
        // Only cross an assigned field's boundary when it meets the window
        // reaching one grid point in each direction. Use the complete axes,
        // including for singleton batches and nonuniform grids.
        std::array<std::array<double, 2>, 2> window;
        for (std::size_t axis = 0; axis < 2; ++axis) {
          const auto &coordinates = grid_axes_[axis];
          const double value = q[static_cast<Eigen::Index>(axis)];
          const auto k =
              std::lower_bound(coordinates.data(),
                               coordinates.data() + coordinates.size(), value) -
              coordinates.data();
          window[axis] = {value, value};
          if (coordinates.size() > 1) {
            window[axis][0] =
                k > 0 ? coordinates[k - 1] : 2. * value - coordinates[1];
            window[axis][1] = k + 1 < coordinates.size()
                                  ? coordinates[k + 1]
                                  : 2. * value - coordinates[k - 1];
          }
        }
        std::vector<std::pair<double, std::size_t>> nearby;
        for (const auto &line : source.boundaries) {
          const auto sharing = boundary_fields_.find(line.id);
          if (sharing == boundary_fields_.end() ||
              std::find(sharing->second.begin(), sharing->second.end(),
                        static_cast<std::size_t>(assigned)) ==
                  sharing->second.end())
            continue;
          double distance = std::numeric_limits<double>::infinity();
          for (std::size_t k = 1; k < line.points.size(); ++k) {
            const auto a =
                diagram_coordinates(line.points[k - 1], source.section.type);
            const auto b =
                diagram_coordinates(line.points[k], source.section.type);
            if (detail::intersects_region(a, b, window))
              distance = std::min(distance, detail::segment_distance(
                                                Eigen::Vector2d::Zero(),
                                                (a - q).cwiseQuotient(range),
                                                (b - q).cwiseQuotient(range)));
          }
          if (std::isfinite(distance))
            for (auto f : sharing->second)
              if (static_cast<int>(f) != assigned)
                nearby.emplace_back(distance, f);
        }
        std::sort(nearby.begin(), nearby.end());
        std::vector<std::vector<int>> tried{
            geometry.polygons[static_cast<std::size_t>(assigned)].phases};
        for (const auto &[distance, f] : nearby) {
          (void)distance;
          const auto &ids = geometry.polygons[f].phases;
          if (std::find(tried.begin(), tried.end(), ids) != tried.end())
            continue;
          tried.push_back(ids);
          if (attempt(f, true))
            break;
        }
      }
      if (!accepted) {
        issue(point + failure);
        continue;
      }
      auto state = detail::equilibrium_state(engine, *accepted, accepted_ids);
      if (!consistent(state)) {
        std::vector<State> starts;
        for (const auto &n : neighbours)
          starts.push_back(*n.first);
        auto recovered = detail::recover_thermodynamic_state(
            engine, {state, accepted, accepted_ids}, starts);
        if (!recovered.state.success) {
          issue(point +
                "Thermodynamic recovery failed: " + recovered.state.message);
          result.equilibrium_success(j, i) = result.success(j, i) = false;
          continue;
        }
        accepted = std::move(recovered.assemblage);
        accepted_ids = std::move(recovered.ids);
        state = std::move(recovered.state);
        const auto active = engine.canonical_ids(engine.active_phases(state));
        const auto previous_field = std::find_if(
            geometry.polygons.begin(), geometry.polygons.end(),
            [&](const auto &f) { return f.field_number == chosen; });
        if (previous_field == geometry.polygons.end() ||
            engine.canonical_ids(previous_field->phases) != active) {
          chosen = 0;
          double nearest = std::numeric_limits<double>::infinity();
          for (const auto &field : geometry.polygons)
            if (engine.canonical_ids(field.phases) == active) {
              const double distance =
                  (field.label_position - q).cwiseQuotient(range).squaredNorm();
              if (distance < nearest) {
                nearest = distance;
                chosen = field.field_number;
              }
            }
          if (!chosen)
            issue(point + "Recovered stable assemblage has no field in the "
                          "saved diagram.");
        }
        if (!consistent(state))
          issue(point + "Thermodynamic inconsistency with a neighbouring "
                        "equilibrium remains after recovery.");
      }
      result.field_numbers(j, i) = chosen;
      result.thermodynamic_samples[static_cast<std::size_t>(j * x.size() + i)] =
          detail::thermodynamic_sample(state, source.section);
      current_row_[q[0]] = std::move(state);
      if (!result.solutions.empty())
        result.solutions[static_cast<std::size_t>(j * x.size() + i)] =
            capture_solution(*accepted, engine, accepted_ids);
      result.equilibrium_success(j, i) = true;
      result.success(j, i) = true;
      Properties properties(*accepted, result.relaxed);
      for (auto &[name, values] : result.values) {
        try {
          values(j, i) = properties.value(name);
        } catch (const std::exception &e) {
          result.success(j, i) = false;
          issue("Field " + std::to_string(chosen) + ": " + point + name + ": " +
                    e.what(),
                false);
        }
      }
    }
  }
  result.equilibrium_solves += engine.equilibrium_solves;
  result.resolved = result.diagnostics.empty();
}

PropertyGrid
phase_diagram_grid(const Result &source,
                   const std::vector<std::shared_ptr<Material>> &candidates,
                   const std::vector<std::string> &names,
                   const Eigen::VectorXd &x, const Eigen::VectorXd &y,
                   bool relaxed, bool store_solutions) {
  PropertyGridder gridder(source, candidates, x, y);
  auto grid = gridder.evaluate(names, x, y, relaxed, true);
  gridder.retry(grid);
  if (!store_solutions) {
    grid.solutions.clear();
    grid.solutions.shrink_to_fit();
  }
  return grid;
}

PropertyGridder::PropertyGridder(
    Result source, std::vector<std::shared_ptr<Material>> candidates,
    Eigen::VectorXd x, Eigen::VectorXd y)
    : source_(std::move(source)), geometry_(field_polygons(source_)),
      grid_axes_{std::move(x), std::move(y)} {
  validate_candidates(source_, candidates);
  const auto bounds = source_.coordinate_ranges();
  origin_ << bounds[0][0], bounds[1][0];
  range_ << bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0];
  if (!origin_.allFinite() || !range_.allFinite() || range_.minCoeff() <= 0.)
    throw std::invalid_argument(
        "Grid diagram ranges must be finite and increasing.");
  for (std::size_t axis = 0; axis < 2; ++axis)
    validate_grid_axis(grid_axes_[axis], bounds[axis]);
  engine_ = std::make_unique<detail::Engine>(
      source_.composition_start, candidates, source_.settings, source_.section);
  engine_->pressure_bounds = source_.pressure_range;
  engine_->temperature_bounds = source_.temperature_range;
  physical_seed_ = engine_->physical_seed;
  fields_.resize(geometry_.polygons.size());
  for (std::size_t f = 0; f < geometry_.polygons.size(); ++f) {
    fields_[f].low =
        geometry_.polygons[f].vertices.colwise().minCoeff().transpose();
    fields_[f].high =
        geometry_.polygons[f].vertices.colwise().maxCoeff().transpose();
    for (int id : geometry_.polygons[f].boundary_ids)
      boundary_fields_[id].push_back(f);
  }
}

PropertyGrid
evaluate_property_grid(const PropertyGrid &saved,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const std::vector<std::string> &names, bool relaxed,
                       types::AveragingType averaging_scheme) {
  validate_properties(names);
  if (saved.solutions.empty())
    throw std::invalid_argument(
        "Grid has no saved solutions; calculate with store_solutions=True.");
  if (saved.solutions.size() !=
      static_cast<std::size_t>(saved.x.size() * saved.y.size()))
    throw std::invalid_argument("Saved grid solutions do not match its shape.");
  validate_candidates(saved.context, candidates);
  detail::Engine engine(saved.context.composition_start, candidates,
                        saved.context.settings, saved.context.section);
  PropertyGrid result = saved;
  result.relaxed = relaxed;
  result.equilibrium_solves = 0;
  result.diagnostics = saved.equilibrium_diagnostics;
  result.resolved = result.diagnostics.empty();
  result.success.setConstant(false);
  result.values.clear();
  for (const auto &name : names)
    result.values[name] =
        Eigen::MatrixXd::Constant(saved.y.size(), saved.x.size(),
                                  std::numeric_limits<double>::quiet_NaN());
  auto issue = [&](const std::string &message) {
    result.diagnostics.push_back(message);
    result.resolved = false;
  };
  for (Eigen::Index j = 0; j < saved.y.size(); ++j)
    for (Eigen::Index i = 0; i < saved.x.size(); ++i) {
      const auto &solution =
          saved.solutions[static_cast<std::size_t>(j * saved.x.size() + i)];
      if (!solution)
        continue;
      const auto prefix = "Field " + std::to_string(saved.field_numbers(j, i)) +
                          ": grid [" + std::to_string(j) + ", " +
                          std::to_string(i) + "]: ";
      try {
        auto a = restore_solution(*solution, engine);
        a->set_averaging_scheme(averaging_scheme);
        if (!engine.free_vectors.empty())
          engine.set_bulk(solution->state.composition_coordinate);
        if (engine.mass_error(*a, false) >
            saved.context.settings.mass_balance_tolerance * 2.)
          throw std::invalid_argument(
              "Saved grid solution does not match its bulk composition.");
        Properties properties(*a, relaxed);
        properties.check_stability();
        result.success(j, i) = true;
        for (const auto &name : names) {
          try {
            result.values.at(name)(j, i) = properties.value(name);
          } catch (const std::exception &e) {
            result.success(j, i) = false;
            issue(prefix + name + ": " + e.what());
          }
        }
      } catch (const std::exception &e) {
        issue(prefix + e.what());
      }
    }
  return result;
}
} // namespace burnman::phase_diagrams
