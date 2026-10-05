/* GPL v3 or later. Native Gibbs minimisation and equilibrium validation. */
#include "burnman/eos/slb.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/index_utils.hpp"
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <nlopt.hpp>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
namespace burnman::pseudosections::detail {
std::shared_ptr<Material> clone(const std::shared_ptr<Material> &phase) {
  if (auto s = std::dynamic_pointer_cast<Solution>(phase)) {
    std::shared_ptr<Solution> out;
    if (auto face = std::dynamic_pointer_cast<FaceSolution>(phase))
      out = std::make_shared<FaceSolution>(*s, face->original_basis);
    else
      out = std::make_shared<Solution>(*s);
    auto m = s->get_solution_model();
    if (auto r =
            dynamic_cast<const solution_models::SymmetricRegularSolution *>(
                m.get()))
      out->set_solution_model(
          std::make_shared<solution_models::SymmetricRegularSolution>(*r));
    else if (auto asymmetric = dynamic_cast<
                 const solution_models::AsymmetricRegularSolution *>(m.get()))
      out->set_solution_model(
          std::make_shared<solution_models::AsymmetricRegularSolution>(
              *asymmetric));
    else if (const auto *model = m.get();
             typeid(*model) == typeid(solution_models::IdealSolution))
      out->set_solution_model(std::make_shared<solution_models::IdealSolution>(
          *static_cast<const solution_models::IdealSolution *>(model)));
    else
      throw std::invalid_argument("Pseudosections support ideal and "
                                  "symmetric/asymmetric regular solutions.");
    out->reset_cache();
    return out;
  }
  if (auto face = std::dynamic_pointer_cast<FaceMineral>(phase))
    return std::make_shared<FaceMineral>(*face, face->original_basis);
  if (auto m = std::dynamic_pointer_cast<Mineral>(phase))
    return std::make_shared<Mineral>(*m);
  throw std::invalid_argument(
      "Pseudosection candidates must be minerals or solutions.");
}
ConstraintList constraints(std::unique_ptr<EqualityConstraint> a,
                           std::unique_ptr<EqualityConstraint> b) {
  ConstraintList c(2);
  c[0].push_back(std::move(a));
  c[1].push_back(std::move(b));
  return c;
}
Engine::Engine(const types::FormulaMap &composition,
               const std::vector<std::shared_ptr<Material>> &candidates,
               const Settings &opts)
    : bulk(composition), settings(opts) {
  if (candidates.empty() || bulk.empty())
    throw std::invalid_argument(
        "Supply a nonempty elemental bulk and candidate phases.");
  if (settings.pressure_seeds < 2 || settings.temperature_seeds < 2 ||
      settings.max_refinement_iterations < 1 ||
      settings.minimization_starts < 1 || settings.max_phase_instances < 1 ||
      settings.max_trace_steps < 2 || settings.max_lines < 1 ||
      settings.max_recovery_passes < 0 || settings.max_recovery_passes > 5 ||
      !(std::isfinite(settings.step) && settings.step > 0 &&
        settings.step <= .2 && std::isfinite(settings.min_step) &&
        settings.min_step > 0 && settings.min_step < settings.step) ||
      !(std::isfinite(settings.affinity_tolerance) &&
        settings.affinity_tolerance > 0) ||
      !(std::isfinite(settings.mass_balance_tolerance) &&
        settings.mass_balance_tolerance > 0) ||
      !(std::isfinite(settings.amount_tolerance) &&
        settings.amount_tolerance > 0 && settings.amount_tolerance < .01) ||
      !(std::isfinite(settings.composition_tolerance) &&
        settings.composition_tolerance > 0) ||
      !(std::isfinite(settings.node_tolerance) && settings.node_tolerance > 0 &&
        settings.node_tolerance < .1))
    throw std::invalid_argument("Invalid pseudosection settings.");
  std::set<std::string> names, els;
  double sum = 0;
  for (auto &[el, n] : bulk) {
    if (!std::isfinite(n) || n < 0)
      throw std::invalid_argument(
          "Bulk amounts must be finite and nonnegative.");
    els.insert(el);
    sum += n;
  }
  if (sum <= 0)
    throw std::invalid_argument("Bulk must have positive total amount.");
  for (auto &input : candidates) {
    if (!input)
      throw std::invalid_argument("Null phase candidate.");
    if (!names.insert(input->get_name()).second)
      throw std::invalid_argument("Candidate phase names must be unique; "
                                  "solution copies are generated internally.");
    Phase phase;
    phase.material = clone(input);
    phase.solution = std::dynamic_pointer_cast<Solution>(phase.material);
    std::vector<types::FormulaMap> formulae;
    if (phase.solution) {
      auto m = phase.solution->get_solution_model();
      for (auto &em : m->endmembers)
        formulae.push_back(em.get_formula());
      phase.occupancies = m->get_endmember_occupancies().matrix();
      phase.vertices = polytope::solution_polytope_from_endmember_occupancies(
                           phase.occupancies)
                           .get_vertices();
    } else {
      formulae.push_back(phase.material->get_formula());
      phase.vertices = Eigen::MatrixXd::Ones(1, 1);
      phase.occupancies = phase.vertices;
    }
    for (auto &f : formulae)
      for (auto &[el, n] : f)
        els.insert(el);
    phases.push_back(std::move(phase));
  }
  elements.assign(els.begin(), els.end());
  full_bulk = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(elements.size()));
  for (auto &name : settings.required_eos_phases)
    if (!names.count(name))
      throw std::invalid_argument("Required EOS phase is not a candidate: " +
                                  name);
  if (!settings.required_eos_phases.empty() && !settings.exclude_invalid_eos)
    throw std::invalid_argument(
        "required_eos_phases requires exclude_invalid_eos.");
  for (std::size_t k = 0; k < elements.size(); ++k)
    if (bulk.count(elements[k]))
      full_bulk[static_cast<Eigen::Index>(k)] = bulk.at(elements[k]);
  Eigen::Index rows = 0;
  for (auto &phase : phases)
    rows += phase.vertices.cols();
  Eigen::MatrixXd full(rows, elements.size());
  Eigen::Index offset = 0;
  for (auto &phase : phases) {
    phase.a = Eigen::MatrixXd::Zero(phase.vertices.cols(),
                                    static_cast<Eigen::Index>(elements.size()));
    for (int i = 0; i < phase.a.rows(); ++i) {
      auto f = phase.solution ? phase.solution->get_solution_model()
                                    ->endmembers[static_cast<std::size_t>(i)]
                                    .get_formula()
                              : phase.material->get_formula();
      for (std::size_t k = 0; k < elements.size(); ++k)
        if (f.count(elements[k]))
          phase.a(i, static_cast<Eigen::Index>(k)) = f.at(elements[k]);
    }
    full.middleRows(offset, phase.a.rows()) = phase.a;
    offset += phase.a.rows();
  }
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(full);
  qr.setThreshold(1.e-10);
  for (int k = 0; k < qr.rank(); ++k)
    components.push_back(qr.colsPermutation().indices()[k]);
  // The bulk must lie in the linear span before any expensive minimisation.
  auto coefficients = full.transpose()
                          .completeOrthogonalDecomposition()
                          .solve(full_bulk)
                          .eval();
  if ((full.transpose() * coefficients - full_bulk).norm() >
      settings.mass_balance_tolerance * full_bulk.norm())
    throw std::invalid_argument("Bulk is outside the candidate chemical span.");
  b.resize(static_cast<Eigen::Index>(components.size()));
  for (std::size_t k = 0; k < components.size(); ++k)
    b[static_cast<Eigen::Index>(k)] = full_bulk[components[k]];
}
void Engine::set_pt(double p, double t) {
  for (auto &phase : phases) {
    phase.available = true;
    phase.domain_error.clear();
    try {
      phase.material->set_state(p, t);
      phase.g.resize(phase.vertices.cols());
      if (phase.solution) {
        for (int i = 0; i < phase.g.size(); ++i) {
          auto m = phase.solution->get_solution_model()
                       ->endmembers[static_cast<std::size_t>(i)];
          m.set_state(p, t);
          phase.g[i] = m.get_molar_gibbs();
        }
      } else
        phase.g[0] = phase.material->get_molar_gibbs();
    } catch (const eos::SLBDomainError &error) {
      if (!settings.exclude_invalid_eos)
        throw;
      phase.available = false;
      phase.domain_error = phase.material->get_name() + ": " + error.what();
    }
  }
}
bool Engine::eos_bulk_feasible(double p, double t) {
  set_pt(p, t);
  Eigen::Index count = 0;
  for (auto &ph : phases)
    if (ph.available)
      count += ph.vertices.rows();
  for (auto &ph : phases)
    if (!ph.available && std::find(settings.required_eos_phases.begin(),
                                   settings.required_eos_phases.end(),
                                   ph.material->get_name()) !=
                             settings.required_eos_phases.end())
      return false;
  if (!count)
    return false;
  Eigen::MatrixXd a(count, components.size());
  Eigen::Index row = 0;
  for (auto &ph : phases)
    if (ph.available)
      for (int i = 0; i < ph.vertices.rows(); ++i) {
        Eigen::VectorXd formula =
            ph.a.transpose() * ph.vertices.row(i).transpose();
        for (std::size_t j = 0; j < components.size(); ++j)
          a(row, static_cast<Eigen::Index>(j)) = formula[components[j]];
        ++row;
      }
  try {
    polytope::gibbs_linear_program(a, Eigen::VectorXd::Zero(count), b);
    return true;
  } catch (const polytope::InfeasibleBulk &) {
    return false;
  }
}
double Engine::energy(int index, const Eigen::VectorXd &p) const {
  auto &ph = phases[static_cast<std::size_t>(index)];
  double g = ph.g.dot(p);
  if (ph.solution)
    g += ph.solution->get_solution_model()->compute_excess_gibbs_free_energy(
        ph.material->get_pressure(), ph.material->get_temperature(), p.array());
  return g;
}
namespace {
struct Objective {
  const Phase *phase;
  Eigen::VectorXd adjusted;
  double scale;
};
Eigen::VectorXd unpack(unsigned n, const double *x) {
  Eigen::VectorXd p(n + 1);
  p.tail(n) = Eigen::Map<const Eigen::VectorXd>(x, n);
  p[0] = 1. - p.tail(n).sum();
  return p;
}
double objective(const std::vector<double> &x, std::vector<double> &grad,
                 void *ptr) {
  auto &d = *static_cast<Objective *>(ptr);
  auto p =
      unpack(static_cast<unsigned int>(burnman::utils::checked_int(x.size())),
             x.data());
  auto &ph = *d.phase;
  double v = d.adjusted.dot(p);
  if (ph.solution) {
    auto m = ph.solution->get_solution_model();
    double P = ph.material->get_pressure(), T = ph.material->get_temperature();
    v += m->compute_excess_gibbs_free_energy(P, T, p.array());
    if (!grad.empty()) {
      auto mu = (d.adjusted.array() +
                 m->compute_excess_partial_gibbs_free_energies(P, T, p.array()))
                    .eval();
      for (std::size_t i = 0; i < x.size(); ++i)
        grad[i] = (mu[static_cast<Eigen::Index>(i + 1)] - mu[0]) / d.scale;
    }
  }
  return v / d.scale;
}
void site_constraints(unsigned m, double *result, unsigned n, const double *x,
                      double *grad, void *ptr) {
  auto &o = *static_cast<const Eigen::MatrixXd *>(ptr);
  auto p = unpack(n, x);
  for (unsigned k = 0; k < m; ++k) {
    result[k] = -o.col(k).dot(p);
    if (grad)
      for (unsigned i = 0; i < n; ++i)
        grad[k * n + i] = -(o(i + 1, k) - o(0, k));
  }
}
} // namespace
Minimum Engine::minimize(int index, const Eigen::VectorXd &mu,
                         const Eigen::VectorXd &start) {
  auto &ph = phases[static_cast<std::size_t>(index)];
  Eigen::VectorXd adj = ph.g;
  for (int j = 0; j < adj.size(); ++j)
    for (std::size_t k = 0; k < components.size(); ++k)
      adj[j] -= ph.a(j, components[k]) * mu[static_cast<Eigen::Index>(k)];
  if (!ph.solution || ph.g.size() == 1)
    return {Eigen::VectorXd::Ones(1), adj[0]};
  ++minimization_calls;
  Eigen::VectorXd p =
      start.size() ? start : ph.vertices.colwise().mean().transpose();
  Objective data{&ph, adj,
                 std::max(1000., 8.314 * ph.material->get_temperature())};
  nlopt::opt opt(
      nlopt::LD_SLSQP,
      static_cast<unsigned int>(burnman::utils::checked_int(ph.g.size() - 1)));
  opt.set_min_objective(objective, &data);
  std::vector<double> low, high, x;
  for (int i = 1; i < ph.g.size(); ++i) {
    low.push_back(ph.vertices.col(i).minCoeff());
    high.push_back(ph.vertices.col(i).maxCoeff());
    x.push_back(p[i]);
  }
  opt.set_lower_bounds(low);
  opt.set_upper_bounds(high);
  opt.add_inequality_mconstraint(
      site_constraints, &ph.occupancies,
      std::vector<double>(static_cast<std::size_t>(ph.occupancies.cols()),
                          1.e-10));
  opt.set_xtol_abs(settings.composition_tolerance * .01);
  opt.set_ftol_abs(1.e-11);
  opt.set_maxeval(250);
  double value = 0.;
  try {
    opt.optimize(x, value);
  } catch (const nlopt::roundoff_limited &) {
  } // validate below
  p = unpack(static_cast<unsigned int>(burnman::utils::checked_int(x.size())),
             x.data());
  if (!p.allFinite() || (ph.occupancies.transpose() * p).minCoeff() < -1.e-7)
    throw std::runtime_error(
        "Solution minimisation returned an infeasible composition.");
  std::vector<double> empty;
  value = objective(x, empty, &data) * data.scale;
  return {p, value};
}
std::vector<Minimum> Engine::minima(int index, const Eigen::VectorXd &mu,
                                    const Eigen::VectorXd &start) {
  auto &ph = phases[static_cast<std::size_t>(index)];
  if (!ph.available)
    return {{ph.vertices.colwise().mean().transpose(),
             std::numeric_limits<double>::infinity()}};
  if (!ph.solution || ph.g.size() == 1)
    return {minimize(index, mu, start)};
  auto mean = ph.vertices.colwise().mean().transpose().eval();
  std::vector<std::pair<double, Eigen::VectorXd>> seeds;
  if (start.size())
    seeds.push_back({-std::numeric_limits<double>::infinity(), start});
  seeds.push_back({-std::numeric_limits<double>::max(), mean});
  Eigen::VectorXd adjusted = ph.g;
  for (int i = 0; i < adjusted.size(); ++i)
    for (std::size_t k = 0; k < components.size(); ++k)
      adjusted[i] -= ph.a(i, components[k]) * mu[static_cast<Eigen::Index>(k)];
  for (int i = 0; i < ph.vertices.rows(); ++i) {
    Eigen::VectorXd p = .995 * ph.vertices.row(i).transpose() + .005 * mean;
    double value =
        adjusted.dot(p) +
        ph.solution->get_solution_model()->compute_excess_gibbs_free_energy(
            ph.material->get_pressure(), ph.material->get_temperature(),
            p.array());
    seeds.push_back({value, p});
  }
  std::stable_sort(seeds.begin(), seeds.end(), [](auto &left, auto &right) {
    return left.first < right.first;
  });
  std::vector<Minimum> out;
  for (int i = 0; i < std::min<int>(settings.minimization_starts,
                                    burnman::utils::checked_int(seeds.size()));
       ++i) {
    Minimum m;
    try {
      m = minimize(index, mu, seeds[static_cast<std::size_t>(i)].second);
    } catch (const std::runtime_error &error) {
      // NLopt can exhaust its internal SQP loop at a vertex, especially at
      // zero kelvin. Retry away from that singular starting point. A failed
      // minimisation is never treated as an infinite (stable) phase affinity.
      try {
        m = minimize(index, mu,
                     .5 * seeds[static_cast<std::size_t>(i)].second +
                         .5 * mean);
      } catch (const std::runtime_error &) {
        if (settings.verbose)
          std::cerr << "Minimum start failed for " << ph.material->get_name()
                    << ": " << error.what() << '\n';
        continue;
      }
    }
    bool exists = false;
    for (auto &old : out)
      if ((old.p - m.p).norm() < settings.composition_tolerance * 10) {
        if (m.affinity < old.affinity)
          old = m;
        exists = true;
        break;
      }
    if (!exists)
      out.push_back(m);
  }
  if (out.empty())
    throw std::runtime_error("All composition minimisation starts failed for " +
                             ph.material->get_name() + ".");
  std::sort(out.begin(), out.end(), [](auto &left, auto &right) {
    return left.affinity < right.affinity;
  });
  return out;
}
std::shared_ptr<Assemblage>
Engine::make_assemblage(const std::vector<int> &ids,
                        const std::vector<PhaseState> &states, double p,
                        double t, double face_tolerance) {
  auto a = std::make_shared<Assemblage>();
  Eigen::ArrayXd amounts(ids.size());
  for (std::size_t k = 0; k < ids.size(); ++k) {
    int base = ids[k] / settings.max_phase_instances;
    auto m = clone(phases.at(static_cast<std::size_t>(base)).material);
    auto s = std::dynamic_pointer_cast<Solution>(m);
    const PhaseState *init = nullptr;
    for (auto &old : states)
      if (old.id == ids[k]) {
        init = &old;
        break;
      }
    if (s) {
      Eigen::VectorXd x = init ? init->composition
                               : phases[static_cast<std::size_t>(base)]
                                     .vertices.colwise()
                                     .mean()
                                     .transpose()
                                     .eval();
      s->set_composition(x.array());
      if (settings.active_solution_faces && init) {
        auto &ph = phases[static_cast<std::size_t>(base)];
        auto sites = (ph.occupancies.transpose() * x).eval();
        std::vector<int> vertices;
        for (int row = 0; row < ph.vertices.rows(); ++row) {
          bool inside = true;
          for (int col = 0; col < sites.size(); ++col)
            if (std::abs(sites[col]) < face_tolerance &&
                std::abs(ph.vertices.row(row).dot(ph.occupancies.col(col))) >
                    1.e-10)
              inside = false;
          if (inside)
            vertices.push_back(row);
        }
        if (!vertices.empty()) {
          Eigen::MatrixXd face(vertices.size(), x.size());
          for (std::size_t i = 0; i < vertices.size(); ++i)
            face.row(static_cast<Eigen::Index>(i)) =
                ph.vertices.row(vertices[i]);
          Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(face.transpose());
          qr.setThreshold(1.e-10);
          if (qr.rank() < x.size()) {
            Eigen::MatrixXd basis(qr.rank(), x.size());
            for (int i = 0; i < qr.rank(); ++i)
              basis.row(i) = face.row(qr.colsPermutation().indices()[i]);
            Eigen::VectorXd q =
                basis.transpose().completeOrthogonalDecomposition().solve(x);
            q /= q.sum();
            if ((basis.transpose() * q - x).norm() < 1.e-6) {
              auto reduced = polytope::transform_solution_to_new_basis(
                  *s, basis, q.array(), m->get_name());
              if (auto rs = std::dynamic_pointer_cast<Solution>(reduced))
                m = std::make_shared<FaceSolution>(*rs, basis);
              else
                m = std::make_shared<FaceMineral>(
                    *std::dynamic_pointer_cast<Mineral>(reduced), basis);
            }
          }
        }
      }
    }
    if (ids[k] % settings.max_phase_instances)
      m->set_name(m->get_name() + " #" +
                  std::to_string(ids[k] % settings.max_phase_instances + 1));
    a->add_phases({m});
    amounts[static_cast<Eigen::Index>(k)] =
        init ? std::max(0., init->amount) : 0.;
  }
  if (amounts.sum() == 0)
    amounts.setOnes();
  a->set_fractions(amounts / amounts.sum());
  a->set_n_moles(amounts.sum());
  a->set_state(p, t);
  return a;
}
std::shared_ptr<Assemblage>
Engine::copy_assemblage(const Assemblage &input) const {
  auto out = std::make_shared<Assemblage>();
  for (int i = 0; i < input.get_n_phases(); ++i)
    out->add_phases({clone(input.get_phase(static_cast<std::size_t>(i)))});
  out->set_fractions(input.get_molar_fractions());
  out->set_n_moles(input.get_n_moles());
  out->set_state(input.get_pressure(), input.get_temperature());
  return out;
}
optim::roots::DampedNewtonResult Engine::solve(Assemblage &a,
                                               ConstraintList &c) {
  auto prm = get_equilibration_parameters(a, bulk, {});
  auto initial = get_parameter_vector(a, 0);
  // Trace site fractions can be orders of magnitude smaller than 1e-9.
  // Tight composition steps are needed to resolve their chemical potentials.
  Eigen::VectorXd tol = Eigen::VectorXd::Constant(prm.n_parameters, 1.e-12);
  for (int k = 0; k < prm.phase_amount_indices.size(); ++k)
    tol[prm.phase_amount_indices[k]] = 1.e-9;
  tol[0] = .2;
  tol[1] = 1.e-6;
  Eigen::VectorXd scales = Eigen::VectorXd::Ones(prm.n_parameters);
  scales[0] = 1.e9;
  scales[1] = 1000.;
  for (int k = 0; k < prm.phase_amount_indices.size(); ++k)
    scales[prm.phase_amount_indices[k]] = std::max(1.e-3, a.get_n_moles());
  auto verify = [&](optim::roots::DampedNewtonResult &s) {
    if (s.x.allFinite()) {
      set_composition_and_state_from_parameters(a, s.x);
      double reactions = a.get_n_reactions()
                             ? a.get_reaction_affinities().cwiseAbs().maxCoeff()
                             : 0.;
      bool verified = reactions <= settings.affinity_tolerance * .1 &&
                      mass_error(a) <= settings.mass_balance_tolerance;
      auto inequalities = calculate_constraints(a, 0);
      verified =
          verified &&
          (inequalities.first * s.x + inequalities.second).maxCoeff() <= 1.e-9;
      for (auto &group : c) {
        auto &constraint = group[0];
        auto derivative = constraint->derivative(s.x, a, s.x.size());
        double allowance =
            std::max(1.e-10, derivative.cwiseAbs().dot(tol) * 2.);
        verified =
            verified && std::abs(constraint->evaluate(s.x, a)) <= allowance;
      }
      if (verified && !s.success) {
        s.success = true;
        s.code = 0;
        s.message = "Equilibrium verified from reaction, mass, equality and "
                    "site-bound residuals.";
      } else if (!verified)
        s.success = false;
    }
    return s.success;
  };
  // Preserve the established continuation path when it verifies. Retry from
  // the original accepted state if an ill-conditioned constraint walk fails.
  try {
    ++equilibrium_solves;
    auto result =
        equilibrate(bulk, a, c, {}, 1.e-9, false, false, 150, false, tol);
    auto s = result.sol_array(0);
    if (verify(s))
      return s;
  } catch (const std::exception &) {
  }
  set_composition_and_state_from_parameters(a, initial);
  ++equilibrium_solves;
  auto result =
      equilibrate(bulk, a, c, {}, 1.e-9, false, false, 150, false, tol, scales);
  auto s = result.sol_array(0);
  verify(s);
  return s;
}
Eigen::VectorXd Engine::potentials(const Assemblage &a) const {
  auto stoich = a.get_stoichiometric_matrix();
  auto elems = a.get_elements();
  Eigen::MatrixXd reduced(stoich.rows(), components.size());
  for (std::size_t k = 0; k < components.size(); ++k) {
    auto it = std::find(elems.begin(), elems.end(),
                        elements[static_cast<std::size_t>(components[k])]);
    if (it == elems.end())
      reduced.col(static_cast<Eigen::Index>(k)).setZero();
    else
      reduced.col(static_cast<Eigen::Index>(k)) =
          stoich.col(it - elems.begin());
  }
  Eigen::VectorXd prior =
      potential_seed.size() == static_cast<Eigen::Index>(components.size())
          ? potential_seed
          : Eigen::VectorXd::Zero(static_cast<Eigen::Index>(components.size()));
  return prior + reduced.completeOrthogonalDecomposition().solve(
                     a.get_partial_gibbs().matrix() - reduced * prior);
}
double Engine::stability(const Assemblage &a, std::vector<Minimum> *output) {
  set_pt(a.get_pressure(), a.get_temperature());
  auto mu = potentials(a);
  double worst = 0.;
  for (auto &ph : phases)
    if (!ph.available && std::find(settings.required_eos_phases.begin(),
                                   settings.required_eos_phases.end(),
                                   ph.material->get_name()) !=
                             settings.required_eos_phases.end())
      throw eos::SLBDomainError("Required model outside EOS domain: " +
                                ph.domain_error);
  if (output)
    output->clear();
  for (std::size_t i = 0; i < phases.size(); ++i) {
    auto ms = minima(burnman::utils::checked_int(i), mu);
    worst = std::min(worst, ms[0].affinity);
    if (output)
      output->push_back(ms[0]);
  }
  return worst;
}
double Engine::mass_error(const Assemblage &a) const {
  Eigen::VectorXd amount =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(elements.size()));
  auto formula = a.get_formula();
  for (std::size_t k = 0; k < elements.size(); ++k)
    if (formula.count(elements[k]))
      amount[static_cast<Eigen::Index>(k)] =
          formula.at(elements[k]) * a.get_n_moles();
  return (amount - full_bulk).norm() / full_bulk.norm();
}
std::vector<PhaseState> Engine::snapshot(const Assemblage &a,
                                         const std::vector<int> &ids) const {
  std::vector<PhaseState> out;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    PhaseState state;
    state.id = ids[k];
    state.candidate_index = ids[k] / settings.max_phase_instances;
    auto m = a.get_phase(k);
    state.name = m->get_name();
    state.amount =
        a.get_molar_fractions()[static_cast<Eigen::Index>(k)] * a.get_n_moles();
    if (auto s = std::dynamic_pointer_cast<Solution>(m))
      state.composition = s->get_molar_fractions().matrix();
    else
      state.composition = Eigen::VectorXd::Ones(1);
    state.composition =
        composition_basis(*m, state.candidate_index).transpose() *
        state.composition;
    out.push_back(state);
  }
  return out;
}
Eigen::MatrixXd Engine::composition_basis(const Material &m, int index) const {
  if (auto face = dynamic_cast<const FaceSolution *>(&m))
    return face->original_basis;
  if (auto face = dynamic_cast<const FaceMineral *>(&m))
    return face->original_basis;
  return Eigen::MatrixXd::Identity(
      phases[static_cast<std::size_t>(index)].vertices.cols(),
      phases[static_cast<std::size_t>(index)].vertices.cols());
}
WorkState Engine::fixed_pt(const std::vector<int> &ids,
                           const std::vector<PhaseState> &states, double p,
                           double t) {
  WorkState out;
  out.ids = ids;
  out.state.pressure = p;
  out.state.temperature = t;
  try {
    out.assemblage = make_assemblage(ids, states, p, t);
    optim::roots::DampedNewtonResult sol;
    for (std::size_t pass = 0; pass < ids.size(); ++pass) {
      auto c = constraints(std::make_unique<PressureConstraint>(p),
                           std::make_unique<TemperatureConstraint>(t));
      sol = solve(*out.assemblage, c);
      out.state.phases = snapshot(*out.assemblage, out.ids);
      bool merged = false;
      for (std::size_t i = 0; i < out.state.phases.size(); ++i)
        for (std::size_t j = i + 1; j < out.state.phases.size();) {
          auto &first = out.state.phases[i];
          const auto &second = out.state.phases[j];
          double amount = first.amount + second.amount;
          if (first.candidate_index == second.candidate_index && amount > 0. &&
              (first.composition - second.composition).norm() <
                  settings.composition_tolerance) {
            first.composition = (first.amount * first.composition +
                                 second.amount * second.composition) /
                                amount;
            first.amount = amount;
            out.state.phases.erase(out.state.phases.begin() +
                                   static_cast<std::ptrdiff_t>(j));
            out.ids.erase(out.ids.begin() + static_cast<std::ptrdiff_t>(j));
            merged = true;
          } else
            ++j;
        }
      if (merged) {
        // Coincident solution copies describe one field phase. Rebuild its
        // equations while preserving the bulk, including at zero kelvin.
        out.assemblage = make_assemblage(out.ids, out.state.phases, p, t);
        continue;
      }
      out.state.mass_balance_error = mass_error(*out.assemblage);
      out.state.minimum_affinity = stability(*out.assemblage);
      out.state.excluded_phases.clear();
      for (auto &ph : phases)
        if (!ph.available)
          out.state.excluded_phases.push_back(ph.domain_error);
      out.state.gibbs =
          out.assemblage->get_molar_gibbs() * out.assemblage->get_n_moles();
      out.state.equilibrium_error =
          out.assemblage->get_n_reactions()
              ? out.assemblage->get_reaction_affinities().cwiseAbs().maxCoeff()
              : 0.;
      out.state.success =
          sol.success &&
          out.state.equilibrium_error <= settings.affinity_tolerance * .1 &&
          out.state.mass_balance_error <= settings.mass_balance_tolerance &&
          out.state.minimum_affinity >= -settings.affinity_tolerance;
      if (out.state.success) {
        std::vector<int> copies(phases.size(), 0);
        for (std::size_t i = 0; i < out.state.phases.size(); ++i) {
          auto &phase = out.state.phases[i];
          int copy = copies[static_cast<std::size_t>(phase.candidate_index)]++;
          phase.id =
              phase.candidate_index * settings.max_phase_instances + copy;
          phase.name = phases[static_cast<std::size_t>(phase.candidate_index)]
                           .material->get_name() +
                       (copy ? " #" + std::to_string(copy + 1) : "");
          out.ids[i] = phase.id;
          out.assemblage->get_phase(i)->set_name(phase.name);
        }
        out.state.message = "Stable equilibrium";
        return out;
      }
      // A phase blocked at zero amount cannot be required to coexist. Drop its
      // chemical equations and retry from the last compositions of the
      // surviving phases. This applies to fixed-P,T fields, never zero-amount
      // phase lines.
      std::vector<int> surviving;
      double total = 0.;
      for (auto &ph : out.state.phases)
        total += ph.amount;
      for (auto &ph : out.state.phases)
        if (ph.amount > settings.amount_tolerance * total)
          surviving.push_back(ph.id);
      if (surviving.empty() || surviving.size() == out.ids.size())
        break;
      out.ids = std::move(surviving);
      out.assemblage = make_assemblage(out.ids, out.state.phases, p, t);
    }
    std::ostringstream message;
    message << "Equilibrium validation failed: reaction residual="
            << out.state.equilibrium_error
            << " J/mol, minimum candidate affinity="
            << out.state.minimum_affinity
            << " J/mol, relative mass error=" << out.state.mass_balance_error
            << ". " << sol.message;
    out.state.message = message.str();
  } catch (const std::exception &error) {
    out.state.message = error.what();
    out.state.success = false;
  }
  for (auto &ph : phases)
    if (!ph.available) {
      if (std::find(out.state.excluded_phases.begin(),
                    out.state.excluded_phases.end(),
                    ph.domain_error) == out.state.excluded_phases.end())
        out.state.excluded_phases.push_back(ph.domain_error);
      if (std::find(settings.required_eos_phases.begin(),
                    settings.required_eos_phases.end(),
                    ph.material->get_name()) !=
          settings.required_eos_phases.end())
        out.state.outside_model_domain = true;
    }
  return out;
}
WorkState Engine::stable(double p, double t) {
  WorkState out;
  out.state.pressure = p;
  out.state.temperature = t;
  try {
    if (t == 0. && settings.active_solution_faces) {
      // Resolve flat zero-temperature solution faces with a warm seed, then
      // validate the equilibrium at exactly zero kelvin.
      auto warm = stable(p, .05);
      if (warm.state.success) {
        auto cold = fixed_pt(warm.ids, warm.state.phases, p, t);
        if (cold.state.success)
          return cold;
      }
    }
    set_pt(p, t);
    for (auto &ph : phases)
      if (!ph.available && std::find(settings.required_eos_phases.begin(),
                                     settings.required_eos_phases.end(),
                                     ph.material->get_name()) !=
                               settings.required_eos_phases.end())
        throw polytope::InfeasibleBulk("Required model outside EOS domain: " +
                                       ph.domain_error);
    struct Compound {
      int phase;
      Eigen::VectorXd p;
    };
    std::vector<Compound> compounds;
    for (std::size_t i = 0; i < phases.size(); ++i) {
      if (!phases[i].available)
        continue;
      for (int j = 0; j < phases[i].vertices.rows(); ++j)
        compounds.push_back(
            {static_cast<int>(i), phases[i].vertices.row(j).transpose()});
      if (phases[i].solution)
        compounds.push_back({static_cast<int>(i),
                             phases[i].vertices.colwise().mean().transpose()});
    }
    polytope::GibbsLPResult lp;
    auto equilibrate_lp = [&]() {
      std::vector<PhaseState> seeds;
      std::vector<int> ids;
      for (std::size_t i = 0; i < phases.size(); ++i) {
        std::vector<PhaseState> clusters;
        for (int j = 0; j < lp.amounts.size(); ++j)
          if (compounds[static_cast<std::size_t>(j)].phase ==
                  static_cast<int>(i) &&
              lp.amounts[j] > settings.amount_tolerance * lp.amounts.sum()) {
            auto &c = compounds[static_cast<std::size_t>(j)];
            auto it =
                std::find_if(clusters.begin(), clusters.end(), [&](auto &old) {
                  return (old.composition - c.p).norm() < .05;
                });
            if (it == clusters.end()) {
              PhaseState s;
              s.candidate_index = burnman::utils::checked_int(i);
              s.amount = lp.amounts[j];
              s.composition = c.p;
              clusters.push_back(s);
            } else {
              it->composition =
                  (it->amount * it->composition + lp.amounts[j] * c.p) /
                  (it->amount + lp.amounts[j]);
              it->amount += lp.amounts[j];
            }
          }
        if (clusters.size() >
            static_cast<std::size_t>(settings.max_phase_instances))
          throw std::runtime_error(
              "More coexisting solution instances required; "
              "increase max_phase_instances.");
        std::sort(
            clusters.begin(), clusters.end(), [](auto &first, auto &second) {
              for (Eigen::Index j = 0; j < first.composition.size(); ++j) {
                auto left = std::llround(first.composition[j] * 1.e7),
                     right = std::llround(second.composition[j] * 1.e7);
                if (left != right)
                  return left < right;
              }
              return false;
            });
        for (std::size_t j = 0; j < clusters.size(); ++j) {
          clusters[j].id = burnman::utils::checked_int(
              i * static_cast<std::size_t>(settings.max_phase_instances) + j);
          seeds.push_back(clusters[j]);
          ids.push_back(clusters[j].id);
        }
      }
      potential_seed = lp.chemical_potentials;
      return fixed_pt(ids, seeds, p, t);
    };
    auto add_compound = [&](int phase, const Eigen::VectorXd &composition) {
      for (const auto &old : compounds)
        if (old.phase == phase && (old.p - composition).norm() < 1.e-9)
          return false;
      compounds.push_back({phase, composition});
      return true;
    };
    for (int iteration = 0; iteration < settings.max_refinement_iterations;
         ++iteration) {
      Eigen::MatrixXd a(compounds.size(), components.size());
      Eigen::VectorXd g(compounds.size());
      for (std::size_t i = 0; i < compounds.size(); ++i) {
        auto &c = compounds[i];
        g[static_cast<Eigen::Index>(i)] = energy(c.phase, c.p);
        auto full =
            (phases[static_cast<std::size_t>(c.phase)].a.transpose() * c.p)
                .eval();
        for (std::size_t k = 0; k < components.size(); ++k)
          a(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) =
              full[components[k]];
      }
      lp = polytope::gibbs_linear_program(a, g, b);
      double worst = 0.;
      int added = 0;
      for (std::size_t i = 0; i < phases.size(); ++i) {
        Eigen::VectorXd seed;
        double maxamount = 0;
        for (std::size_t j = 0; j < static_cast<std::size_t>(lp.amounts.size());
             ++j)
          if (compounds[j].phase == static_cast<int>(i) &&
              lp.amounts[static_cast<Eigen::Index>(j)] > maxamount) {
            maxamount = lp.amounts[static_cast<Eigen::Index>(j)];
            seed = compounds[j].p;
          }
        for (auto &m : minima(burnman::utils::checked_int(i),
                              lp.chemical_potentials, seed))
          if (m.affinity < -settings.affinity_tolerance * .1) {
            worst = std::min(worst, m.affinity);
            bool exists = false;
            for (auto &old : compounds)
              if (old.phase == static_cast<int>(i) &&
                  (old.p - m.p).norm() < 1.e-9) {
                exists = true;
                break;
              }
            if (!exists) {
              compounds.push_back({static_cast<int>(i), m.p});
              ++added;
            }
          }
      }
      if (settings.verbose &&
          (iteration % 10 == 0 || worst >= -settings.affinity_tolerance * .1))
        std::cerr << "LP " << iteration << " P=" << p << " T=" << t
                  << " affinity=" << worst << " compounds=" << compounds.size()
                  << '\n';
      if (worst >= -settings.affinity_tolerance * .1) {
        auto equilibrium = equilibrate_lp();
        if (equilibrium.state.success || !equilibrium.assemblage ||
            equilibrium.state.minimum_affinity >= -settings.affinity_tolerance)
          return equilibrium;
        // The refined compositions must enter the LP too: frozen mesh points
        // can otherwise hide a feasible direction towards a lower-energy phase.
        for (const auto &state : equilibrium.state.phases) {
          added += add_compound(state.candidate_index, state.composition);
          const auto &vertices =
              phases[static_cast<std::size_t>(state.candidate_index)].vertices;
          // Neighbouring feasible compositions constrain chemical-potential
          // slopes that a discrete LP otherwise leaves underdetermined.
          for (int vertex = 0; vertex < vertices.rows(); ++vertex)
            added += add_compound(
                state.candidate_index,
                state.composition + 1.e-7 * (vertices.row(vertex).transpose() -
                                             state.composition));
        }
        std::vector<Minimum> missing;
        stability(*equilibrium.assemblage, &missing);
        for (std::size_t i = 0; i < missing.size(); ++i)
          if (missing[i].affinity < -settings.affinity_tolerance)
            added += add_compound(static_cast<int>(i), missing[i].p);
        if (!added)
          return equilibrium;
      }
      if (!added)
        break;
    }
    throw std::runtime_error(
        "Tangent-plane refinement did not converge; increase "
        "max_refinement_iterations or minimization_starts.");
  } catch (const polytope::InfeasibleBulk &e) {
    out.state.outside_model_domain = std::any_of(
        phases.begin(), phases.end(), [](auto &ph) { return !ph.available; });
    out.state.message = e.what();
    out.state.success = false;
  } catch (const std::exception &e) {
    out.state.message = e.what();
    out.state.success = false;
  }
  for (auto &ph : phases)
    if (!ph.available)
      out.state.excluded_phases.push_back(ph.domain_error);
  return out;
}
} // namespace burnman::pseudosections::detail
namespace burnman::pseudosections {
State stable_equilibrium(
    const types::FormulaMap &bulk,
    const std::vector<std::shared_ptr<Material>> &candidates, double p,
    double t, const Settings &settings) {
  if (!std::isfinite(p) || p < 0 || !std::isfinite(t) || t < 0 ||
      (t == 0 && !settings.active_solution_faces))
    throw std::invalid_argument("Equilibrium requires finite P>=0 and T>0 (T=0 "
                                "requires active_solution_faces).");
  detail::Engine engine(bulk, candidates, settings);
  return engine.stable(p, t).state;
}
} // namespace burnman::pseudosections
