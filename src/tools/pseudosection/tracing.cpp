/* GPL v3 or later. Phase-boundary predictor/corrector and topology search. */
#include "critical.hpp"
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
namespace burnman::pseudosections {
namespace {
using namespace detail;
struct Seed {
  std::vector<int> ids;
  int zero = -1, node = -1;
  std::vector<PhaseState> phases;
  Eigen::Vector2d pt, direction;
  std::vector<BoundaryPoint> prefix;
  int bridge_node = -1;
};
class Tracer {
  Engine engine;
  Result result;
  Eigen::Vector2d origin, range;
  std::deque<Seed> queue;
  std::map<std::vector<int>, int> field_index;
  std::vector<WorkState> grid;
  std::set<std::string> attempted;
  std::set<std::string> expanded;
  Eigen::VectorXd critical_axis;
  Eigen::Vector2d normalise(double p, double t) const {
    return (Eigen::Vector2d(p, t) - origin).cwiseQuotient(range);
  }
  Eigen::Vector2d pt(const Assemblage &a) const {
    return normalise(a.get_pressure(), a.get_temperature());
  }
  bool inside(const Eigen::Vector2d &u, double margin = 1.e-8) const {
    return (u.array() >= -margin).all() && (u.array() <= 1 + margin).all();
  }
  std::unique_ptr<EqualityConstraint>
  section(int n, const Eigen::Vector2d &u,
          const Eigen::Vector2d &normal) const {
    Eigen::VectorXd A = Eigen::VectorXd::Zero(n);
    A.head<2>() = normal.cwiseQuotient(range);
    return std::make_unique<LinearXConstraint>(
        A, normal.dot(u + origin.cwiseQuotient(range)));
  }
  ConstraintList boundary_constraints(const Assemblage &a, int zero,
                                      const Eigen::Vector2d &u,
                                      const Eigen::Vector2d &normal) {
    auto prm = get_equilibration_parameters(a, engine.bulk, {});
    auto it = std::find(current_ids.begin(), current_ids.end(), zero);
    if (it == current_ids.end())
      throw std::runtime_error("Missing zero phase in boundary.");
    return constraints(std::make_unique<PhaseFractionConstraint>(
                           it - current_ids.begin(), 0., prm),
                       section(prm.n_parameters, u, normal));
  }
  std::vector<int> current_ids;
  // Project the complete Jacobian null space into scaled P,T. Amount-only
  // null directions at reduced variance must not become spurious lines.
  Eigen::VectorXd tangent(const optim::roots::DampedNewtonResult &s,
                          const Assemblage &a, int *pt_rank = nullptr) {
    Eigen::VectorXd scales = Eigen::VectorXd::Ones(s.x.size());
    scales.head<2>() = range;
    auto prm = get_equilibration_parameters(a, engine.bulk, {});
    for (int k = 0; k < prm.phase_amount_indices.size(); ++k)
      scales[prm.phase_amount_indices[k]] = std::max(1.e-3, a.get_n_moles());
    Eigen::MatrixXd j(s.J.rows() - 1, s.J.cols());
    j.topRows(1) = s.J.topRows(1);
    j.bottomRows(s.J.rows() - 2) = s.J.bottomRows(s.J.rows() - 2);
    j = j * scales.asDiagonal();
    for (int k = 0; k < j.rows(); ++k) {
      double norm = j.row(k).norm();
      if (norm > 0)
        j.row(k) /= norm;
    }
    // Trace populations produce strongly separated singular values even after
    // row scaling. Retain their coupling to P,T when using active faces.
    double rank_tolerance =
        engine.settings.active_solution_faces ? 1.e-12 : 1.e-9;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(j, Eigen::ComputeFullV);
    int rank = (svd.singularValues().array() > rank_tolerance).count();
    Eigen::MatrixXd null = svd.matrixV().rightCols(j.cols() - rank);
    Eigen::JacobiSVD<Eigen::MatrixXd> projection(
        null.topRows(2), Eigen::ComputeThinU | Eigen::ComputeThinV);
    int dim = (projection.singularValues().array() > 1.e-7).count();
    if (pt_rank)
      *pt_rank = dim;
    if (dim != 1)
      return Eigen::VectorXd();
    auto d = (null * projection.matrixV().col(0)).eval();
    d /= d.head(2).norm();
    return scales.asDiagonal() * d;
  }
  BoundaryPoint point(const Assemblage &a, const std::vector<int> &ids,
                      const optim::roots::DampedNewtonResult &solve,
                      double affinity) {
    BoundaryPoint p;
    p.pressure = a.get_pressure();
    p.temperature = a.get_temperature();
    p.phases = engine.snapshot(a, ids);
    p.mass_balance_error = engine.mass_error(a);
    p.minimum_affinity = affinity;
    p.residual =
        solve.F.size() > 2 ? solve.F.tail(solve.F.size() - 2).norm() : 0.;
    return p;
  }
  bool valid(const Assemblage &a,
             const optim::roots::DampedNewtonResult &s) const {
    return s.success && s.x.allFinite() && inside(pt(a)) &&
           engine.mass_error(a) <= engine.settings.mass_balance_tolerance;
  }
  double separation(const std::vector<PhaseState> &states) const {
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < states.size(); ++i)
      for (std::size_t j = i + 1; j < states.size(); ++j)
        if (states[i].candidate_index == states[j].candidate_index)
          minimum = std::min(
              minimum, (states[i].composition - states[j].composition).norm());
    return minimum;
  }
  bool distinct(const Assemblage &a, const std::vector<int> &ids) const {
    return separation(engine.snapshot(a, ids)) >
           engine.settings.composition_tolerance * 10.;
  }
  bool same_branch(const Assemblage &a, const Assemblage &b,
                   const std::vector<int> &ids) const {
    auto old = engine.snapshot(a, ids), next = engine.snapshot(b, ids);
    for (std::size_t i = 0; i < ids.size(); ++i)
      for (std::size_t j = i + 1; j < ids.size(); ++j)
        if (old[i].candidate_index == old[j].candidate_index &&
            (old[i].composition - old[j].composition)
                    .dot(next[i].composition - next[j].composition) <= 0.)
          return false;
    return true;
  }
  // Predict compositions as well as P,T, shortening only the compositional
  // displacement when it would cross a linear site-occupancy face. Signed
  // ordering endmembers need site bounds, rather than endmember clipping.
  void predict(Assemblage &a, const Eigen::VectorXd &x,
               const Eigen::VectorXd &delta,
               const EquilibrationParameters &prm) {
    Eigen::VectorXd guess = x + delta;
    for (int j : prm.phase_amount_indices)
      guess[j] = std::max(0., guess[j]);
    auto change = (guess - x).eval();
    auto slack = (prm.constraint_matrix * x + prm.constraint_vector).eval();
    auto slope = (prm.constraint_matrix * change).eval();
    double fraction = 1.;
    for (int i = 2; i < slope.size(); ++i)
      if (slope[i] > 1.e-14)
        fraction = std::min(fraction, std::max(0., -slack[i]) / slope[i]);
    guess = x + (fraction < 1. ? fraction * .99 : 1.) * change;
    guess.head<2>() = (x + delta).head<2>();
    for (int j : prm.phase_amount_indices)
      guess[j] = std::max(0., guess[j]);
    set_composition_and_state_from_parameters(a, guess);
  }
  std::vector<int> active(const State &s) const {
    double total = 0.;
    for (auto &ph : s.phases)
      total += ph.amount;
    std::map<int, int> count;
    for (auto &ph : s.phases)
      if (ph.amount > engine.settings.amount_tolerance * total)
        ++count[ph.candidate_index];
    std::vector<int> ids;
    for (auto [base, n] : count)
      for (int j = 0; j < n; ++j)
        ids.push_back(base * engine.settings.max_phase_instances + j);
    return ids;
  }
  static bool composition_order(const PhaseState &a, const PhaseState &b) {
    if (a.candidate_index != b.candidate_index)
      return a.candidate_index < b.candidate_index;
    for (int k = 0; k < a.composition.size(); ++k) {
      auto left = std::llround(a.composition[k] * 1.e7),
           right = std::llround(b.composition[k] * 1.e7);
      if (left != right)
        return left < right;
    }
    return a.id < b.id;
  }
  Seed canonical_seed(const Seed &original) const {
    Seed out = original;
    std::vector<PhaseState> states;
    for (auto &state : out.phases)
      if (std::find(out.ids.begin(), out.ids.end(), state.id) != out.ids.end())
        states.push_back(state);
    std::sort(states.begin(), states.end(), composition_order);
    std::map<int, int> count, renumber;
    out.ids.clear();
    for (auto &state : states) {
      int id = state.candidate_index * engine.settings.max_phase_instances +
               count[state.candidate_index]++;
      if (state.id == original.zero)
        out.zero = id;
      renumber[state.id] = id;
      state.id = id;
      state.name = result.phase_names[id];
      out.ids.push_back(id);
    }
    for (auto &point : out.prefix) {
      for (auto &phase : point.phases) {
        phase.id = renumber.at(phase.id);
        phase.name = result.phase_names[phase.id];
      }
      std::sort(point.phases.begin(), point.phases.end(),
                [](auto &a, auto &b) { return a.id < b.id; });
    }
    out.phases = std::move(states);
    return out;
  }
  void add_sample(const State &state, int existing = -1) {
    int index = existing >= 0 ? existing : result.samples.size();
    State canonical = state;
    if (state.success) {
      double total = 0.;
      for (auto &ph : state.phases)
        total += ph.amount;
      canonical.phases.erase(
          std::remove_if(canonical.phases.begin(), canonical.phases.end(),
                         [&](auto &ph) {
                           return ph.amount <=
                                  engine.settings.amount_tolerance * total;
                         }),
          canonical.phases.end());
      // A saved, nearly identical solution pair is one phase, not two minima.
      // Weighted averaging preserves elemental mass exactly for linear
      // formulae.
      for (std::size_t i = 0; i < canonical.phases.size(); ++i)
        for (std::size_t j = i + 1; j < canonical.phases.size();) {
          auto &a = canonical.phases[i];
          auto &b = canonical.phases[j];
          if (a.candidate_index == b.candidate_index &&
              (a.composition - b.composition).norm() <=
                  engine.settings.composition_tolerance * 10.) {
            a.composition =
                (a.amount * a.composition + b.amount * b.composition) /
                (a.amount + b.amount);
            a.amount += b.amount;
            canonical.phases.erase(canonical.phases.begin() + j);
          } else
            ++j;
        }
      std::sort(canonical.phases.begin(), canonical.phases.end(),
                composition_order);
      std::map<int, int> count;
      for (auto &ph : canonical.phases) {
        ph.id = ph.candidate_index * engine.settings.max_phase_instances +
                count[ph.candidate_index]++;
        ph.name = result.phase_names[ph.id];
      }
    }
    if (existing >= 0)
      result.samples[existing] = canonical;
    else
      result.samples.push_back(canonical);
    if (!state.success) {
      std::ostringstream msg;
      msg << "Unresolved seed P=" << state.pressure
          << " Pa, T=" << state.temperature << " K: " << state.message;
      result.diagnostics.push_back(msg.str());
      return;
    }
    auto ids = active(canonical);
    auto it = field_index.find(ids);
    int id;
    if (it == field_index.end()) {
      id = result.fields.size();
      field_index[ids] = id;
      result.fields.push_back({id, ids, {}});
    } else
      id = it->second;
    result.fields[id].sample_indices.push_back(index);
  }
  bool covered(const Seed &seed) const {
    for (auto &line : result.boundaries)
      if (line.assemblage == seed.ids && line.zero_phase == seed.zero)
        for (std::size_t k = 1; k < line.points.size(); ++k) {
          auto a = normalise(line.points[k - 1].pressure,
                             line.points[k - 1].temperature),
               b = normalise(line.points[k].pressure,
                             line.points[k].temperature);
          auto d = (b - a).eval();
          double f =
              d.squaredNorm() > 0
                  ? std::clamp((seed.pt - a).dot(d) / d.squaredNorm(), 0., 1.)
                  : 0.;
          if ((a + f * d - seed.pt).norm() < engine.settings.node_tolerance * 2)
            return true;
        }
    return false;
  }
  static bool unfinished(const std::string &status) {
    return status.find("failed") != std::string::npos ||
           status.find("unresolved") != std::string::npos ||
           status == "trace step limit";
  }
  static std::pair<std::string, std::string> statuses(const Boundary &line) {
    auto split = line.termination.find(';');
    if (split == std::string::npos)
      throw std::invalid_argument(
          "Boundary termination must describe both directions.");
    auto second = line.termination.substr(split + 1);
    second.erase(0, second.find_first_not_of(' '));
    return {line.termination.substr(0, split), second};
  }
  void trim_junction_overshoot(Boundary &line) const {
    auto [back, forward] = statuses(line);
    for (bool start : {false, true}) {
      int n = start ? line.start_node : line.end_node;
      if (n < 0 || (start ? back : forward) != "junction")
        continue;
      // The stability tolerance can admit a point just beyond zero affinity.
      // Keep the exact junction and remove only the local reversing tail,
      // rather than displaying a doubled-back spur as an unfinished phase
      // boundary.
      auto junction =
          normalise(result.nodes[n].pressure, result.nodes[n].temperature);
      while (line.points.size() > 2) {
        std::size_t tip = start ? 1 : line.points.size() - 2,
                    previous = start ? 2 : line.points.size() - 3;
        auto u =
            normalise(line.points[tip].pressure, line.points[tip].temperature);
        auto before = normalise(line.points[previous].pressure,
                                line.points[previous].temperature);
        auto incoming = (u - before).eval(), remaining = (junction - u).eval();
        if (remaining.norm() > engine.settings.node_tolerance * 2. ||
            remaining.dot(incoming) >= -1.e-18)
          break;
        line.points.erase(line.points.begin() + tip);
      }
    }
  }
  std::vector<int> event_signature(const std::vector<int> &ids,
                                   bool polymorphs = false) const {
    std::vector<int> out;
    for (int id : ids) {
      int base = id / engine.settings.max_phase_instances;
      if (polymorphs && !engine.phases[base].solution)
        for (int j = 0; j < base; ++j)
          if (!engine.phases[j].solution &&
              engine.phases[j].material->get_formula() ==
                  engine.phases[base].material->get_formula()) {
            base = j;
            break;
          }
      out.push_back(base);
    }
    std::sort(out.begin(), out.end());
    return out;
  }
  bool same_event(const Node &previous, const std::vector<int> &ids,
                  const std::vector<int> &zero) const {
    return event_signature(previous.assemblage) == event_signature(ids) &&
           event_signature(previous.zero_phases, true) ==
               event_signature(zero, true);
  }
  int node(Assemblage &a, const std::vector<int> &ids,
           const std::vector<int> &zero, const std::string &kind,
           const Eigen::MatrixXd *jacobian = nullptr) {
    auto u = pt(a);
    for (auto &old : result.nodes)
      if ((normalise(old.pressure, old.temperature) - u).norm() <
          engine.settings.node_tolerance) {
        double distance = (normalise(old.pressure, old.temperature) - u).norm();
        bool compatible_kind =
            old.kind == kind ||
            (kind == "critical_point" && old.kind == "reduced_variance");
        if (distance > 1.e-8 &&
            (!compatible_kind || !same_event(old, ids, zero)))
          continue;
        if (kind == "critical_point" &&
            separation(engine.snapshot(a, ids)) < 1.e-8 &&
            critical_axis.size()) {
          old.pressure = a.get_pressure();
          old.temperature = a.get_temperature();
          old.critical_mode = critical_axis;
          old.kind = "critical_point";
        }
        if (kind == "junction")
          expand(a, ids, zero, old.id);
        if (kind == "critical_point")
          expand_critical(a, ids, zero, old.id);
        return old.id;
      }
    Node n;
    n.id = result.nodes.size();
    n.pressure = a.get_pressure();
    n.temperature = a.get_temperature();
    n.kind = kind;
    n.assemblage = ids;
    n.zero_phases = zero;
    if (kind == "critical_point" && separation(engine.snapshot(a, ids)) < 1.e-8)
      n.critical_mode = critical_axis;
    n.gibbs_variance =
        a.get_independent_element_indices().size() - ids.size() + 2;
    n.pt_nullity = 0;
    if (jacobian) {
      Eigen::VectorXd scales = Eigen::VectorXd::Ones(jacobian->cols());
      scales.head<2>() = range;
      Eigen::MatrixXd j = (*jacobian) * scales.asDiagonal();
      for (int i = 0; i < j.rows(); ++i)
        if (j.row(i).norm() > 0)
          j.row(i) /= j.row(i).norm();
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(j, Eigen::ComputeFullV);
      int rank = (svd.singularValues().array() > 1.e-9).count();
      if (rank < j.cols()) {
        Eigen::MatrixXd projection =
            svd.matrixV().rightCols(j.cols() - rank).topRows(2);
        Eigen::JacobiSVD<Eigen::MatrixXd> ptsvd(projection);
        n.pt_nullity = (ptsvd.singularValues().array() > 1.e-7).count();
      }
    }
    result.nodes.push_back(n);
    if (kind == "junction")
      expand(a, ids, zero, n.id);
    if (kind == "critical_point")
      expand_critical(a, ids, zero, n.id);
    return n.id;
  }
  void separate_junctions() {
    // Older traces merged nearby, thermodynamically different events. Their
    // accepted endpoint states still locate each event; restore that topology
    // without moving any thermodynamic point or drawing a guessed connection.
    for (auto &line : result.boundaries) {
      if (line.points.empty())
        continue;
      auto [back, forward] = statuses(line);
      for (bool start : {true, false}) {
        if ((start ? back : forward) != "junction")
          continue;
        auto &id = start ? line.start_node : line.end_node;
        if (id < 0 || result.nodes.at(id).kind != "junction")
          continue;
        auto &p = start ? line.points.front() : line.points.back();
        auto u = normalise(p.pressure, p.temperature);
        std::vector<int> ids, zeros;
        double total = 0.;
        for (auto &ph : p.phases)
          total += std::abs(ph.amount);
        for (auto &ph : p.phases) {
          ids.push_back(ph.id);
          if (std::abs(ph.amount) <= 1.e-10 * std::max(1., total))
            zeros.push_back(ph.id);
        }
        if (zeros.size() < 2)
          continue;
        int found = -1;
        for (auto &n : result.nodes)
          if (n.kind == "junction") {
            double distance = (normalise(n.pressure, n.temperature) - u).norm();
            if (distance < 1.e-8 ||
                (distance < engine.settings.node_tolerance &&
                 same_event(n, ids, zeros))) {
              found = n.id;
              break;
            }
          }
        if (found < 0) {
          Node n;
          n.id = result.nodes.size();
          n.pressure = p.pressure;
          n.temperature = p.temperature;
          n.kind = "junction";
          n.assemblage = ids;
          n.zero_phases = zeros;
          auto a =
              engine.make_assemblage(ids, p.phases, p.pressure, p.temperature);
          n.gibbs_variance =
              a->get_independent_element_indices().size() - ids.size() + 2;
          result.nodes.push_back(n);
          found = n.id;
        }
        id = found;
      }
    }
  }
  // Fix the projection of the composition difference instead of P or T. This
  // excludes the trivial identical-copy root at a solvus and remains useful
  // when the P,T projection of the equilibrium null space loses rank.
  ConstraintList separation_constraints(const Assemblage &a,
                                        const std::vector<int> &ids, int zero,
                                        int first, int second,
                                        const Eigen::VectorXd &axis,
                                        double target) {
    auto prm = get_equilibration_parameters(a, engine.bulk, {});
    Eigen::VectorXd coefficient = Eigen::VectorXd::Zero(prm.n_parameters);
    for (auto [phase, sign] :
         {std::pair<int, double>{first, 1.}, {second, -1.}}) {
      Eigen::VectorXd projected =
          engine.composition_basis(*a.get_phase(phase),
                                   ids[phase] /
                                       engine.settings.max_phase_instances) *
          axis;
      coefficient.segment(prm.phase_amount_indices[phase] + 1,
                          projected.size() - 1) =
          sign * (projected.tail(projected.size() - 1).array() - projected[0])
                     .matrix();
    }
    int z = std::find(ids.begin(), ids.end(), zero) - ids.begin();
    return constraints(
        std::make_unique<PhaseFractionConstraint>(z, 0., prm),
        std::make_unique<LinearXConstraint>(coefficient, target));
  }
  bool approach_critical(std::shared_ptr<Assemblage> &a,
                         const std::vector<int> &ids, int zero,
                         optim::roots::DampedNewtonResult &solve,
                         std::vector<BoundaryPoint> &points) {
    critical_axis.resize(0);
    auto initial = engine.snapshot(*a, ids);
    int z = std::find(ids.begin(), ids.end(), zero) - ids.begin();
    for (std::size_t k = 0; k < ids.size(); ++k)
      if (k != static_cast<std::size_t>(z) &&
          initial[k].candidate_index == initial[z].candidate_index) {
        Eigen::VectorXd axis = initial[z].composition - initial[k].composition;
        double delta = axis.norm();
        if (delta <= 1.e-8) {
          auto &previous = points.empty() ? initial : points.back().phases;
          for (auto &ph : previous)
            if (ph.id == zero)
              for (auto &other : previous)
                if (other.id == ids[k])
                  axis = ph.composition - other.composition;
          if (axis.norm() <= 1.e-8)
            for (auto &n : result.nodes)
              if (n.critical_mode.size() == axis.size() &&
                  (normalise(n.pressure, n.temperature) - pt(*a)).norm() <
                      engine.settings.node_tolerance)
                axis = n.critical_mode * .001;
          delta = axis.norm();
        }
        if (delta <= 1.e-8 || delta >= .04)
          continue;
        axis /= delta;
        // Solve the intrinsic critical conditions on a single surviving copy.
        // The third derivative is analytic, avoiding cancellation of chemical
        // potentials that differ by the cube of the composition separation.
        try {
          auto subset = ids;
          subset.erase(subset.begin() + z);
          auto states = initial;
          states[k].composition =
              (initial[z].composition + initial[k].composition) * .5;
          states[k].amount += initial[z].amount;
          auto critical = engine.make_assemblage(
              subset, states, a->get_pressure(), a->get_temperature());
          int phase =
              std::find(subset.begin(), subset.end(), ids[k]) - subset.begin();
          if (!critical->get_phase<Solution>(phase))
            continue;
          auto prm = get_equilibration_parameters(*critical, engine.bulk, {});
          auto mode_basis = engine.composition_basis(
              *critical->get_phase(phase),
              ids[k] / engine.settings.max_phase_instances);
          Eigen::VectorXd projected_axis =
              mode_basis.transpose().completeOrthogonalDecomposition().solve(
                  axis);
          auto curvature = std::make_unique<CriticalConstraint>(
              *critical, phase, prm, projected_axis, false);
          auto third = std::make_unique<CriticalConstraint>(
              *critical, phase, prm, projected_axis, true);
          auto c = constraints(std::move(curvature), std::move(third));
          auto s = engine.solve(*critical, c);
          if (valid(*critical, s)) {
            double affinity = engine.stability(*critical);
            if (affinity >= -engine.settings.affinity_tolerance) {
              critical_axis = mode_basis.transpose() *
                              static_cast<CriticalConstraint *>(c[0][0].get())
                                  ->mode(*critical, phase);
              auto full = engine.snapshot(*critical, subset);
              PhaseState absent = initial[z];
              absent.amount = 0.;
              absent.composition = full[phase].composition;
              full.push_back(absent);
              a = engine.make_assemblage(ids, full, critical->get_pressure(),
                                         critical->get_temperature());
              solve = s;
              points.push_back(point(*a, ids, solve, affinity));
              return true;
            }
          }
        } catch (const std::exception &error) {
          if (engine.settings.verbose)
            std::cerr << "Intrinsic critical solve: " << error.what() << '\n';
        }
        auto reflected = initial;
        reflected[z].amount = initial[k].amount;
        reflected[z].composition = initial[k].composition;
        reflected[k].amount = 0.;
        reflected[k].composition =
            2. * initial[k].composition - initial[z].composition;
        const double p = a->get_pressure(), t = a->get_temperature();
        try {
          auto left_base = engine.make_assemblage(ids, initial, p, t),
               right_base = engine.make_assemblage(ids, reflected, p, t);
          auto left_constraints =
              separation_constraints(*left_base, ids, zero, z, k, axis, delta);
          auto right_constraints = separation_constraints(
              *right_base, ids, ids[k], z, k, axis, delta);
          auto left_solve = engine.solve(*left_base, left_constraints),
               right_solve = engine.solve(*right_base, right_constraints);
          if (!valid(*left_base, left_solve) ||
              !valid(*right_base, right_solve))
            continue;
          // Both arms must converge, remain stable and agree in P,T within the
          // explicitly chosen node tolerance. Never connect distant open
          // endpoints.
          for (double factor = 1.;
               factor >= 1. / 4096. && delta * factor > 1.e-8; factor *= .5)
            try {
              auto left = engine.copy_assemblage(*left_base);
              auto right = engine.copy_assemblage(*right_base);
              for (auto entry : {std::pair{left, left_solve},
                                 std::pair{right, right_solve}}) {
                Eigen::VectorXd rhs =
                    Eigen::VectorXd::Zero(entry.second.x.size());
                rhs[1] = delta * (factor - 1.);
                Eigen::VectorXd dx = entry.second.J.partialPivLu().solve(rhs);
                if (dx.allFinite())
                  predict(*entry.first, entry.second.x, dx,
                          get_equilibration_parameters(*entry.first,
                                                       engine.bulk, {}));
              }
              auto lc = separation_constraints(*left, ids, zero, z, k, axis,
                                               delta * factor);
              auto rc = separation_constraints(*right, ids, ids[k], z, k, axis,
                                               delta * factor);
              auto ls = engine.solve(*left, lc), rs = engine.solve(*right, rc);
              if (engine.settings.verbose)
                std::cerr << "Critical approach " << zero
                          << ", separation=" << delta * factor << ": arms "
                          << ls.success << '/' << rs.success
                          << ", distance=" << (pt(*left) - pt(*right)).norm()
                          << ", left constraints=" << ls.F.head<2>().transpose()
                          << ", left residual="
                          << ls.F.tail(ls.F.size() - 2).cwiseAbs().maxCoeff()
                          << ", left mass=" << engine.mass_error(*left) << '\n';
              if (!valid(*left, ls) || !valid(*right, rs) ||
                  separation(engine.snapshot(*left, ids)) <
                      delta * factor * .5 ||
                  separation(engine.snapshot(*right, ids)) <
                      delta * factor * .5 ||
                  (pt(*left) - pt(*right)).norm() >
                      engine.settings.node_tolerance)
                continue;
              double la = engine.stability(*left),
                     ra = engine.stability(*right);
              if (std::min(la, ra) < -engine.settings.affinity_tolerance)
                continue;
              a = left;
              solve = ls;
              points.push_back(point(*a, ids, solve, la));
              return true;
            } catch (const std::exception &) {
            }
        } catch (const std::exception &) {
        }
      }
    return false;
  }
  void expand_critical(const Assemblage &a, const std::vector<int> &ids,
                       const std::vector<int> &zeros, int index) {
    // Construct the adjoining solvus arm with a composition-section corrector,
    // then continue away from coalescence until a unique P,T tangent returns.
    auto states = engine.snapshot(a, ids);
    for (int zero : zeros)
      for (std::size_t k = 0; k < ids.size(); ++k)
        if (ids[k] != zero && ids[k] / engine.settings.max_phase_instances ==
                                  zero / engine.settings.max_phase_instances) {
          std::ostringstream key;
          key << "critical:" << index << ':' << zero << ':' << ids[k];
          if (!expanded.insert(key.str()).second)
            continue;
          auto reflected = states;
          auto z = std::find(ids.begin(), ids.end(), zero) - ids.begin();
          reflected[z].amount = states[k].amount;
          reflected[z].composition = states[k].composition;
          reflected[k].amount = 0.;
          reflected[k].composition =
              2. * states[k].composition - states[z].composition;
          auto &phase = engine.phases[states[k].candidate_index];
          if (!phase.solution ||
              (reflected[k].composition.transpose() * phase.occupancies)
                      .minCoeff() < -1.e-10)
            continue;
          int other = ids[k];
          Eigen::VectorXd axis = states[z].composition - states[k].composition;
          double delta = axis.norm();
          bool exact = delta <= 1.e-8 &&
                       result.nodes[index].critical_mode.size() == axis.size();
          if (exact) {
            axis = result.nodes[index].critical_mode;
            delta = .008;
            reflected[z].composition =
                states[k].composition + axis * delta * .5;
            reflected[k].composition =
                states[k].composition - axis * delta * .5;
          } else {
            if (delta <= 1.e-8)
              continue;
            axis /= delta;
          }
          try {
            auto near = engine.make_assemblage(ids, reflected, a.get_pressure(),
                                               a.get_temperature());
            auto nc =
                separation_constraints(*near, ids, other, z, k, axis, delta);
            auto ns = engine.solve(*near, nc);
            if (engine.settings.verbose)
              std::cerr << "Critical adjoining arm " << other << ": "
                        << ns.success
                        << ", distance=" << (pt(*near) - pt(a)).norm() << '\n';
            if (!valid(*near, ns) ||
                separation(engine.snapshot(*near, ids)) < delta * .5 ||
                (!exact &&
                 (pt(*near) - pt(a)).norm() > engine.settings.node_tolerance))
              continue;
            double affinity = engine.stability(*near);
            if (affinity < -engine.settings.affinity_tolerance)
              continue;
            std::vector<BoundaryPoint> prefix;
            if (exact) {
              BoundaryPoint critical;
              critical.pressure = a.get_pressure();
              critical.temperature = a.get_temperature();
              critical.phases = states;
              critical.phases[z].amount = states[k].amount;
              critical.phases[k].amount = 0.;
              critical.mass_balance_error = engine.mass_error(a);
              critical.minimum_affinity = affinity;
              prefix.push_back(std::move(critical));
            }
            prefix.push_back(point(*near, ids, ns, affinity));
            // Retain only verified points. A failed trial starts again from the
            // last accepted compositions, with an enlarged separation section
            // as fallback.
            auto accepted = near;
            auto accepted_solve = ns;
            for (double target = delta * 2.; target <= .04; target *= 2.) {
              auto trial = engine.make_assemblage(
                  ids, engine.snapshot(*accepted, ids),
                  accepted->get_pressure(), accepted->get_temperature());
              Eigen::VectorXd rhs =
                  Eigen::VectorXd::Zero(accepted_solve.x.size());
              auto current = engine.snapshot(*accepted, ids);
              rhs[1] = target - axis.dot(current[z].composition -
                                         current[k].composition);
              Eigen::VectorXd dx = accepted_solve.J.partialPivLu().solve(rhs);
              if (dx.allFinite())
                predict(*trial, accepted_solve.x, dx,
                        get_equilibration_parameters(*trial, engine.bulk, {}));
              auto c = separation_constraints(*trial, ids, other, z, k, axis,
                                              target);
              auto s = engine.solve(*trial, c);
              if (engine.settings.verbose)
                std::cerr << "Critical arm separation " << target << ": "
                          << s.success << '\n';
              if (!valid(*trial, s) ||
                  separation(engine.snapshot(*trial, ids)) < target * .5)
                continue;
              double aff = engine.stability(*trial);
              if (aff < -engine.settings.affinity_tolerance)
                continue;
              prefix.push_back(point(*trial, ids, s, aff));
              accepted = trial;
              accepted_solve = s;
              auto d = tangent(s, *trial);
              if (target >= .008 && d.size() && prefix.size() > 1) {
                auto previous =
                    normalise(prefix[prefix.size() - 2].pressure,
                              prefix[prefix.size() - 2].temperature);
                Seed seed{ids,        other,
                          -2,         engine.snapshot(*trial, ids),
                          pt(*trial), (pt(*trial) - previous).normalized()};
                seed.prefix = std::move(prefix);
                seed.bridge_node = index;
                queue.push_back(std::move(seed));
                break;
              }
            }
          } catch (const std::exception &error) {
            if (engine.settings.verbose)
              std::cerr << "Critical branch recovery: " << error.what() << '\n';
          }
        }
  }
  void expand(const Assemblage &a, const std::vector<int> &ids,
              const std::vector<int> &zeros, int index) {
    std::ostringstream key;
    key << index << ':';
    for (int id : ids)
      key << id << ',';
    key << ':';
    for (int z : zeros)
      key << z << ',';
    if (!expanded.insert(key.str()).second)
      return;
    // A generic double-zero point supplies four branches: each zero phase in
    // the full assemblage, and each zero with the other absent. A broader
    // removal search also discovers polymorph replacements at reduced variance.
    auto states = engine.snapshot(a, ids);
    Eigen::MatrixXd formulae =
        Eigen::MatrixXd::Zero(ids.size(), engine.elements.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      auto f = a.get_phase(i)->get_formula();
      for (std::size_t j = 0; j < engine.elements.size(); ++j)
        if (f.count(engine.elements[j]))
          formulae(i, j) = f.at(engine.elements[j]);
    }
    Eigen::FullPivLU<Eigen::MatrixXd> amounts_rank(formulae);
    amounts_rank.setThreshold(1.e-9);
    bool reduced_amount_variance =
        amounts_rank.rank() < static_cast<int>(ids.size());
    int gibbs_variance =
        a.get_independent_element_indices().size() - ids.size() + 2;
    for (int z : zeros) {
      if (gibbs_variance >= 1)
        queue.push_back(
            {ids, z, index, states, pt(a), Eigen::Vector2d::Zero()});
      for (int removed : ids)
        if (removed != z &&
            (reduced_amount_variance ||
             std::find(zeros.begin(), zeros.end(), removed) != zeros.end())) {
          auto subset = ids;
          subset.erase(std::find(subset.begin(), subset.end(), removed));
          queue.push_back(
              {subset, z, index, states, pt(a), Eigen::Vector2d::Zero()});
        }
    }
  }
  void bracket(WorkState left, WorkState right, int depth = 0) {
    if (!left.state.success || !right.state.success)
      return;
    auto a = active(left.state), b = active(right.state);
    if (a == b)
      return;
    std::vector<int> joined, changed;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(joined));
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                  std::back_inserter(changed));
    if (changed.size() > 2 && depth < 7) {
      auto u = (.5 * (normalise(left.state.pressure, left.state.temperature) +
                      normalise(right.state.pressure, right.state.temperature)))
                   .eval();
      auto actual = (origin + u.cwiseProduct(range)).eval();
      auto mid = engine.stable(actual[0], actual[1]);
      add_sample(mid.state);
      bracket(left, mid, depth + 1);
      bracket(mid, right, depth + 1);
      return;
    }
    auto states = left.state.phases;
    for (auto &s : right.state.phases) {
      auto it = std::find_if(states.begin(), states.end(),
                             [&](auto &old) { return old.id == s.id; });
      if (it == states.end())
        states.push_back(s);
      else {
        it->composition = .5 * (it->composition + s.composition);
        it->amount = .5 * (it->amount + s.amount);
      }
    }
    auto u = (.5 * (normalise(left.state.pressure, left.state.temperature) +
                    normalise(right.state.pressure, right.state.temperature)))
                 .eval();
    auto direction = (normalise(right.state.pressure, right.state.temperature) -
                      normalise(left.state.pressure, left.state.temperature))
                         .normalized()
                         .eval();
    for (int z : changed)
      queue.push_back({joined, z, -1, states, u,
                       Eigen::Vector2d(-direction[1], direction[0])});
  }
  int endpoint(Assemblage &a, const std::vector<int> &ids, int z, int other,
               const Eigen::Vector2d &previous, double maximum_distance,
               optim::roots::DampedNewtonResult *solution = nullptr) {
    current_ids = ids;
    auto prm = get_equilibration_parameters(a, engine.bulk, {});
    auto zi = std::find(ids.begin(), ids.end(), z) - ids.begin(),
         oi = std::find(ids.begin(), ids.end(), other) - ids.begin();
    auto c =
        constraints(std::make_unique<PhaseFractionConstraint>(zi, 0., prm),
                    std::make_unique<PhaseFractionConstraint>(oi, 0., prm));
    auto s = engine.solve(a, c);
    bool verified = valid(a, s), separate = distinct(a, ids);
    std::vector<Minimum> minima;
    double affinity = verified ? engine.stability(a, &minima) : 0.;
    if (verified && separate &&
        affinity < -engine.settings.affinity_tolerance &&
        engine.settings.active_solution_faces) {
      // A new trace site can become active at a double-zero junction too.
      // Preserve both phase-fraction constraints while releasing that face.
      auto refitted = engine.copy_assemblage(a);
      if (refit_composition_faces(refitted, ids, z, previous,
                                  Eigen::Vector2d::Zero(), s, &minima, 1.e-7,
                                  other)) {
        a = *refitted;
        verified = valid(a, s);
        separate = distinct(a, ids);
        affinity = engine.stability(a);
      }
    }
    double displacement = (pt(a) - previous).norm();
    if (!verified || !separate ||
        affinity < -engine.settings.affinity_tolerance ||
        displacement > maximum_distance) {
      if (engine.settings.verbose)
        std::cerr << "Rejected junction zeros=" << z << ',' << other
                  << " verified=" << verified << " distinct=" << separate
                  << " affinity=" << affinity
                  << " displacement=" << displacement
                  << " maximum=" << maximum_distance
                  << " P=" << a.get_pressure() << " T=" << a.get_temperature()
                  << ": " << s.message << '\n';
      return -1;
    }
    if (solution)
      *solution = s;
    return node(a, ids, {z, other}, "junction", &s.J);
  }
  // Bracket phase entry on the already verified boundary before adding the
  // new zero-amount phase. A direct double-zero Newton solve can otherwise
  // jump across a nearby event, or leave the EOS domain while the entering
  // phase's composition is still far from equilibrium.
  std::shared_ptr<Assemblage>
  phase_entry(const Assemblage &previous, const Assemblage &next,
              const std::vector<int> &ids, int zero, int added,
              const Minimum &entering, const Eigen::Vector2d &unit,
              double maximum_distance, int &index,
              optim::roots::DampedNewtonResult &solution) {
    auto original = engine.snapshot(previous, ids);
    auto origin_pt = pt(previous);
    const int base = added / engine.settings.max_phase_instances;
    struct Trial {
      std::shared_ptr<Assemblage> a;
      Minimum minimum;
      Eigen::Vector2d u;
    };
    auto evaluate = [&](const Eigen::Vector2d &u,
                        const std::vector<PhaseState> &warm, Trial &trial) {
      if (!inside(u))
        return false;
      auto location = (origin + u.cwiseProduct(range)).eval();
      try {
        trial.a = engine.make_assemblage(ids, warm, location[0], location[1]);
        current_ids = ids;
        auto c = boundary_constraints(*trial.a, zero, u, unit);
        auto s = engine.solve(*trial.a, c);
        if (!valid(*trial.a, s) || !distinct(*trial.a, ids) ||
            !same_branch(previous, *trial.a, ids))
          return false;
        engine.set_pt(trial.a->get_pressure(), trial.a->get_temperature());
        trial.minimum =
            engine.minimize(base, engine.potentials(*trial.a), entering.p);
        trial.u = pt(*trial.a);
        return true;
      } catch (const std::exception &) {
        return false;
      }
    };
    Trial left, right;
    if (!evaluate(origin_pt, original, left) ||
        !evaluate(pt(next), engine.snapshot(next, ids), right) ||
        right.minimum.affinity >= 0.)
      return {};
    // An accepted point can have a small negative affinity within tolerance.
    // Step back on the same branch to obtain a genuine sign-changing bracket.
    if (left.minimum.affinity < 0.) {
      double distance =
          std::max((right.u - left.u).norm(), engine.settings.min_step);
      for (int k = 0; k < 12 && left.minimum.affinity < 0.; ++k) {
        Trial earlier;
        if (distance > maximum_distance ||
            !evaluate(origin_pt - distance * unit, original, earlier))
          break;
        left = std::move(earlier);
        distance *= 2.;
      }
    }
    if (left.minimum.affinity < 0.)
      return {};
    Trial best =
        std::abs(left.minimum.affinity) < std::abs(right.minimum.affinity)
            ? left
            : right;
    for (int k = 0; k < 60 && std::abs(best.minimum.affinity) > 1.e-7; ++k) {
      double fraction = left.minimum.affinity /
                        (left.minimum.affinity - right.minimum.affinity);
      fraction = std::clamp(fraction, .1, .9);
      Trial trial;
      auto u = (left.u + fraction * (right.u - left.u)).eval();
      auto warm = engine.snapshot(*(fraction < .5 ? left.a : right.a), ids);
      if (!evaluate(u, warm, trial))
        return {};
      if (std::abs(trial.minimum.affinity) < std::abs(best.minimum.affinity))
        best = trial;
      if (trial.minimum.affinity >= 0.)
        left = std::move(trial);
      else
        right = std::move(trial);
      if ((right.u - left.u).norm() < 1.e-11)
        break;
    }
    auto states = engine.snapshot(*best.a, ids);
    PhaseState fresh;
    fresh.id = added;
    fresh.candidate_index = base;
    fresh.composition = best.minimum.p;
    fresh.amount = 0.;
    states.push_back(fresh);
    auto joined = ids;
    joined.push_back(added);
    std::sort(joined.begin(), joined.end());
    try {
      auto junction = engine.make_assemblage(
          joined, states, best.a->get_pressure(), best.a->get_temperature());
      index = endpoint(*junction, joined, zero, added, origin_pt,
                       maximum_distance, &solution);
      if (index >= 0)
        return junction;
    } catch (const std::exception &error) {
      if (engine.settings.verbose)
        std::cerr << "Bracketed phase entry " << added << ": " << error.what()
                  << '\n';
    }
    return {};
  }
  // A solution can activate/deactivate a trace site without changing the
  // phase assemblage. Refit that composition face with equilibrate instead
  // of interpreting the nearby full-polytope minimum as a second phase.
  bool refit_composition_faces(std::shared_ptr<Assemblage> &input,
                               const std::vector<int> &ids, int zero,
                               const Eigen::Vector2d &where,
                               const Eigen::Vector2d &normal,
                               optim::roots::DampedNewtonResult &solved,
                               std::vector<Minimum> *minima = nullptr,
                               double face_tolerance = 1.e-7,
                               int second_zero = -1) {
    if (!engine.settings.active_solution_faces)
      return false;
    auto current = input;
    std::vector<Minimum> candidates = minima ? *minima : std::vector<Minimum>();
    for (int pass = 0; pass < 3; ++pass)
      try {
        auto states = engine.snapshot(*current, ids);
        for (std::size_t k = 0; k < ids.size(); ++k) {
          int base = states[k].candidate_index;
          auto basis = engine.composition_basis(*current->get_phase(k), base);
          if (basis.rows() == basis.cols() || candidates.empty())
            continue;
          auto &minimum = candidates[base];
          if (minimum.affinity < -engine.settings.affinity_tolerance * .05 &&
              (minimum.p - states[k].composition).norm() <
                  engine.settings.composition_tolerance * 100.)
            states[k].composition = minimum.p;
        }
        auto trial =
            engine.make_assemblage(ids, states, current->get_pressure(),
                                   current->get_temperature(), face_tolerance);
        bool changed = false;
        for (std::size_t k = 0; k < ids.size(); ++k) {
          auto old = engine.composition_basis(*current->get_phase(k),
                                              states[k].candidate_index),
               next = engine.composition_basis(*trial->get_phase(k),
                                               states[k].candidate_index);
          changed = changed || old.rows() != next.rows() ||
                    (old.rows() == next.rows() && (old - next).norm() > 1.e-8);
        }
        if (!changed)
          return false;
        current_ids = ids;
        ConstraintList c;
        if (second_zero >= 0) {
          auto prm = get_equilibration_parameters(*trial, engine.bulk, {});
          c = constraints(
              std::make_unique<PhaseFractionConstraint>(
                  std::find(ids.begin(), ids.end(), zero) - ids.begin(), 0.,
                  prm),
              std::make_unique<PhaseFractionConstraint>(
                  std::find(ids.begin(), ids.end(), second_zero) - ids.begin(),
                  0., prm));
        } else
          c = boundary_constraints(*trial, zero, where, normal);
        auto solve = engine.solve(*trial, c);
        if (!valid(*trial, solve) || !distinct(*trial, ids) ||
            (pt(*trial) - where).norm() > engine.settings.step * 3.)
          return false;
        double affinity = engine.stability(*trial, &candidates);
        if (affinity >= -engine.settings.affinity_tolerance) {
          input = trial;
          solved = solve;
          if (minima)
            *minima = candidates;
          return true;
        }
        current = trial;
      } catch (const std::exception &) {
        return false;
      }
    return false;
  }
  // Each direction returns points after the seed, including an exact endpoint.
  std::vector<BoundaryPoint> follow(std::shared_ptr<Assemblage> a,
                                    const Seed &seed,
                                    optim::roots::DampedNewtonResult solve,
                                    Eigen::VectorXd direction, int &end_node,
                                    std::string &termination) {
    // An unsuccessful endpoint must not inherit the seed's junction ID: that
    // would snap an unfinished curve back to its starting node when plotting.
    end_node = -1;
    std::vector<BoundaryPoint> points;
    double step = engine.settings.step;
    auto ids = seed.ids;
    double travelled = 0.;
    Eigen::Vector2d initial_unit = direction.head<2>().cwiseQuotient(range);
    for (int iteration = 0; iteration < engine.settings.max_trace_steps;
         ++iteration) {
      auto old_x = solve.x;
      auto old_u = pt(*a);
      current_ids = ids;
      double requested = step;
      // Predict the first phase-out event from amount derivatives.
      auto prm = get_equilibration_parameters(*a, engine.bulk, {});
      int leaving = -1;
      double event_step = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < ids.size(); ++k)
        if (ids[k] != seed.zero) {
          int j = prm.phase_amount_indices[k];
          if (direction[j] < -1.e-12) {
            double distance = -old_x[j] / direction[j];
            if (distance >= -engine.settings.min_step &&
                distance < event_step) {
              event_step = std::max(0., distance);
              leaving = ids[k];
            }
          }
        }
      if (leaving >= 0 && event_step < requested * 1.1) {
        auto event_a = engine.copy_assemblage(*a);
        try {
          auto delta = (std::max(0., event_step) * direction).eval();
          if (old_x[0] + delta[0] >= 0 && old_x[1] + delta[1] > 0)
            predict(*event_a, old_x, delta, prm);
          optim::roots::DampedNewtonResult event_s;
          int n = endpoint(
              *event_a, ids, seed.zero, leaving, old_u,
              std::max(requested * 3., engine.settings.node_tolerance * 2.),
              &event_s);
          if (n >= 0) {
            end_node = n;
            points.push_back(
                point(*event_a, ids, event_s, engine.stability(*event_a)));
            termination = "junction";
            return points;
          }
        } catch (const std::exception &) {
        }
        if (event_step < engine.settings.min_step) {
          // A nearly inactive composition coordinate can make the projected
          // amount derivatives singular. Verify a neighbouring face before
          // interpreting that derivative as an instantaneous phase-out event.
          Eigen::Vector2d event_unit = direction.head(2).cwiseQuotient(range);
          if (refit_composition_faces(a, ids, seed.zero, old_u, event_unit,
                                      solve, nullptr, 5.e-7)) {
            auto refitted = tangent(solve, *a);
            if (refitted.size()) {
              if (refitted.head(2).cwiseQuotient(range).dot(event_unit) < 0.)
                refitted = -refitted;
              direction = refitted;
              continue;
            }
          }
          termination = "unresolved phase-out event";
          return points;
        }
        requested = std::min(requested, event_step * .7);
      }
      auto unit = direction.head(2).cwiseQuotient(range).eval();
      int border_axis = -1;
      double border = 0., border_step = requested;
      for (int k = 0; k < 2; ++k)
        if (std::abs(unit[k]) > 1.e-12) {
          double edge = unit[k] > 0 ? 1. : 0.,
                 distance = (edge - old_u[k]) / unit[k];
          if (distance >= -1.e-8 && distance <= border_step) {
            border_step = std::max(0., distance);
            border_axis = k;
            border = edge;
          }
        }
      std::shared_ptr<Assemblage> next;
      optim::roots::DampedNewtonResult next_s;
      bool accepted = false;
      double used = border_axis >= 0 ? border_step : requested;
      std::string failure;
      bool merged = false;
      for (int retry = 0; retry < 16; ++retry) {
        if (used < engine.settings.min_step && border_axis < 0)
          break;
        next = engine.copy_assemblage(*a);
        try {
          predict(*next, old_x, used * direction, prm);
          ConstraintList c;
          current_ids = ids;
          if (border_axis >= 0) {
            Eigen::Vector2d n = Eigen::Vector2d::Zero(), u = old_u;
            n[border_axis] = 1;
            u[border_axis] = border;
            c = boundary_constraints(*next, seed.zero, u, n);
          } else
            c = boundary_constraints(*next, seed.zero, old_u + used * unit,
                                     unit);
          next_s = engine.solve(*next, c);
          merged = !distinct(*next, ids) || !same_branch(*a, *next, ids);
          accepted = valid(*next, next_s) && !merged &&
                     (pt(*next) - old_u).norm() <=
                         std::max(used * 3, engine.settings.node_tolerance);
          failure = next_s.message;
          if (accepted)
            break;
        } catch (const std::exception &error) {
          failure = error.what();
        }
        used *= .5;
        border_axis = -1;
      }
      if (!accepted && next) {
        auto target =
            (old_u + std::max(used, engine.settings.min_step) * unit).eval();
        if (refit_composition_faces(next, ids, seed.zero, target, unit,
                                    next_s)) {
          accepted = true;
          border_axis = -1;
        }
      }
      if (!accepted) {
        if (merged && separation(engine.snapshot(*a, ids)) < .01) {
          approach_critical(a, ids, seed.zero, solve, points);
          end_node = node(*a, ids, {seed.zero}, "critical_point", &solve.J);
          termination = "solution critical point";
        } else
          termination = "corrector failed";
        if (engine.settings.verbose)
          std::cerr << "Corrector stopped at P=" << a->get_pressure()
                    << ", T=" << a->get_temperature() << ": " << failure
                    << '\n';
        return points;
      }
      std::vector<Minimum> ms;
      double affinity = engine.stability(*next, &ms);
      if (engine.settings.active_solution_faces) {
        Eigen::Vector2d target = old_u + used * unit, normal = unit;
        if (border_axis >= 0) {
          target[border_axis] = border;
          normal.setZero();
          normal[border_axis] = 1.;
        }
        if (refit_composition_faces(next, ids, seed.zero, target, normal,
                                    next_s, &ms))
          affinity = engine.stability(*next, &ms);
      }
      if (affinity < -engine.settings.affinity_tolerance) {
        // A previously absent phase becomes stable: add it at zero amount and
        // solve the two zero constraints to locate the node, then spawn
        // branches.
        bool located = false;
        for (std::size_t base = 0; base < ms.size(); ++base)
          if (ms[base].affinity < -engine.settings.affinity_tolerance) {
            int added = base * engine.settings.max_phase_instances;
            while (std::find(ids.begin(), ids.end(), added) != ids.end() &&
                   added < (static_cast<int>(base) + 1) *
                               engine.settings.max_phase_instances)
              ++added;
            if (added >= (static_cast<int>(base) + 1) *
                             engine.settings.max_phase_instances)
              continue;
            auto joined = ids;
            joined.push_back(added);
            std::sort(joined.begin(), joined.end());
            auto states = engine.snapshot(*a, ids);
            PhaseState fresh;
            fresh.id = added;
            fresh.candidate_index = base;
            fresh.composition = ms[base].p;
            fresh.amount = 0.;
            states.push_back(fresh);
            // The accepted state may already be slightly beyond zero affinity:
            // stability permits a finite numerical tolerance. Halving the step
            // must not shrink the endpoint acceptance radius below node
            // tolerance.
            try {
              auto junction = engine.make_assemblage(
                  joined, states, a->get_pressure(), a->get_temperature());
              optim::roots::DampedNewtonResult event_s;
              int n = endpoint(
                  *junction, joined, seed.zero, added, old_u,
                  std::max(used * 3., engine.settings.node_tolerance * 2.),
                  &event_s);
              if (n >= 0) {
                auto p = point(*junction, joined, event_s,
                               engine.stability(*junction));
                points.push_back(p);
                end_node = n;
                located = true;
                break;
              }
            } catch (const std::exception &error) {
              if (engine.settings.verbose)
                std::cerr << "Junction candidate " << added << " on zero "
                          << seed.zero << " failed: " << error.what() << '\n';
            }
            if (!located) {
              int n = -1;
              optim::roots::DampedNewtonResult event_s;
              auto junction = phase_entry(
                  *a, *next, ids, seed.zero, added, ms[base], unit,
                  std::max(used * 3., engine.settings.node_tolerance * 2.), n,
                  event_s);
              if (junction) {
                points.push_back(point(*junction, joined, event_s,
                                       engine.stability(*junction)));
                end_node = n;
                located = true;
                break;
              }
            }
          }
        current_ids = ids;
        if (located) {
          termination = "junction";
          return points;
        }
        step = used * .5;
        if (step < engine.settings.min_step) {
          termination = "unresolved phase-in or solvus event";
          return points;
        }
        continue;
      }
      auto u = pt(*next);
      travelled += (u - old_u).norm();
      points.push_back(point(*next, ids, next_s, affinity));
      if (border_axis >= 0) {
        end_node = node(*next, ids, {seed.zero}, "domain_edge", &next_s.J);
        termination = "domain edge";
        return points;
      }
      // A small step near a solvus is not a closed loop. Require an actual
      // return segment, consistent compositions and an aligned outgoing
      // tangent.
      auto displacement = (u - old_u).eval();
      double projection = displacement.squaredNorm()
                              ? std::clamp((seed.pt - old_u).dot(displacement) /
                                               displacement.squaredNorm(),
                                           0., 1.)
                              : 0.;
      bool return_segment =
          (old_u + projection * displacement - seed.pt).norm() <
          engine.settings.node_tolerance;
      bool same_compositions = true;
      auto now = engine.snapshot(*next, ids);
      for (std::size_t i = 0; i < ids.size(); ++i)
        if ((now[i].composition - seed.phases[i].composition).norm() >
            engine.settings.composition_tolerance * 100.)
          same_compositions = false;
      if (seed.node == -1 && points.size() > 6 &&
          travelled > engine.settings.step * 3. && return_segment &&
          same_compositions && unit.dot(initial_unit) > .5) {
        termination = "closed loop";
        return points;
      }
      auto new_direction = tangent(next_s, *next);
      if (!new_direction.size()) {
        bool critical = separation(engine.snapshot(*next, ids)) < .01;
        if (critical)
          approach_critical(next, ids, seed.zero, next_s, points);
        end_node =
            node(*next, ids, {seed.zero},
                 critical ? "critical_point" : "reduced_variance", &next_s.J);
        termination = critical ? "solution critical point" : "reduced variance";
        return points;
      }
      if (new_direction.head(2).cwiseQuotient(range).dot(unit) < 0)
        new_direction = -new_direction;
      a = next;
      solve = next_s;
      direction = new_direction;
      step = std::min(engine.settings.step, used * 1.4);
    }
    termination = "trace step limit";
    return points;
  }
  void label(Boundary &line) {
    if (line.points.size() < 2)
      return;
    double length = 0.;
    for (std::size_t k = 1; k < line.points.size(); ++k)
      length +=
          (normalise(line.points[k].pressure, line.points[k].temperature) -
           normalise(line.points[k - 1].pressure,
                     line.points[k - 1].temperature))
              .norm();
    // Label away from nodes, where a normal perturbation crosses only this
    // edge. At a junction the same perturbation can cross several fields.
    // Choose the geometric midpoint and correct it onto the edge. On short
    // branches the middle stored point can still be the starting junction.
    double walked = 0.;
    std::size_t segment = 1;
    for (; segment + 1 < line.points.size(); ++segment) {
      double span = (normalise(line.points[segment].pressure,
                               line.points[segment].temperature) -
                     normalise(line.points[segment - 1].pressure,
                               line.points[segment - 1].temperature))
                        .norm();
      if (walked + span >= length * .5)
        break;
      walked += span;
    }
    auto left = normalise(line.points[segment - 1].pressure,
                          line.points[segment - 1].temperature);
    auto right = normalise(line.points[segment].pressure,
                           line.points[segment].temperature);
    auto along = (right - left).eval();
    double span = along.norm();
    Eigen::Vector2d centre =
        span > 0
            ? left + std::clamp((length * .5 - walked) / span, 0., 1.) * along
            : left;
    if (span < 1.e-10)
      along = Eigen::Vector2d(1., 0.);
    else
      along /= span;
    auto location = (origin + centre.cwiseProduct(range)).eval();
    auto anchor_a =
        engine.make_assemblage(line.assemblage, line.points[segment - 1].phases,
                               location[0], location[1]);
    current_ids = line.assemblage;
    auto anchor_constraints =
        boundary_constraints(*anchor_a, line.zero_phase, centre, along);
    auto anchor_solve = engine.solve(*anchor_a, anchor_constraints);
    BoundaryPoint anchor;
    if (valid(*anchor_a, anchor_solve)) {
      centre = pt(*anchor_a);
      anchor = point(*anchor_a, line.assemblage, anchor_solve,
                     engine.stability(*anchor_a));
    } else {
      anchor = line.points[line.points.size() / 2];
      centre = normalise(anchor.pressure, anchor.temperature);
    }
    Eigen::Vector2d side(-along[1], along[0]);
    std::vector<std::vector<int>> side_assemblages = {line.assemblage};
    auto absent = line.assemblage;
    absent.erase(std::find(absent.begin(), absent.end(), line.zero_phase));
    if (!absent.empty())
      side_assemblages.push_back(absent);
    // At a phase replacement, two stoichiometrically indistinguishable
    // amounts are interchangeable at coexistence. Try the opposite removal.
    Eigen::MatrixXd formulae =
        Eigen::MatrixXd::Zero(line.assemblage.size(), engine.elements.size());
    for (std::size_t k = 0; k < anchor.phases.size(); ++k) {
      auto &ph = anchor.phases[k];
      auto it =
          std::find(line.assemblage.begin(), line.assemblage.end(), ph.id);
      if (it == line.assemblage.end())
        continue;
      formulae.row(it - line.assemblage.begin()) =
          ph.composition.transpose() * engine.phases[ph.candidate_index].a;
    }
    Eigen::FullPivLU<Eigen::MatrixXd> variance(formulae);
    variance.setThreshold(1.e-9);
    if (variance.rank() < formulae.rows())
      for (int removed : line.assemblage)
        if (removed != line.zero_phase) {
          auto subset = line.assemblage;
          subset.erase(std::find(subset.begin(), subset.end(), removed));
          if (!subset.empty())
            side_assemblages.push_back(subset);
        }
    std::array<State, 2> side_states;
    for (int retry = 0; retry < 16; ++retry) {
      double offset = std::max(engine.settings.node_tolerance * 4, 1.e-3) *
                      std::pow(.5, retry);
      for (int sign : {-1, 1}) {
        auto u = (centre + sign * offset * side).eval();
        if (!inside(u))
          continue;
        int slot = sign < 0 ? 0 : 1;
        side_states[slot] = State{};
        auto actual = (origin + u.cwiseProduct(range)).eval();
        for (auto &subset : side_assemblages)
          try {
            auto work =
                engine.fixed_pt(subset, anchor.phases, actual[0], actual[1]);
            if (!work.state.success || !distinct(*work.assemblage, work.ids))
              continue;
            work.state.message = "Stable neighbouring field";
            side_states[slot] = work.state;
            break;
          } catch (const std::exception &) {
          }
      }
      if (side_states[0].success && side_states[1].success &&
          active(side_states[0]) != active(side_states[1]))
        break;
    }
    if (side_states[0].success) {
      line.side_a = active(side_states[0]);
      add_sample(side_states[0]);
    }
    if (side_states[1].success) {
      line.side_b = active(side_states[1]);
      add_sample(side_states[1]);
    }
    if (line.side_a.empty() || line.side_b.empty() ||
        line.side_a == line.side_b)
      result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                   ": neighbouring fields could not be "
                                   "distinguished within tolerances.");
  }
  bool same_curve(const Boundary &first, const Boundary &second) {
    auto near = [&](const Boundary &one, const Boundary &two) {
      for (auto &p : one.points) {
        auto u = normalise(p.pressure, p.temperature);
        double minimum = std::numeric_limits<double>::infinity();
        for (std::size_t k = 1; k < two.points.size(); ++k) {
          auto a = normalise(two.points[k - 1].pressure,
                             two.points[k - 1].temperature),
               b = normalise(two.points[k].pressure, two.points[k].temperature);
          auto d = (b - a).eval();
          double f = d.squaredNorm()
                         ? std::clamp((u - a).dot(d) / d.squaredNorm(), 0., 1.)
                         : 0.;
          minimum = std::min(minimum, (u - a - f * d).norm());
        }
        if (minimum > engine.settings.node_tolerance * 2.)
          return false;
      }
      return true;
    };
    if (near(first, second) && near(second, first))
      return true;
    // Coarse chords of the same curved equilibrium line can be farther apart
    // than node tolerance. Verify equivalence by correcting the other saved
    // state onto each interior point's section, including its compositions.
    auto verified = [&](const Boundary &one, const Boundary &two) {
      int checked = 0;
      for (std::size_t i = 1; i + 1 < one.points.size(); ++i) {
        auto &target = one.points[i];
        if (separation(target.phases) < .002)
          continue;
        auto u = normalise(target.pressure, target.temperature);
        double minimum = std::numeric_limits<double>::infinity();
        std::size_t closest = 0;
        for (std::size_t k = 0; k < two.points.size(); ++k) {
          auto &p = two.points[k];
          double distance = (u - normalise(p.pressure, p.temperature)).norm();
          if (distance < minimum) {
            minimum = distance;
            closest = k;
          }
        }
        if (minimum > engine.settings.step * 3.)
          return false;
        auto &left = one.points[i - 1];
        auto &right = one.points[i + 1];
        Eigen::Vector2d normal = normalise(right.pressure, right.temperature) -
                                 normalise(left.pressure, left.temperature);
        if (normal.norm() < 1.e-12)
          continue;
        normal.normalize();
        try {
          auto a =
              engine.make_assemblage(two.assemblage, two.points[closest].phases,
                                     target.pressure, target.temperature);
          current_ids = two.assemblage;
          auto c = boundary_constraints(*a, two.zero_phase, u, normal);
          auto s = engine.solve(*a, c);
          if (!valid(*a, s) || !distinct(*a, two.assemblage) ||
              (pt(*a) - u).norm() > engine.settings.node_tolerance)
            return false;
          auto states = engine.snapshot(*a, two.assemblage);
          for (auto &ph : states) {
            auto it = std::find_if(target.phases.begin(), target.phases.end(),
                                   [&](auto &old) { return old.id == ph.id; });
            if (it == target.phases.end() ||
                (it->composition - ph.composition).norm() >
                    engine.settings.composition_tolerance * 100.)
              return false;
          }
          ++checked;
        } catch (const std::exception &) {
          return false;
        }
      }
      return checked > 0;
    };
    return verified(first, second) && verified(second, first);
  }
  void trace(const Seed &seed) {
    if (covered(seed))
      return;
    current_ids = seed.ids;
    try {
      auto actual = (origin + seed.pt.cwiseProduct(range)).eval();
      std::vector<Eigen::Vector2d> normals;
      if (seed.direction.norm() > 0)
        normals.push_back(seed.direction);
      normals.push_back(Eigen::Vector2d(0., 1.));
      normals.push_back(Eigen::Vector2d(1., 0.));
      std::shared_ptr<Assemblage> a;
      optim::roots::DampedNewtonResult s;
      Eigen::VectorXd d;
      double affinity = 0.;
      bool accepted = false;
      // At a node the new branch direction is initially unknown. Try both
      // coordinate sections: one can duplicate an equilibrium equation (for
      // example fixing T on a horizontal edge), leaving a singular corrector.
      for (auto &normal : normals)
        try {
          a = engine.make_assemblage(seed.ids, seed.phases, actual[0],
                                     actual[1]);
          auto c = boundary_constraints(*a, seed.zero, seed.pt, normal);
          s = engine.solve(*a, c);
          if (!valid(*a, s) || !distinct(*a, seed.ids) ||
              (seed.node >= 0 &&
               (pt(*a) - seed.pt).norm() > engine.settings.node_tolerance))
            continue;
          d = tangent(s, *a);
          if (!d.size())
            continue;
          affinity = engine.stability(*a);
          if (affinity < -engine.settings.affinity_tolerance)
            continue;
          accepted = true;
          break;
        } catch (const std::exception &) {
        }
      if (!accepted)
        return;
      Seed corrected = seed;
      corrected.pt = pt(*a);
      corrected.phases = engine.snapshot(*a, seed.ids);
      if (covered(corrected))
        return;
      Boundary line;
      line.id = result.boundaries.size();
      line.zero_phase = seed.zero;
      line.assemblage = seed.ids;
      line.start_node = seed.node;
      line.end_node = seed.node;
      auto unit = d.head(2).cwiseQuotient(range).eval();
      std::string forward_status, backward_status;
      auto backward_a = engine.make_assemblage(
          seed.ids, corrected.phases, a->get_pressure(), a->get_temperature());
      if (seed.bridge_node >= 0 && unit.dot(seed.direction) < 0.)
        d = -d;
      auto forward = follow(a, corrected, s, d, line.end_node, forward_status);
      std::vector<BoundaryPoint> backward;
      if (seed.bridge_node >= 0) {
        line.start_node = seed.bridge_node;
        backward_status = "solution critical point";
      } else if (forward_status == "closed loop")
        backward_status = "closed loop";
      else
        backward = follow(backward_a, corrected, s, -d, line.start_node,
                          backward_status);
      std::reverse(backward.begin(), backward.end());
      line.points = std::move(backward);
      line.points.push_back(point(*backward_a, seed.ids, s, affinity));
      // The seed must retain its corrected state, not the final backward state.
      auto seed_a = engine.make_assemblage(seed.ids, corrected.phases,
                                           actual[0], actual[1]);
      seed_a->set_state(s.x[0], s.x[1]);
      line.points.back() = point(*seed_a, seed.ids, s, affinity);
      line.points.insert(line.points.end(), forward.begin(), forward.end());
      if (seed.bridge_node >= 0)
        line.points.insert(line.points.begin(), seed.prefix.begin(),
                           seed.prefix.end() - 1);
      if (forward_status == "closed loop")
        line.points.push_back(line.points.front());
      line.termination = backward_status + "; " + forward_status;
      if (line.points.size() < 2)
        return;
      double length = 0.;
      for (std::size_t k = 1; k < line.points.size(); ++k)
        length +=
            (normalise(line.points[k].pressure, line.points[k].temperature) -
             normalise(line.points[k - 1].pressure,
                       line.points[k - 1].temperature))
                .norm();
      if (length < engine.settings.min_step * 2.)
        return;
      label(line);
      // Polymorph replacement can be represented by either zero amount; keep
      // one geometrical edge and one incident line at its nodes.
      for (auto &old : result.boundaries) {
        bool same_sides =
            (old.side_a == line.side_a && old.side_b == line.side_b) ||
            (old.side_a == line.side_b && old.side_b == line.side_a);
        bool same_ends = line.start_node >= 0 && line.end_node >= 0 &&
                         ((old.start_node == line.start_node &&
                           old.end_node == line.end_node) ||
                          (old.start_node == line.end_node &&
                           old.end_node == line.start_node));
        if (same_sides && same_ends && old.assemblage == line.assemblage &&
            same_curve(old, line))
          return;
      }
      for (int n : {line.start_node, line.end_node})
        if (n >= 0) {
          auto &incident = result.nodes[n].incident_lines;
          if (std::find(incident.begin(), incident.end(), line.id) ==
              incident.end())
            incident.push_back(line.id);
        }
      if (backward_status.find("unresolved") != std::string::npos ||
          forward_status.find("unresolved") != std::string::npos ||
          backward_status == "corrector failed" ||
          forward_status == "corrector failed" ||
          backward_status == "trace step limit" ||
          forward_status == "trace step limit")
        result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                     ": " + line.termination);
      result.boundaries.push_back(std::move(line));
    } catch (const std::exception &e) {
      if (engine.settings.verbose)
        std::cerr << "Rejected boundary seed: " << e.what() << '\n';
    }
  }
  void search() {
    int seeds = 0;
    const int limit = engine.settings.max_lines * 30;
    while (!queue.empty() &&
           result.boundaries.size() <
               static_cast<std::size_t>(engine.settings.max_lines) &&
           seeds < limit) {
      auto seed = canonical_seed(queue.front());
      queue.pop_front();
      std::ostringstream key;
      for (int id : seed.ids)
        key << id << ',';
      key << ':' << seed.zero << ':' << seed.node;
      if (seed.node >= 0 && !attempted.insert(key.str()).second)
        continue;
      trace(seed);
      ++seeds;
      if (engine.settings.verbose)
        std::cerr << "Traced " << result.boundaries.size() << " lines; "
                  << queue.size() << " seeds remaining\n";
    }
  }
  void recover() {
    for (int pass = 0; pass < engine.settings.max_recovery_passes; ++pass) {
      bool progress = false;
      const std::size_t count = result.boundaries.size();
      for (std::size_t index = 0; index < count; ++index) {
        auto &line = result.boundaries[index];
        auto [backward, forward] = statuses(line);
        for (bool back : {false, true}) {
          auto &status = back ? backward : forward;
          if (!line.points.empty()) {
            auto &p = back ? line.points.front() : line.points.back();
            if (model_excluded(p.pressure, p.temperature))
              continue;
          }
          // Older saved traces could finish at an identical-copy root and call
          // it a junction. Such a point does not describe two coexisting
          // phases.
          if (status != "solution critical point" && !line.points.empty() &&
              separation(
                  (back ? line.points.front() : line.points.back()).phases) <=
                  engine.settings.composition_tolerance * 10.) {
            status = "reduced variance";
            (back ? line.start_node : line.end_node) = -1;
          }
          bool critical = status == "solution critical point";
          if (critical) {
            int n = back ? line.start_node : line.end_node;
            if (n >= 0 && result.nodes[n].incident_lines.size() > 1 &&
                result.nodes[n].critical_mode.size())
              continue;
          }
          if (!unfinished(status) && status != "reduced variance" && !critical)
            continue;
          if (line.points.size() < 2)
            continue;
          auto original = back ? line.points.front() : line.points.back();
          auto adjacent =
              back ? line.points[1] : line.points[line.points.size() - 2];
          Eigen::Vector2d hint =
              normalise(original.pressure, original.temperature) -
              normalise(adjacent.pressure, adjacent.temperature);
          // A former rank stop may have been a trivial root with identical
          // solution copies. Rewind to the last genuinely distinct state.
          while (
              line.points.size() > 2 &&
              separation(
                  (back ? line.points.front() : line.points.back()).phases) <=
                  engine.settings.composition_tolerance * 10.) {
            if (back)
              line.points.erase(line.points.begin());
            else
              line.points.pop_back();
          }
          while (line.points.size() > 1) {
            auto &p = back ? line.points.front() : line.points.back();
            auto &q =
                back ? line.points[1] : line.points[line.points.size() - 2];
            if ((normalise(p.pressure, p.temperature) -
                 normalise(q.pressure, q.temperature))
                    .norm() > 1.e-12)
              break;
            if (back)
              line.points.erase(line.points.begin());
            else
              line.points.pop_back();
          }
          auto saved = back ? line.points.front() : line.points.back();
          Eigen::Vector2d u = normalise(saved.pressure, saved.temperature);
          Eigen::Vector2d along = hint;
          if (line.points.size() > 1) {
            auto &neighbour =
                back ? line.points[1] : line.points[line.points.size() - 2];
            along = u - normalise(neighbour.pressure, neighbour.temperature);
          }
          if (along.norm() < 1.e-12)
            continue;
          along.normalize();
          if (unfinished(status))
            (back ? line.start_node : line.end_node) = -1;
          try {
            auto a = engine.make_assemblage(line.assemblage, saved.phases,
                                            saved.pressure, saved.temperature);
            current_ids = line.assemblage;
            auto c = boundary_constraints(*a, line.zero_phase, u, along);
            auto s = engine.solve(*a, c);
            if (!valid(*a, s) ||
                engine.stability(*a) < -engine.settings.affinity_tolerance)
              continue;
            bool near_critical = false;
            if (unfinished(status) || status == "reduced variance") {
              auto states = engine.snapshot(*a, line.assemblage);
              auto z = std::find_if(
                  states.begin(), states.end(),
                  [&](const PhaseState &p) { return p.id == line.zero_phase; });
              if (z != states.end())
                for (auto &other : states)
                  if (other.id != z->id &&
                      other.candidate_index == z->candidate_index &&
                      (other.composition - z->composition).norm() < .04)
                    near_critical = true;
            }
            if (critical || near_critical) {
              std::vector<BoundaryPoint> extra;
              if (approach_critical(a, line.assemblage, line.zero_phase, s,
                                    extra)) {
                int n = node(*a, line.assemblage, {line.zero_phase},
                             "critical_point", &s.J);
                if (back) {
                  std::reverse(extra.begin(), extra.end());
                  line.points.insert(line.points.begin(), extra.begin(),
                                     extra.end());
                } else
                  line.points.insert(line.points.end(), extra.begin(),
                                     extra.end());
                (back ? line.start_node : line.end_node) = n;
                status = "solution critical point";
                progress = true;
                continue;
              }
              if (critical)
                continue;
            }
            auto d = tangent(s, *a);
            if (!d.size())
              continue;
            if (d.head<2>().cwiseQuotient(range).dot(along) < 0.)
              d = -d;
            Seed seed{line.assemblage,
                      line.zero_phase,
                      -2,
                      engine.snapshot(*a, line.assemblage),
                      pt(*a),
                      along};
            int end = -1;
            std::string ended;
            auto extra = follow(a, seed, s, d, end, ended);
            if (!extra.empty()) {
              progress = true;
              if (back) {
                std::reverse(extra.begin(), extra.end());
                line.points.insert(line.points.begin(), extra.begin(),
                                   extra.end());
              } else
                line.points.insert(line.points.end(), extra.begin(),
                                   extra.end());
              status = ended;
              (back ? line.start_node : line.end_node) = end;
            } else if (end >= 0) {
              progress = true;
              status = ended;
              (back ? line.start_node : line.end_node) = end;
            }
            if (engine.settings.verbose)
              std::cerr << "Recovered boundary " << line.id
                        << (back ? " start: " : " end: ") << extra.size()
                        << " accepted points; " << ended << '\n';
          } catch (const std::exception &error) {
            if (engine.settings.verbose)
              std::cerr << "Recovery of boundary " << line.id << ": "
                        << error.what() << '\n';
          }
        }
        line.termination = backward + "; " + forward;
      }
      search();
      if (!progress)
        break;
    }
  }
  State field_state(const Eigen::Vector2d &location) {
    std::vector<std::pair<double, std::size_t>> nearby;
    for (std::size_t i = 0; i < result.samples.size(); ++i)
      if (result.samples[i].success) {
        auto &s = result.samples[i];
        nearby.push_back({(normalise(s.pressure, s.temperature) -
                           normalise(location[0], location[1]))
                              .squaredNorm(),
                          i});
      }
    std::sort(nearby.begin(), nearby.end());
    for (std::size_t i = 0; i < std::min<std::size_t>(32, nearby.size()); ++i)
      try {
        auto &previous = result.samples[nearby[i].second];
        std::vector<int> ids;
        for (auto &ph : previous.phases)
          ids.push_back(ph.id);
        auto work =
            engine.fixed_pt(ids, previous.phases, location[0], location[1]);
        if (!work.state.success || !distinct(*work.assemblage, work.ids))
          continue;
        work.state.message =
            "Closed field verified from a neighbouring equilibrium state";
        return work.state;
      } catch (const std::exception &) {
      }
    return engine.stable(location[0], location[1]).state;
  }
  bool model_excluded(double pressure, double temperature) const {
    Eigen::Vector2d p = normalise(pressure, temperature);
    for (auto &region : result.excluded_regions) {
      bool inside = false;
      for (int i = 1; i < region.rows(); ++i) {
        auto a = normalise(region(i - 1, 0), region(i - 1, 1)),
             b = normalise(region(i, 0), region(i, 1));
        if ((a[1] > p[1]) != (b[1] > p[1]) &&
            p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
          inside = !inside;
      }
      if (inside)
        return true;
    }
    return false;
  }
  void join_frame_limits() {
    // A corrector can already have accepted a frame point before an amount
    // null direction obscures the outgoing P,T tangent (especially at T=0).
    // Classify that accepted endpoint; do not extend an interior curve to it.
    for (auto &line : result.boundaries) {
      if (line.points.empty())
        continue;
      auto [back, forward] = statuses(line);
      for (bool start : {true, false}) {
        auto &status = start ? back : forward;
        auto &id = start ? line.start_node : line.end_node;
        if (!unfinished(status))
          continue;
        auto &point = start ? line.points.front() : line.points.back();
        auto u = normalise(point.pressure, point.temperature);
        if (!u.allFinite() || !inside(u, 1.e-10) ||
            std::min(u.cwiseAbs().minCoeff(),
                     (u - Eigen::Vector2d::Ones()).cwiseAbs().minCoeff()) >
                1.e-10 ||
            point.mass_balance_error > engine.settings.mass_balance_tolerance ||
            point.minimum_affinity < -engine.settings.affinity_tolerance)
          continue;
        id = -1;
        for (auto &previous : result.nodes)
          if (previous.kind == "domain_edge" &&
              (normalise(previous.pressure, previous.temperature) - u).norm() <
                  1.e-10) {
            id = previous.id;
            break;
          }
        if (id < 0) {
          Node n;
          n.id = result.nodes.size();
          n.kind = "domain_edge";
          n.pressure = point.pressure;
          n.temperature = point.temperature;
          n.assemblage = line.assemblage;
          n.zero_phases = {line.zero_phase};
          auto a = engine.make_assemblage(line.assemblage, point.phases,
                                          point.pressure, point.temperature);
          n.gibbs_variance = a->get_independent_element_indices().size() -
                             line.assemblage.size() + 2;
          result.nodes.push_back(n);
          id = n.id;
        }
        status = "domain edge";
      }
      line.termination = back + "; " + forward;
    }
  }
  void join_model_limits() {
    for (auto &line : result.boundaries) {
      auto [back, forward] = statuses(line);
      for (bool start : {true, false}) {
        auto &id = start ? line.start_node : line.end_node;
        if (line.points.empty() ||
            (id >= 0 && result.nodes.at(id).kind != "model_domain_limit"))
          continue;
        auto &point = start ? line.points.front() : line.points.back();
        auto u = normalise(point.pressure, point.temperature);
        double distance = std::numeric_limits<double>::infinity();
        int region_index = -1, segment = -1;
        for (std::size_t r = 0; r < result.excluded_regions.size(); ++r) {
          auto &ring = result.excluded_regions[r];
          // Exclude the top/frame closure: only the actual EOS limit can
          // receive a terminated phase line. Keep the last thermodynamically
          // verified point, adjusting the plotted domain envelope within node
          // tolerance.
          for (int i = 1; i < ring.rows() - 2; ++i) {
            auto a = normalise(ring(i - 1, 0), ring(i - 1, 1)),
                 b = normalise(ring(i, 0), ring(i, 1));
            auto delta = (b - a).eval();
            double f =
                delta.squaredNorm()
                    ? std::clamp((u - a).dot(delta) / delta.squaredNorm(), 0.,
                                 1.)
                    : 0.;
            double d = (u - a - f * delta).norm();
            if (d < distance) {
              distance = d;
              region_index = r;
              segment = i;
            }
          }
        }
        if (region_index < 0 || distance > engine.settings.node_tolerance * 2.)
          continue;
        auto &ring = result.excluded_regions[region_index];
        bool attached = false;
        for (int i = 0; i < ring.rows(); ++i)
          if ((normalise(ring(i, 0), ring(i, 1)) - u).norm() < 1.e-12)
            attached = true;
        if (!attached) {
          Eigen::MatrixXd updated(ring.rows() + 1, 2);
          updated.topRows(segment) = ring.topRows(segment);
          updated.row(segment) << point.pressure, point.temperature;
          updated.bottomRows(ring.rows() - segment) =
              ring.bottomRows(ring.rows() - segment);
          ring = std::move(updated);
        }
        // Recomputed EOS envelopes must retain connections from a saved run
        // too.
        if (id < 0) {
          Node n;
          n.id = result.nodes.size();
          n.kind = "model_domain_limit";
          n.pressure = point.pressure;
          n.temperature = point.temperature;
          n.zero_phases = {line.zero_phase};
          n.assemblage = line.assemblage;
          result.nodes.push_back(n);
          id = n.id;
        }
        (start ? back : forward) = "model domain limit";
      }
      line.termination = back + "; " + forward;
    }
  }
  // Find the lower-pressure EOS feasibility limit. This is a boundary of
  // the admissible model domain, not a zero-amount phase-equilibrium line.
  void model_domain() {
    if (!engine.settings.exclude_invalid_eos ||
        !result.excluded_regions.empty())
      return;
    auto feasible = [&](double pn, double tn) {
      return engine.eos_bulk_feasible(origin[0] + pn * range[0],
                                      origin[1] + tn * range[1]);
    };
    auto lower_pressure = [&](double tn) {
      if (feasible(0., tn))
        return 0.;
      if (!feasible(1., tn))
        return 1.;
      double low = 0., high = 1.;
      for (int i = 0; i < 35; ++i) {
        double middle = .5 * (low + high);
        if (feasible(middle, tn))
          high = middle;
        else
          low = middle;
      }
      return .5 * (low + high);
    };
    std::vector<Eigen::Vector2d> curve;
    auto flush = [&]() {
      if (curve.empty())
        return;
      std::vector<Eigen::Vector2d> ring{{0., curve.front()[1]}};
      ring.insert(ring.end(), curve.begin(), curve.end());
      ring.push_back({0., curve.back()[1]});
      ring.push_back(ring.front());
      Eigen::MatrixXd physical(ring.size(), 2);
      for (std::size_t i = 0; i < ring.size(); ++i)
        physical.row(i) = (origin + ring[i].cwiseProduct(range)).transpose();
      result.excluded_regions.push_back(std::move(physical));
      curve.clear();
    };
    double previous = lower_pressure(0.);
    if (previous > 0.)
      curve.push_back({previous, 0.});
    for (int j = 1; j <= 64; ++j) {
      double t = double(j) / 64., left = double(j - 1) / 64.,
             p = lower_pressure(t);
      if ((previous == 0.) != (p == 0.)) {
        double low = left, high = t;
        for (int i = 0; i < 32; ++i) {
          double mid = .5 * (low + high);
          bool exists = lower_pressure(mid) > 0.;
          if (exists == (p > 0.))
            high = mid;
          else
            low = mid;
        }
        Eigen::Vector2d edge(0., .5 * (low + high));
        if (p > 0.)
          curve.push_back(edge);
        else {
          curve.push_back(edge);
          flush();
        }
      }
      if (p > 0.)
        curve.push_back({p, t});
      previous = p;
    }
    flush();
  }
  Result finish() {
    separate_junctions();
    model_domain();
    join_frame_limits();
    join_model_limits();
    for (auto &state : result.samples)
      if (model_excluded(state.pressure, state.temperature)) {
        state.success = false;
        state.outside_model_domain = true;
        state.message = "Outside the required EOS model domain.";
      }
    if (!result.excluded_regions.empty()) {
      auto states = std::move(result.samples);
      result.samples.clear();
      result.fields.clear();
      field_index.clear();
      for (auto &state : states)
        add_sample(state);
    }
    result.boundaries.erase(
        std::remove_if(result.boundaries.begin(), result.boundaries.end(),
                       [&](const Boundary &line) {
                         return !line.points.empty() &&
                                std::all_of(line.points.begin(),
                                            line.points.end(),
                                            [&](const BoundaryPoint &p) {
                                              return model_excluded(
                                                  p.pressure, p.temperature);
                                            });
                       }),
        result.boundaries.end());
    for (auto &line : result.boundaries)
      trim_junction_overshoot(line);
    // An early LP seed can fail even when the subsequently discovered field
    // supplies an accurate warm start. Reverify that location instead of
    // retaining a stale failure after the boundary construction has succeeded.
    const std::size_t sample_count = result.samples.size();
    for (std::size_t i = 0; i < sample_count; ++i)
      if (!result.samples[i].success &&
          !result.samples[i].outside_model_domain) {
        auto &old = result.samples[i];
        auto verified =
            field_state(Eigen::Vector2d(old.pressure, old.temperature));
        if (verified.success)
          add_sample(verified, i);
      }
    for (auto &line : result.boundaries)
      if (line.side_a.empty() || line.side_b.empty() ||
          line.side_a == line.side_b) {
        try {
          label(line);
        } catch (const std::exception &error) {
          if (engine.settings.verbose)
            std::cerr << "Field-label recovery: " << error.what() << '\n';
        }
      }
    // Resuming overlapping partial traces can complete several records of the
    // same physical edge. Keep one polyline so chord differences do not create
    // artificial sliver fields or inflate the number of branches at a node.
    std::vector<Boundary> unique;
    for (auto &line : result.boundaries) {
      bool duplicate = false;
      if (line.start_node >= 0 && line.end_node >= 0)
        for (auto &old : unique) {
          bool same_sides =
              (old.side_a == line.side_a && old.side_b == line.side_b) ||
              (old.side_a == line.side_b && old.side_b == line.side_a);
          bool same_ends = (old.start_node == line.start_node &&
                            old.end_node == line.end_node) ||
                           (old.start_node == line.end_node &&
                            old.end_node == line.start_node);
          if (!same_sides || !same_ends || old.assemblage != line.assemblage)
            continue;
          if (same_curve(line, old)) {
            duplicate = true;
            if (line.points.size() > old.points.size())
              old = std::move(line);
            break;
          }
        }
      if (!duplicate)
        unique.push_back(std::move(line));
    }
    result.boundaries = std::move(unique);
    for (std::size_t i = 0; i < result.boundaries.size(); ++i)
      result.boundaries[i].id = i;
    result.diagnostics.clear();
    for (auto &n : result.nodes)
      n.incident_lines.clear();
    for (auto &line : result.boundaries) {
      auto [back, forward] = statuses(line);
      if (unfinished(back))
        line.start_node = -1;
      if (unfinished(forward))
        line.end_node = -1;
      if (unfinished(back) || unfinished(forward))
        result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                     ": " + line.termination);
      if (line.side_a.empty() || line.side_b.empty() ||
          line.side_a == line.side_b)
        result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                     ": neighbouring fields could not be "
                                     "distinguished within tolerances.");
      for (int n : {line.start_node, line.end_node})
        if (n >= 0) {
          auto &incident = result.nodes.at(n).incident_lines;
          if (std::find(incident.begin(), incident.end(), line.id) ==
              incident.end())
            incident.push_back(line.id);
        }
    }
    for (auto &state : result.samples)
      if (!state.success && !state.outside_model_domain)
        result.diagnostics.push_back("Unresolved seed: " + state.message);
    // A coarse curved trace can put conflicting edge labels around a tiny
    // closed face. Verify its interior through native Gibbs minimisation rather
    // than guessing a colour or assemblage. Open faces remain unresolved.
    model_domain();
    auto geometry = field_polygons(result, 1.e-8, true, false);
    for (std::size_t i = 0; i < geometry.polygons.size(); ++i) {
      auto &polygon = geometry.polygons[i];
      if (polygon.outside_model_domain || polygon.has_open_boundary ||
          (polygon.n_phases > 0 && !polygon.phases.empty()) ||
          polygon.label_clearance <= 0.)
        continue;
      auto state = field_state(polygon.label_position);
      if (state.success) {
        state.is_field_verification = true;
        add_sample(state);
      } else
        result.diagnostics.push_back(
            "Could not verify the assemblage of closed region " +
            std::to_string(i) + ": " + state.message);
    }
    // Report the geometry after interior verification. Successful endpoint
    // solves alone do not guarantee a closed, identified planar subdivision.
    auto verified_geometry = field_polygons(result, 1.e-8, true, false);
    result.diagnostics.insert(result.diagnostics.end(),
                              verified_geometry.diagnostics.begin(),
                              verified_geometry.diagnostics.end());
    for (auto &n : result.nodes)
      if (n.kind == "critical_point" && n.incident_lines.size() == 1)
        result.diagnostics.push_back(
            "Unpaired solution critical endpoint at node " +
            std::to_string(n.id) + "; its adjoining field remains unresolved.");
    if (!queue.empty())
      result.diagnostics.push_back(
          "Boundary search limit reached; increase max_lines.");
    if (result.fields.size() > 1 && result.boundaries.empty())
      result.diagnostics.push_back(
          "Field changes detected but no boundaries converged.");
    if (result.fields.size() > 1) {
      std::set<std::vector<int>> bordered;
      for (auto &line : result.boundaries) {
        bordered.insert(line.side_a);
        bordered.insert(line.side_b);
      }
      for (auto &field : result.fields)
        if (!bordered.count(field.phases))
          result.diagnostics.push_back(
              "No field edge resolved for sampled field " +
              std::to_string(field.id) +
              "; increase seed density or inspect failed solves.");
    }
    result.equilibrium_solves += engine.equilibrium_solves;
    result.minimization_calls += engine.minimization_calls;
    result.resolved = result.diagnostics.empty();
    return result;
  }

public:
  Tracer(const types::FormulaMap &bulk,
         const std::vector<std::shared_ptr<Material>> &phases,
         const std::array<double, 2> &pr, const std::array<double, 2> &tr,
         const Settings &settings)
      : engine(bulk, phases, settings) {
    for (auto r : {pr, tr})
      if (!std::isfinite(r[0]) || !std::isfinite(r[1]) || r[1] <= r[0])
        throw std::invalid_argument(
            "P/T ranges must be finite and increasing.");
    if (pr[0] < 0 || tr[0] < 0 ||
        (tr[0] == 0 && !settings.active_solution_faces))
      throw std::invalid_argument("Pseudosections require P>=0 and T>0 (T=0 "
                                  "requires active_solution_faces).");
    result.pressure_range = pr;
    result.temperature_range = tr;
    origin << pr[0], tr[0];
    range << pr[1] - pr[0], tr[1] - tr[0];
    result.settings = engine.settings;
    for (auto &phase : engine.phases)
      for (int j = 0; j < settings.max_phase_instances; ++j)
        result.phase_names.push_back(phase.material->get_name() +
                                     (j ? " #" + std::to_string(j + 1) : ""));
  }
  Result run() {
    auto &opts = engine.settings;
    for (int i = 0; i < opts.pressure_seeds; ++i)
      for (int j = 0; j < opts.temperature_seeds; ++j) {
        auto actual =
            (origin + Eigen::Vector2d(double(i) / (opts.pressure_seeds - 1),
                                      double(j) / (opts.temperature_seeds - 1))
                          .cwiseProduct(range))
                .eval();
        auto state = engine.stable(actual[0], actual[1]);
        add_sample(state.state);
        grid.push_back(std::move(state));
        if (opts.verbose)
          std::cerr << "Seed " << grid.size() << '/'
                    << opts.pressure_seeds * opts.temperature_seeds << " "
                    << grid.back().state.message << '\n';
      }
    for (int i = 0; i < opts.pressure_seeds; ++i)
      for (int j = 0; j < opts.temperature_seeds; ++j) {
        int k = i * opts.temperature_seeds + j;
        if (i + 1 < opts.pressure_seeds)
          bracket(grid[k], grid[k + opts.temperature_seeds]);
        if (j + 1 < opts.temperature_seeds)
          bracket(grid[k], grid[k + 1]);
      }
    search();
    recover();
    return finish();
  }
  Result resume(const Result &previous) {
    if (previous.phase_names != result.phase_names)
      throw std::invalid_argument(
          "Resume with the same candidate names and max_phase_instances.");
    result = previous;
    result.settings = engine.settings;
    separate_junctions();
    if (!engine.settings.required_eos_phases.empty())
      result.excluded_regions.clear();
    model_domain();
    join_frame_limits();
    join_model_limits();
    auto samples = std::move(result.samples);
    result.samples.clear();
    result.fields.clear();
    for (auto &state : samples)
      add_sample(state);
    for (std::size_t i = 0; i < result.nodes.size(); ++i)
      if (result.nodes[i].id != static_cast<int>(i))
        throw std::invalid_argument("Saved node IDs must be contiguous.");
    // Successful endpoint states seed all untraced branches at their junction.
    for (auto &line : result.boundaries) {
      if (line.points.empty())
        continue;
      auto [back, forward] = statuses(line);
      for (bool start : {true, false}) {
        auto status = start ? back : forward;
        int id = start ? line.start_node : line.end_node;
        if ((status != "junction" && status != "solution critical point") ||
            id < 0)
          continue;
        auto &point = start ? line.points.front() : line.points.back();
        auto &n = result.nodes.at(id);
        if ((normalise(point.pressure, point.temperature) -
             normalise(n.pressure, n.temperature))
                .norm() > engine.settings.node_tolerance)
          continue;
        auto a = engine.make_assemblage(n.assemblage, point.phases, n.pressure,
                                        n.temperature);
        if (status == "junction")
          expand(*a, n.assemblage, n.zero_phases, id);
        // Critical endpoints are first approached from both arms in recover().
      }
    }
    recover();
    search();
    return finish();
  }
};
} // namespace
Result pseudosection(const types::FormulaMap &bulk,
                     const std::vector<std::shared_ptr<Material>> &phases,
                     const std::array<double, 2> &pr,
                     const std::array<double, 2> &tr,
                     const Settings &settings) {
  return Tracer(bulk, phases, pr, tr, settings).run();
}
Result
refine_pseudosection(const types::FormulaMap &bulk,
                     const std::vector<std::shared_ptr<Material>> &phases,
                     const Result &previous, const Settings &settings) {
  return Tracer(bulk, phases, previous.pressure_range,
                previous.temperature_range, settings)
      .resume(previous);
}
Result
refine_pseudosection(const types::FormulaMap &bulk,
                     const std::vector<std::shared_ptr<Material>> &phases,
                     const Result &previous) {
  return refine_pseudosection(bulk, phases, previous, previous.settings);
}
} // namespace burnman::pseudosections
