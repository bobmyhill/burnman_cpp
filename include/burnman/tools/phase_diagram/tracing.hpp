// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

#pragma once
#include "burnman/tools/phase_diagram/refinement.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include <deque>
#include <map>
#include <set>

namespace burnman::phase_diagrams::detail {
/// State and reusable stages for one native diagram calculation.
/// calculation.cpp gives the master sequence; tracing.cpp implements
/// reaction following, junction discovery and recovery.
class Tracer {
  // Starting estimate for one reaction branch: assemblage, zero-amount phase,
  // mineral compositions and a direction in the rescaled diagram.
  struct Seed {
    std::vector<int> ids;
    int zero = -1, node = -1;
    PhaseStates phases;
    // Location and direction after rescaling each diagram axis to 0..1.
    Eigen::Vector2d position, direction;
    std::vector<BoundaryPoint> prefix{};
    int bridge_node = -1;
    // Retain the physical state even when P/T are not selected diagram axes.
    std::optional<Eigen::Vector2d> physical = std::nullopt;
  };

  Engine engine;
  Result result;
  Eigen::Vector2d origin, range;
  std::deque<Seed> queue;
  std::map<std::vector<int>, int> field_index;
  std::vector<WorkState> grid;
  std::map<std::string, std::vector<PhaseStates>> attempted;
  std::set<std::string> expanded;
  std::map<std::vector<int>, Eigen::MatrixXd> amount_vertices;
  Eigen::VectorXd critical_axis;
  std::optional<LineResolution> line_resolution;
  std::set<int> unstable_lines;
  std::vector<std::string> thermodynamic_diagnostics;
  Eigen::Vector2d normalise(double first, double second) const;
  Eigen::Vector2d normalise(const Eigen::Vector2d &q) const;
  template <typename Point> Eigen::Vector2d normalise(const Point &p) const {
    return normalise(diagram_coordinates(p, result.section.type));
  }
  // Position in the selected diagram after rescaling each axis to 0..1.
  Eigen::Vector2d normalised_coordinates(const Assemblage &a) const;
  bool inside(const Eigen::Vector2d &u, double margin = 1.e-8) const;
  std::unique_ptr<EqualityConstraint>
  section(const Assemblage &a, int n, const Eigen::Vector2d &u,
          const Eigen::Vector2d &normal) const;
  ConstraintList boundary_constraints(const Assemblage &a,
                                      const std::vector<int> &ids, int zero,
                                      const Eigen::Vector2d &u,
                                      const Eigen::Vector2d &normal);
  EquilibriumSolve solve_boundary(Assemblage &a, const std::vector<int> &ids,
                                  int zero, const Eigen::Vector2d &position,
                                  const Eigen::Vector2d &normal);
  bool valid_amount_edge(const Assemblage &a, const std::vector<int> &ids,
                         int zero);
  // Find equilibrium changes that move along a line in the selected diagram.
  // Changes in phase amounts can move S/V even at constant P,T.
  Eigen::VectorXd tangent(const EquilibriumSolve &s, const Assemblage &a,
                          int *coordinate_rank = nullptr);
  BoundaryPoint point(const Assemblage &a, const std::vector<int> &ids,
                      const EquilibriumSolve &solve, double affinity);
  // A stationary solution composition can be a saddle even when its affinity
  // lies inside the discovery tolerance. Check local stability in every
  // section; competing-phase affinity is checked separately.
  bool valid(const Assemblage &a, const EquilibriumSolve &s) const;
  bool needs_face_refit(double affinity, bool composition = false) const;
  // Locate weak, distinct solvus minima before their branch loses stability.
  double phase_entry_tolerance(std::size_t base, const PhaseStates &states,
                               const Minimum &minimum) const;
  // A zero-amount phase still identifies a particular solution minimum.
  // Sharing the basin check with junction and face-refit solves prevents a
  // small residual from accepting a different saturation branch.
  bool same_branch(const Assemblage &a, const Assemblage &b,
                   const std::vector<int> &ids) const;
  bool near_third_copy_event(const std::vector<int> &ids, int zero, int other,
                             const PhaseStates &states) const;
  bool closing_solvus(const Assemblage &a, const std::vector<int> &ids,
                      int zero, const Eigen::VectorXd &direction,
                      const EquilibrationParameters &prm) const;
  // Predict compositions as well as P,T, shortening only the compositional
  // displacement when it would cross a linear site-occupancy face. Signed
  // ordering endmembers need site bounds, rather than endmember clipping.
  void predict(Assemblage &a, const Eigen::VectorXd &x,
               const Eigen::VectorXd &delta,
               const EquilibrationParameters &prm);
  static bool composition_order(const PhaseState &a, const PhaseState &b);
  Seed canonical_seed(const Seed &original) const;
  // Store an equilibrium and group it by present assemblage. These groups are
  // thermodynamic identities, not connected areas: one assemblage may recur.
  void add_sample(const State &state, int existing = -1);
  bool covered(const Seed &seed) const;
  std::vector<int> event_signature(const std::vector<int> &ids,
                                   bool polymorphs = false) const;
  bool same_event(const Node &previous, const std::vector<int> &ids,
                  const std::vector<int> &zero) const;
  int node(Assemblage &a, const std::vector<int> &ids,
           const std::vector<int> &zero, const std::string &kind,
           const Eigen::MatrixXd *jacobian = nullptr);
  void separate_junctions();
  // Fix the projection of the composition difference instead of P or T. This
  // excludes the trivial identical-copy root at a solvus and remains useful
  // when the P,T projection of the equilibrium null space loses rank.
  ConstraintList separation_constraints(const Assemblage &a,
                                        const std::vector<int> &ids, int zero,
                                        int first, int second,
                                        const Eigen::VectorXd &axis,
                                        double target);
  bool approach_critical(std::shared_ptr<Assemblage> &a,
                         const std::vector<int> &ids, int zero,
                         EquilibriumSolve &solve,
                         std::vector<BoundaryPoint> &points);
  void expand_critical(const Assemblage &a, const std::vector<int> &ids,
                       const std::vector<int> &zeros, int index);
  // A junction's zero-amount choices define possible outgoing reactions.
  // Queue their trial branches; later corrector/stability tests decide
  // acceptance.
  void expand(const Assemblage &a, const std::vector<int> &ids,
              const std::vector<int> &zeros, int index);
  // Narrow an interval whose sampled equilibria have different assemblages,
  // then seed the intervening phase-appearance or solution-replacement
  // reactions.
  void bracket(WorkState left, WorkState right, int depth = 0);
  // Set output affinity/solution only on acceptance. A rejected trial can still
  // change a, so preserve the last accepted assemblage before trying the solve.
  int endpoint(Assemblage &a, const std::vector<int> &ids, int z, int other,
               const PhaseStates &previous_phases,
               const Eigen::Vector2d &previous, double maximum_distance,
               double &accepted_affinity, EquilibriumSolve *solution = nullptr,
               const Eigen::Vector2d &direction = Eigen::Vector2d::Zero());
  // Bracket phase entry on the already verified boundary before adding the
  // new zero-amount phase. A direct double-zero Newton solve can otherwise
  // jump across a nearby event, or leave the EOS domain while the entering
  // phase's composition is still far from equilibrium.
  std::shared_ptr<Assemblage>
  phase_entry(const Assemblage &previous, const Assemblage &next,
              const std::vector<int> &ids, int zero, int added,
              const Minimum &entering, const Eigen::Vector2d &unit,
              double maximum_distance, double &accepted_affinity, int &index,
              EquilibriumSolve &solution);
  // A solution can activate/deactivate a trace site without changing the
  // phase assemblage. Refit that composition face with equilibrate instead
  // of interpreting the nearby full-polytope minimum as a second phase.
  // Return affinity and minima only for the accepted state.
  bool refit_composition_faces(
      std::shared_ptr<Assemblage> &input, const std::vector<int> &ids, int zero,
      const Eigen::Vector2d &where, const Eigen::Vector2d &normal,
      EquilibriumSolve &solved, double &accepted_affinity,
      std::vector<Minimum> *minima = nullptr, double face_tolerance = 1.e-7,
      int second_zero = -1);
  // Follow one reaction direction with predictor-corrector equilibrium solves.
  // Phase entry/exit, a critical point or a domain edge can terminate the
  // branch. Return accepted points after the seed, including a solved endpoint
  // when found.
  std::vector<BoundaryPoint> follow(std::shared_ptr<Assemblage> a,
                                    const Seed &seed, EquilibriumSolve solve,
                                    Eigen::VectorXd direction, int &end_node,
                                    std::string &termination);
  // Identify the stable assemblage on each side of a solved reaction segment.
  // Use the local equilibrium response; closed-loop perimeter constraints
  // provide missing neighbours when the reaction is degenerate.
  void label(Boundary &line);
  bool same_curve(const Boundary &first, const Boundary &second,
                  bool partial = false, bool require_equilibrium = false);
  bool complete_boundary(const Boundary &line) const;
  // P-T duplicates keep one coherent sampling; interleaving independently
  // corrected points creates zigzags. Composition-section resumption also
  // preserves the user's accepted saved points when completing an arc.
  void retain_curve(Boundary &kept, Boundary &other);
  // Correct a queued reaction estimate, follow it in both directions, and
  // retain its accepted states, endpoint identities and neighbouring
  // assemblages.
  void trace(const Seed &seed);
  // Process queued reaction branches, including new ones discovered at
  // junctions.
  void search();
  // Retry incomplete boundaries and junction branches from accepted equilibria.
  // This reuses mineral compositions already found in the current diagram.
  void recover();
  // Replace a coarse chord with an additional solved reaction point. A midpoint
  // in the drawing is only a starting estimate; it need not lie on the true
  // curve.
  bool refine_segment(Boundary &line, std::size_t right_index,
                      bool reconnect = false);
  // Repair an unsuccessful seed using the discovered neighbouring fields.
  State field_state(const Eigen::Vector2d &location);
  bool model_excluded(double first, double second) const;
  template <typename Point> bool model_excluded(const Point &point) const {
    const auto q = diagram_coordinates(point, result.section.type);
    return model_excluded(q[0], q[1]);
  }
  // Find the lower-pressure EOS feasibility limit. This is a boundary of
  // the admissible model domain, not a zero-amount phase-equilibrium line.
  void model_domain();
  // Resuming overlapping partial traces can complete several records of the
  // same physical edge. Keep one polyline so chord differences do not create
  // artificial sliver fields or inflate the number of branches at a node.
  void reconcile_boundaries(bool transfer_labels_only = false);
  // Renumber retained reaction records, preserving the set selected for
  // retracing.
  void index_boundaries();
  // Complete frame/EOS-limit endpoints and exclude states outside the models'
  // admissible domain. Model validity limits are distinct from phase reactions.
  void apply_model_domain();
  void remove_retraced_spurs();
  void recover_failed_seeds();
  // Check the reaction graph and refine its accepted arcs. Suspect arcs are
  // returned once to the ordinary tracing queue, then checked by the same
  // process. Accepted, unrelated boundaries retain their saved equilibria.
  void resolve_boundaries();
  // Record incomplete trace ends and failed equilibrium samples, rebuilding
  // node incidence from the surviving lines rather than stale trace history.
  void collect_boundary_diagnostics();
  // With requested line resolution, add interior equilibrium evidence only for
  // closed fields whose labels remain provisional or unknown after perimeter
  // tests.
  void verify_field_interiors();
  // Refresh critical-endpoint certificates and closed-field diagnostics, then
  // set resolved from the outstanding messages. This is completion of the
  // implemented checks, not a proof that every stability field was discovered.
  Result certify_result();
  // Evaluate the initial assemblage grid and bracket changes between
  // neighbours. These equilibria seed reaction tracing; they are not pixels
  // used for filling.
  void seed_boundaries();
  // Require the saved bulk, phase order and phase-instance settings to agree
  // with this calculation before reusing its equilibrium records.
  void validate_previous_identity(const Result &previous) const;
  // Restore accepted states and reaction identities for a resumed calculation.
  // Return whether junction branches need searching as well as boundary
  // recovery.
  bool prepare_previous(const Result &previous,
                        const std::optional<LineResolution> &resolution);

public:
  Tracer(const types::FormulaMap &bulk,
         const std::vector<std::shared_ptr<Material>> &phases,
         const std::array<double, 2> &pr, const std::array<double, 2> &tr,
         const Settings &settings, const CompositionSection &section);
  // Resolved diagrams only gain solved points. Unresolved saved diagrams and
  // fresh calculations share recovery, repair and field-identification stages.
  Result
  calculate(const Result *previous = nullptr,
            const std::optional<LineResolution> &resolution = std::nullopt);
};
} // namespace burnman::phase_diagrams::detail
