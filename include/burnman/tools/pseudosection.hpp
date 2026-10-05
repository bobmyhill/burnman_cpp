/* Native isochemical phase-boundary continuation, GPL v3 or later. */
#pragma once
#include "burnman/core/assemblage.hpp"
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace burnman::pseudosections {

struct Settings {
  int pressure_seeds = 7, temperature_seeds = 7;
  int max_refinement_iterations = 150, minimization_starts = 10;
  int max_phase_instances = 3, max_trace_steps = 500, max_lines = 1000;
  int max_recovery_passes = 2;
  // Steps and node distances are fractions of the supplied P/T domain.
  double step = .025, min_step = 1.e-7;
  // Affinities are J/mol; mass balance and phase amounts use relative errors.
  double affinity_tolerance = .2, mass_balance_tolerance = 1.e-8;
  double amount_tolerance = 1.e-7, composition_tolerance = 1.e-5;
  double node_tolerance = 2.e-4;
  bool verbose = false;
  // Restrict the calculation to candidates with a stable EOS root, recording
  // every exclusion. Only a diagnosed SLB domain error is excluded; other
  // failures propagate. Set false to require every candidate EOS to be valid.
  bool exclude_invalid_eos = true;
  // Equilibrate on active zero-occupancy faces and validate against the full
  // solution polytopes (KKT conditions). Useful at very low temperature.
  bool active_solution_faces = true;
  // Conservatively mask points where an essential candidate model is outside
  // its EOS domain, even if the remaining candidates can represent the bulk.
  std::vector<std::string> required_eos_phases;
};

struct PhaseState {
  int id = -1, candidate_index = -1;
  std::string name;
  double amount = 0.; // moles, on the bulk's supplied amount scale
  Eigen::VectorXd composition;
};

struct State {
  double pressure = 0., temperature = 0.; // Pa, K
  bool success = false;
  // A solve explicitly requested at the interior of a constructed face.
  bool is_field_verification = false;
  bool outside_model_domain = false;
  std::string message;
  std::vector<std::string>
      excluded_phases; // EOS-domain diagnostics at this P,T
  std::vector<PhaseState> phases;
  double gibbs = 0.; // total J, on the bulk's supplied amount scale
  double mass_balance_error = 0., minimum_affinity = 0., equilibrium_error = 0.;
};

struct BoundaryPoint {
  double pressure = 0., temperature = 0.;
  std::vector<PhaseState> phases;
  double mass_balance_error = 0., minimum_affinity = 0., residual = 0.;
};

struct Boundary {
  int id = -1, zero_phase = -1, start_node = -1, end_node = -1;
  // Verified exchange of distinct solution compositions at unchanged counts.
  bool is_solution_replacement = false;
  std::vector<int> assemblage, side_a, side_b;
  std::vector<BoundaryPoint> points;
  std::string termination;
};

struct Node {
  int id = -1, gibbs_variance = 0, pt_nullity = 0;
  std::string kind;
  double pressure = 0., temperature = 0.;
  std::vector<int> zero_phases, assemblage, incident_lines;
  Eigen::VectorXd critical_mode; // normalised endmember direction, if verified
};

struct Field {
  int id = -1;
  std::vector<int> phases, sample_indices;
};

struct Result {
  std::array<double, 2> pressure_range, temperature_range;
  // Actual settings, retained so continuation uses the same composition
  // coordinates, numerical tolerances and EOS-domain policy by default.
  Settings settings;
  std::vector<std::string> phase_names;
  std::vector<State> samples;
  std::vector<Field> fields;
  std::vector<Boundary> boundaries;
  std::vector<Node> nodes;
  std::vector<std::string> diagnostics;
  // Closed SI [P,T] rings outside the admissible EOS model domain: either
  // remaining candidates cannot represent the bulk or a required EOS fails.
  // These are model limits, not equilibrium phase lines.
  std::vector<Eigen::MatrixXd> excluded_regions;
  bool resolved = false;
  int equilibrium_solves = 0, minimization_calls = 0;
};

struct FieldPolygon {
  int field_id = -1, n_phases = 0, sample_index = -1;
  bool has_open_boundary = false;
  bool outside_model_domain = false;
  std::vector<int> phases;
  // Zero-based faces in the original, unmerged planar subdivision.
  std::vector<int> source_regions;
  // Closed rings, columns [pressure (Pa), temperature (K)]. Holes have
  // opposite winding to the exterior. Area is a fraction of the P/T domain.
  Eigen::MatrixXd vertices;
  std::vector<Eigen::MatrixXd> holes;
  double area = 0.;
  // Interior label point with approximate maximum clearance, in SI [P,T].
  Eigen::Vector2d label_position = Eigen::Vector2d::Zero();
  double label_clearance = 0.; // normalised distance to the closest ring
  /// True only when the entire axis-aligned rectangle lies strictly inside
  /// the exterior and outside all holes. Centre and half-size are SI [P,T].
  bool contains_rectangle(const Eigen::Vector2d &centre,
                          const Eigen::Vector2d &half_size) const;
};

struct FieldPolygons {
  std::vector<FieldPolygon> polygons;
  std::vector<std::string> diagnostics;
  // Visible, intersection-split edges in SI [P,T], excluding dissolved
  // internal edges. Unfinished and unclassified edges remain visible.
  std::vector<Eigen::MatrixXd> boundary_segments;
  std::vector<int> boundary_nodes;
};

/// Assemble planar faces from the traced polylines, retaining holes and
/// disconnected regions. Optionally use the calculation frame to close edge
/// fields. No equilibrium solves or raster interpolation are performed.
/// Unknown counts have n_phases=0; ambiguous assemblages have empty phases.
/// Both cases have a diagnostic.
/// Adjacent identified faces with identical phase IDs are merged by default;
/// disconnected regions and different solution multiplicities stay separate.
/// source_regions retains original indices, also used by diagnostics.
FieldPolygons field_polygons(const Result &result, double tolerance = 1.e-8,
                             bool close_domain = true,
                             bool merge_fields = true);

/// Closed elemental bulk, SI P/T. Native Gibbs LP and multistart tangent-plane
/// minimisation find stable seeds; equilibrate() solves and continues zero-
/// amount boundaries. Nodes generate one-phase-in/out and phase-swap branches;
/// rank and stability checks prune reduced-variance/metastable branches.
/// Finite seed sampling does not guarantee discovery of every disconnected
/// field. Inspect resolved and diagnostics; increase seed density to assess it.
Result pseudosection(const types::FormulaMap &composition,
                     const std::vector<std::shared_ptr<Material>> &candidates,
                     const std::array<double, 2> &pressure_range,
                     const std::array<double, 2> &temperature_range,
                     const Settings &settings = Settings{});

/// Resume unfinished boundaries from their last verified compositions and
/// revisit junction branches. Supply the same bulk, candidates and phase-ID
/// settings as the original calculation. Existing accepted points are retained.
Result
refine_pseudosection(const types::FormulaMap &composition,
                     const std::vector<std::shared_ptr<Material>> &candidates,
                     const Result &previous, const Settings &settings);

/// Resume with the settings retained in the previous result.
Result
refine_pseudosection(const types::FormulaMap &composition,
                     const std::vector<std::shared_ptr<Material>> &candidates,
                     const Result &previous);

/// Same native stability search at a single P,T, useful for checking diagrams.
State stable_equilibrium(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates, double pressure,
    double temperature, const Settings &settings = Settings{});

} // namespace burnman::pseudosections
