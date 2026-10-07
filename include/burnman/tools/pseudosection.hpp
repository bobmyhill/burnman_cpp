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
#include "burnman/core/assemblage.hpp"
#include "burnman/tools/equilibration/equality_constraint_base.hpp"
#include "burnman/tools/equilibration/equilibrate_types.hpp"
#include <Eigen/Dense>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace burnman::pseudosections {

// Native records use diagram_axes() order, regardless of plot orientation.
// P/T are Pa/K; S/V are total J/K and m^3 on the supplied elemental bulk scale.
// X linearly interpolates endpoint amounts without normalising their totals.
enum class Coordinate { P, T, S, V, X };
enum class DiagramType { PT, PX, TX, PS, PV, TS, TV, SV, SX, VX };

inline std::array<Coordinate, 2> diagram_axes(DiagramType type) {
  using C = Coordinate;
  switch (type) {
  case DiagramType::PT:
    return {C::P, C::T};
  case DiagramType::PX:
    return { C::P, C::X };
  case DiagramType::TX:
    return { C::T, C::X };
  case DiagramType::PS:
    return {C::P, C::S};
  case DiagramType::PV:
    return {C::P, C::V};
  case DiagramType::TS:
    return {C::T, C::S};
  case DiagramType::TV:
    return {C::T, C::V};
  case DiagramType::SV:
    return {C::S, C::V};
  case DiagramType::SX:
    return { C::S, C::X };
  case DiagramType::VX:
    return { C::V, C::X };
  }
  throw std::invalid_argument("Unknown diagram type.");
}
inline bool has_composition_axis(DiagramType type) {
  return diagram_axes(type)[1] == Coordinate::X;
}

struct CompositionSection {
  DiagramType type = DiagramType::PT;
  // Bulk(X) = (1-X) bulk_start + X composition_end, retaining supplied amounts.
  types::FormulaMap composition_end;
  std::array<double, 2> composition_range{0., 1.};
  // Extensive entropy (J/K) and volume (m^3), on the supplied bulk scale.
  std::array<double, 2> entropy_range{0., 1.}, volume_range{1.e-6, 1.e-4};
  // An X section fixes one thermodynamic coordinate outside its axes.
  // Omitted for legacy PX/TX sections: infer fixed T/P from its range.
  std::optional<Coordinate> fixed_coordinate;
  double fixed_value = 0.;
};

template <typename Point>
double coordinate_value(const Point &point, Coordinate axis) {
  switch (axis) {
  case Coordinate::P:
    return point.pressure;
  case Coordinate::T:
    return point.temperature;
  case Coordinate::S:
    return point.entropy;
  case Coordinate::V:
    return point.volume;
  case Coordinate::X:
    return point.composition_coordinate;
  }
  throw std::invalid_argument("Unknown diagram coordinate.");
}

// Geometry uses the selected pair; P,T always retain their physical values.
template <typename Point>
Eigen::Vector2d diagram_coordinates(const Point &point, DiagramType type) {
  const auto axes = diagram_axes(type);
  return {coordinate_value(point, axes[0]), coordinate_value(point, axes[1])};
}

struct Settings {
  // Seed counts apply to their respective selected diagram axes.
  int pressure_seeds = 7, temperature_seeds = 7;
  int composition_seeds = 7;
  int entropy_seeds = 7, volume_seeds = 7;
  int max_refinement_iterations = 150, minimization_starts = 10;
  int max_phase_instances = 3, max_trace_steps = 500, max_lines = 1000;
  int max_recovery_passes = 2;
  // Steps and node distances are fractions of the supplied diagram domain.
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
  // Use fewer endmembers when some sites are empty, while still testing all
  // allowed compositions for lower Gibbs energy. Useful at low temperature.
  bool active_solution_faces = true;
  // Conservatively mask points where an essential candidate model is outside
  // its EOS domain, even if the remaining candidates can represent the bulk.
  std::vector<std::string> required_eos_phases;
};

struct PhaseState {
  // id = candidate_index * settings.max_phase_instances + copy_index.
  // Solution copy numbers can change between assemblages; compare compositions
  // to track a particular composition across a field change.
  int id = -1, candidate_index = -1;
  std::string name;
  double amount = 0.; // moles, on the bulk's supplied amount scale
  // Proportions in the candidate's original endmembers, summing to one.
  // Pure phases have [1]; ordering coefficients may be negative.
  Eigen::VectorXd composition;
};

struct State {
  double pressure = 0., temperature = 0.; // Pa, K
  double composition_coordinate = 0.;     // X on the supplied bulk path
  double entropy = 0., volume = 0.;       // total J/K, m^3
  bool success = false;
  // A solve explicitly requested at the interior of a constructed face.
  bool is_field_verification = false;
  bool outside_model_domain = false;
  std::string message;
  std::vector<std::string>
      excluded_phases; // EOS-domain diagnostics at this P,T
  std::vector<PhaseState> phases;
  double gibbs = 0.; // total J, on the bulk's supplied amount scale
  // Relative mass-balance error, most negative candidate affinity (or zero),
  // and largest absolute reaction affinity (J/mol). Candidate affinity is the
  // phase Gibbs energy minus that predicted from elemental chemical potentials.
  double mass_balance_error = 0., minimum_affinity = 0., equilibrium_error = 0.;
};

struct BoundaryPoint {
  double pressure = 0., temperature = 0.;
  double composition_coordinate = 0.;
  double entropy = 0., volume = 0.;
  // Boundary states retain the zero-amount phase for reconstructing the solve.
  std::vector<PhaseState> phases;
  double mass_balance_error = 0., minimum_affinity = 0., residual = 0.;
};

struct Boundary {
  int id = -1, zero_phase = -1, start_node = -1, end_node = -1;
  // Verified exchange of distinct solution compositions at unchanged counts.
  bool is_solution_replacement = false;
  // assemblage retains all boundary instances, including zero_phase.
  // In selected-axis order, side_a is right and side_b left along the line.
  std::vector<int> assemblage, side_a, side_b;
  std::vector<BoundaryPoint> points;
  // "start status; end status". Failed/unresolved ends have node ID -1.
  std::string termination;
};

struct Node {
  // pt_nullity refers to the selected diagram axes, including S/V/X sections.
  int id = -1, gibbs_variance = 0, pt_nullity = 0;
  std::string kind;
  double pressure = 0., temperature = 0.;
  double composition_coordinate = 0.;
  double entropy = 0., volume = 0.;
  std::vector<int> zero_phases, assemblage, incident_lines;
  Eigen::VectorXd critical_mode; // normalised endmember direction, if verified
};

struct Field {
  int id = -1;
  // Sorted IDs of phases present in the samples. The same assemblage may
  // occupy several disconnected regions of the diagram.
  std::vector<int> phases, sample_indices;
};

struct Result {
  std::array<double, 2> pressure_range, temperature_range;
  CompositionSection section;
  types::FormulaMap composition_start;
  std::array<std::array<double, 2>, 2> coordinate_ranges() const {
    auto range = [&](Coordinate axis) {
      switch (axis) {
      case Coordinate::P:
        return pressure_range;
      case Coordinate::T:
        return temperature_range;
      case Coordinate::S:
        return section.entropy_range;
      case Coordinate::V:
        return section.volume_range;
      case Coordinate::X:
        return section.composition_range;
      }
      throw std::invalid_argument("Unknown diagram coordinate.");
    };
    auto axes = diagram_axes(section.type);
    return {range(axes[0]), range(axes[1])};
  }
  // Actual settings, retained so continuation uses the same composition
  // coordinates, numerical tolerances and EOS-domain policy by default.
  Settings settings;
  std::vector<std::string> phase_names;
  std::vector<State> samples;
  std::vector<Field> fields;
  std::vector<Boundary> boundaries;
  std::vector<Node> nodes;
  std::vector<std::string> diagnostics;
  // Closed diagram-coordinate rings outside the admissible EOS domain: either
  // remaining candidates cannot represent the bulk or a required EOS fails.
  // These are model limits, not equilibrium phase lines.
  std::vector<Eigen::MatrixXd> excluded_regions;
  // True when no problems remain after checking boundaries and field interiors.
  // Even if true, the fields may not represent the global minimum Gibbs energy.
  bool resolved = false;
  // Cumulative work, including the previous result when refined.
  int equilibrium_solves = 0, minimization_calls = 0;
};

struct FieldPolygon {
  // field_id refers to Field::id; sample_index indexes Result::samples.
  // Either can be -1 when identification is unavailable.
  int field_id = -1, n_phases = 0, sample_index = -1;
  bool has_open_boundary = false;
  bool outside_model_domain = false;
  std::vector<int> phases;
  // Zero-based faces in the original, unmerged planar subdivision.
  std::vector<int> source_regions;
  // Closed rings in the selected diagram coordinates (SI units). Holes
  // have opposite winding. Area is a fraction of the diagram domain.
  Eigen::MatrixXd vertices;
  std::vector<Eigen::MatrixXd> holes;
  double area = 0.;
  // Interior label point with approximate maximum clearance, in diagram units.
  Eigen::Vector2d label_position = Eigen::Vector2d::Zero();
  double label_clearance = 0.; // normalised distance to the closest ring
  /// True only when the entire axis-aligned rectangle lies strictly inside
  /// the exterior and outside all holes. Centre and half-size use diagram
  /// units.
  bool contains_rectangle(const Eigen::Vector2d &centre,
                          const Eigen::Vector2d &half_size) const;
};

struct FieldPolygons {
  std::vector<FieldPolygon> polygons;
  std::vector<std::string> diagnostics;
  // Visible, intersection-split edges in diagram units, excluding dissolved
  // internal edges. Unfinished and unclassified edges remain visible.
  std::vector<Eigen::MatrixXd> boundary_segments;
  // Node IDs where at least three visible segments meet.
  std::vector<int> boundary_nodes;
};

/// Construct field polygons from the traced lines, retaining holes and
/// disconnected regions. Optionally use the calculation frame to close edge
/// fields. No equilibrium solves or raster interpolation are performed.
/// tolerance measures distance after each diagram axis is rescaled to 0..1.
/// Unknown counts have n_phases=0; ambiguous assemblages have empty phases.
/// Both cases have a diagnostic unless outside_model_domain marks an exclusion.
/// Adjacent identified faces with identical phase IDs are merged by default;
/// disconnected regions and different numbers of solution copies stay separate.
/// source_regions retains original indices, also used by diagnostics.
FieldPolygons field_polygons(const Result &result, double tolerance = 1.e-8,
                             bool close_domain = true,
                             bool merge_fields = true);

/// Calculate phase fields for a closed elemental bulk. Find stable starting
/// assemblages by minimising Gibbs energy, then trace boundaries where a phase
/// amount reaches zero. At junctions, try branches with one phase added,
/// removed or exchanged. Accept branches only where equilibrium defines a line
/// in the selected diagram and no lower-Gibbs-energy alternative is found.
/// Finite seed sampling does not guarantee discovery of every disconnected
/// field. Inspect resolved and diagnostics; increase seed density to assess it.
/// section selects any pair of P,T,S,V,X and supplies S/V/X ranges. X diagrams
/// fix one other thermodynamic coordinate. S and V are extensive on the bulk
/// amount scale. Missing P/T are solved using the supplied P/T ranges as search
/// bounds. PX/TX can specify fixed T/P by giving that range equal endpoints.
Result pseudosection(const types::FormulaMap &composition,
                     const std::vector<std::shared_ptr<Material>> &candidates,
                     const std::array<double, 2> &pressure_range,
                     const std::array<double, 2> &temperature_range,
                     const Settings &settings = Settings{},
                     const CompositionSection &section = CompositionSection{});

struct LineResolution {
  // Axis point counts in selected diagram-axis order: 101 means 100 divisions.
  std::array<int, 2> axis_points;
  // Divide the density axis uniformly instead of the volume axis.
  bool reciprocal_volume = false;
};

/// Resume unfinished boundaries from their last verified compositions and
/// revisit junction branches. Supply the same bulk, candidates and phase-ID
/// settings as the original calculation, including candidate order. Recovery
/// may refine straight segments, trim tails or merge duplicate lines.
/// resolution also subdivides every boundary using verified equilibrium solves,
/// limiting point spacing to the axis span divided by (axis_points - 1).
/// Spacing that cannot be reached is reported in the returned diagnostics.
/// Added points per line are limited by
/// max_trace_steps*max_refinement_iterations.
Result refine_pseudosection(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Result &previous, const Settings &settings,
    const std::optional<LineResolution> &resolution = std::nullopt);

/// Resume with the settings retained in the previous result.
Result refine_pseudosection(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Result &previous,
    const std::optional<LineResolution> &resolution = std::nullopt);

/// Same native stability search at a single P,T, useful for checking diagrams.
State stable_equilibrium(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates, double pressure,
    double temperature, const Settings &settings = Settings{});

struct ContourSettings {
  int seed_grid = 5, max_trace_steps = 1000;
  // Distances are fractions of the complete diagram ranges, as in Settings.
  double step = .02, min_step = 1.e-6;
};

struct ContourLine {
  int field_id = -1;
  std::vector<int> phases;
  std::vector<BoundaryPoint> points;
  bool closed = false;
  std::string termination;
};

struct ContourResult {
  DiagramType diagram_type = DiagramType::PT;
  std::vector<ContourLine> lines;
  std::vector<std::string> diagnostics;
  int equilibrium_solves = 0;
  bool resolved = true;
};

/// Construct an equality constraint for the field's actual assemblage and
/// parameter layout. Return nullptr where it is undefined
/// (e.g. a garnet composition in a field without garnet). Called once per
/// field; the constraint and its native derivatives drive continuation.
using ContourConstraintFactory =
    std::function<std::unique_ptr<equilibration::EqualityConstraint>(
        const Assemblage &, const equilibration::EquilibrationParameters &,
        const std::vector<int> &)>;

/// Trace a constraint through saved, identified closed fields, using their
/// accepted phase amounts/compositions as warm starts. No phase boundaries
/// are recalculated and the source is not modified. Candidates and phase-ID
/// spacing must match the original calculation. Finite seed sampling can miss
/// disconnected contours; increase seed_grid to assess their discovery.
/// Use the saved field's phases and check the equilibrium constraints;
/// discovering phases and checking alternative assemblages is not repeated.
ContourResult
pseudosection_contours(const Result &previous,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const ContourConstraintFactory &constraint,
                       const ContourSettings &settings = ContourSettings{});

} // namespace burnman::pseudosections
