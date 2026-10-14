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
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace burnman::phase_diagrams {

// Native records use diagram_axes() order, regardless of plot orientation.
// P/T are Pa/K; S/V are total J/K and m^3 for the supplied elemental mole
// amounts. X is the mixing fraction between two bulk compositions, not a
// mineral's endmember proportion. Endpoint amounts are interpolated without
// normalisation.
enum class Coordinate { P, T, S, V, X };
enum class DiagramType { PT, PX, TX, PS, PV, TS, TV, SV, SX, VX };

/// Selected coordinates in calculation order, e.g. {P,T} for a P-T section.
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
/// Whether one diagram axis varies bulk composition along an X mixing path.
inline bool has_composition_axis(DiagramType type) {
  return diagram_axes(type)[1] == Coordinate::X;
}

/// Choice of diagram axes, extensive-coordinate ranges and bulk mixing path.
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

/// Read a physical coordinate from an equilibrium, boundary point or junction.
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

/// Position in the selected diagram, in native units. Stored P,T still give
/// the physical pressure and temperature even for an S/V/X section.
template <typename Point>
Eigen::Vector2d diagram_coordinates(const Point &point, DiagramType type) {
  const auto axes = diagram_axes(type);
  return {coordinate_value(point, axes[0]), coordinate_value(point, axes[1])};
}

/// Sampling, continuation and acceptance criteria for one calculation.
struct Settings {
  // Initial equilibrium grid along the selected axes; not an output pixel grid.
  int pressure_seeds = 7, temperature_seeds = 7;
  int composition_seeds = 7;
  int entropy_seeds = 7, volume_seeds = 7;
  int max_refinement_iterations = 150, minimization_starts = 10;
  // Allow distinct compositions of the same solution model, e.g. two feldspars.
  int max_phase_instances = 3, max_trace_steps = 500, max_lines = 1000;
  int max_recovery_passes = 2;
  // Steps and node distances are fractions of the supplied diagram domain.
  double step = .025, min_step = 1.e-7;
  // Candidate affinities are J/mol. A negative affinity favours forming that
  // phase; mass balance uses relative elemental errors on the bulk scale.
  double affinity_tolerance = .2, mass_balance_tolerance = 1.e-8;
  // amount_tolerance is a fraction of total phase moles; composition_tolerance
  // compares vectors of endmember proportions. Neither is a geometric width.
  double amount_tolerance = 1.e-7, composition_tolerance = 1.e-5;
  // Geometric junction-search distance in the rescaled diagram.
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

/// Amount and composition of one phase, including an individual solvus branch.
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

/// One sampled equilibrium, or a failed attempt with its diagnostic message.
struct State {
  double pressure = 0., temperature = 0.; // Pa, K
  double composition_coordinate = 0.;     // X on the supplied bulk path
  double entropy = 0., volume = 0.;       // total J/K, m^3
  bool success = false;
  // A solve explicitly requested inside a constructed phase-field polygon.
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

/// An accepted equilibrium on a reaction boundary, used to resume tracing.
struct BoundaryPoint {
  double pressure = 0., temperature = 0.;
  double composition_coordinate = 0.;
  double entropy = 0., volume = 0.;
  // Retain the solved zero-amount phase and its equilibrium composition. Its
  // presence in this record does not make it a present phase on both sides.
  std::vector<PhaseState> phases;
  double mass_balance_error = 0., minimum_affinity = 0., residual = 0.;
};

/// One traced reaction curve, stored as an ordered sequence of solved points.
/// Consecutive points are joined by straight segments only for
/// geometry/drawing.
struct Boundary {
  int id = -1, zero_phase = -1, start_node = -1, end_node = -1;
  // Verified exchange of distinct solution compositions at unchanged counts.
  bool is_solution_replacement = false;
  // assemblage retains all boundary instances, including zero_phase. The side
  // labels identify the adjacent finite-area assemblages. In calculation-axis
  // order, side_a is right and side_b left when following points front to back.
  std::vector<int> assemblage, side_a, side_b;
  std::vector<BoundaryPoint> points;
  // "start status; end status". Failed/unresolved ends have node ID -1.
  std::string termination;
};

/// A junction, critical endpoint, or endpoint on the frame or EOS-domain limit.
struct Node {
  // Gibbs variance describes the assemblage's thermodynamic freedom;
  // pt_nullity counts local directions in the selected diagram, including
  // S/V/X.
  int id = -1, gibbs_variance = 0, pt_nullity = 0;
  std::string kind;
  double pressure = 0., temperature = 0.;
  double composition_coordinate = 0.;
  double entropy = 0., volume = 0.;
  std::vector<int> zero_phases, assemblage, incident_lines;
  Eigen::VectorXd critical_mode; // normalised endmember direction, if verified
  // Stable critical coalescence and both limiting states of a P-T univariant
  // replacement have been re-solved; this endpoint needs no second graph arm.
  bool is_verified_critical_endpoint = false;
};

/// A sampled assemblage, which can occur in several disconnected phase fields.
struct Field {
  // Internal sampled-assemblage ID, distinct from a closed-loop field_number.
  int id = -1;
  // Sorted IDs of phases present in the samples. The same assemblage may
  // occupy several disconnected regions of the diagram.
  std::vector<int> phases, sample_indices;
};

/// Saved thermodynamic states and boundary connectivity for the whole diagram.
/// fields groups samples by assemblage; field_polygons() finds the actual
/// areas.
struct Result {
  std::array<double, 2> pressure_range, temperature_range;
  CompositionSection section;
  types::FormulaMap composition_start;
  /// Bounds in calculation-axis order and native units, independent of
  /// plotting.
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
  // True when the implemented tracing and field-identification checks leave
  // no diagnostics. This does not prove that finite seed sampling found every
  // stability field or every lower-energy composition of the candidate models.
  bool resolved = false;
  // Cumulative work, including the previous result when refined.
  int equilibrium_solves = 0, minimization_calls = 0;
};

/// One drawable closed field: an outer coordinate ring and any interior holes.
/// A ring repeats its first point at the end; holes exclude enclosed areas.
struct FieldPolygon {
  // One-based closed-loop number, shared by plots, keys and diagnostics.
  int field_number = 0;
  // field_id refers to Field::id; sample_index indexes Result::samples.
  // These are internal sampled-assemblage/state IDs, not display numbers.
  // Either can be -1 when identification is unavailable.
  int field_id = -1, n_phases = 0, sample_index = -1;
  // Source reaction arcs forming the exterior and holes; useful for targeted
  // side-label recovery without equilibrating arbitrary polygon interiors.
  std::vector<int> boundary_ids;
  // A closed drawing walk can still contain an unfinished interior reaction
  // line. Such a field is open thermodynamically and cannot be trusted as
  // closed.
  bool has_open_boundary = false;
  // A display label chosen despite missing or conflicting perimeter evidence.
  // This does not assert a resolved topology or a newly verified equilibrium.
  bool is_provisional = false;
  bool outside_model_domain = false;
  std::vector<int> phases;
  // Zero-based closed-loop indices; use field_number for display.
  std::vector<int> source_regions;
  // Closed rings in the selected diagram coordinates (SI units). Holes
  // have opposite winding. Area is a fraction of the diagram domain.
  Eigen::MatrixXd vertices;
  std::vector<Eigen::MatrixXd> holes;
  double area = 0.;
  // Interior label point with approximate maximum clearance, in diagram units.
  Eigen::Vector2d label_position = Eigen::Vector2d::Zero();
  double label_clearance = 0.; // normalised distance to the closest ring
  /// Test whether a label rectangle fits wholly inside this field, including
  /// clearance from holes and concave boundaries. centre and half_size use
  /// native diagram units; contact with a boundary returns false.
  bool contains_rectangle(const Eigen::Vector2d &centre,
                          const Eigen::Vector2d &half_size) const;
};

/// Plotting geometry, inferred neighbouring assemblages and labelling messages.
struct FieldPolygons {
  std::vector<FieldPolygon> polygons;
  std::vector<std::string> diagnostics;
  // Unique closed, non-provisional neighbours compatible with the reaction.
  // Boundary ID -> {right, left}, following that record's native direction.
  std::map<int, std::array<std::vector<int>, 2>> boundary_sides;
  // Saved curve segments in diagram units. Incidental chord crossings do
  // not split edges or create junctions. Unfinished edges remain visible.
  std::vector<Eigen::MatrixXd> boundary_segments;
  // Connected segments with the same two assemblages, stopping at endpoints
  // and junctions. Closed loops repeat their first point.
  std::vector<Eigen::MatrixXd> boundary_lines;
  // Node IDs where at least three visible segments meet.
  std::vector<int> boundary_nodes;
};

} // namespace burnman::phase_diagrams
