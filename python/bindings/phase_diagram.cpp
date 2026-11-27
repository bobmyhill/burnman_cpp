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

#include "burnman/tools/phase_diagram.hpp"
#include "bindings.hpp"
#include "burnman/minerals/water.hpp"
#include "burnman/tools/phase_diagram/refinement.hpp"
#include <set>
namespace burnman::python {
namespace {
std::string reaction_name(phase_diagrams::ReactionKind kind) {
  using phase_diagrams::ReactionKind;
  return kind == ReactionKind::phase_out     ? "phase_out"
         : kind == ReactionKind::replacement ? "replacement"
                                             : "unknown";
}
std::string coordinate_name(phase_diagrams::Coordinate axis) {
  return std::string(1, std::string("PTSVX")[static_cast<std::size_t>(axis)]);
}
phase_diagrams::Coordinate coordinate_type(const std::string &name) {
  auto i = std::string("PTSVX").find(name);
  if (name.size() != 1 || i == std::string::npos)
    throw py::value_error("Unknown coordinate: " + name);
  return static_cast<phase_diagrams::Coordinate>(i);
}
std::string diagram_name(phase_diagrams::DiagramType type) {
  auto axes = phase_diagrams::diagram_axes(type);
  return coordinate_name(axes[0]) + coordinate_name(axes[1]);
}
phase_diagrams::DiagramType diagram_type(std::string name) {
  name.erase(std::remove(name.begin(), name.end(), '-'), name.end());
  for (int i = 0; i < 10; ++i) {
    auto type = static_cast<phase_diagrams::DiagramType>(i);
    if (name == diagram_name(type))
      return type;
  }
  throw py::value_error(
      "diagram must be one of PT, PS, PV, TS, TV, SV, PX, TX, SX, VX.");
}
#define SETTINGS_OPTIONS(X)                                                    \
  X(pressure_seeds)                                                            \
  X(temperature_seeds)                                                         \
  X(entropy_seeds)                                                             \
  X(volume_seeds)                                                              \
  X(composition_seeds)                                                         \
  X(max_refinement_iterations)                                                 \
  X(minimization_starts)                                                       \
  X(max_phase_instances)                                                       \
  X(max_trace_steps)                                                           \
  X(max_lines)                                                                 \
  X(max_recovery_passes)                                                       \
  X(step)                                                                      \
  X(min_step)                                                                  \
  X(affinity_tolerance)                                                        \
  X(mass_balance_tolerance)                                                    \
  X(amount_tolerance)                                                          \
  X(composition_tolerance)                                                     \
  X(node_tolerance)                                                            \
  X(verbose)                                                                   \
  X(exclude_invalid_eos)                                                       \
  X(active_solution_faces)                                                     \
  X(required_eos_phases)                                                       \
  X(required_thermal_phases)

py::dict settings_dict(const phase_diagrams::Settings &settings) {
  py::dict data;
#define SAVE(name) data[#name] = py::cast(settings.name);
  SETTINGS_OPTIONS(SAVE)
#undef SAVE
  return data;
}
phase_diagrams::Settings saved_settings(const py::dict &data,
                                        bool for_calculation = false) {
  phase_diagrams::Settings settings;
  std::set<std::string> names;
#define READ(name)                                                             \
  names.insert(#name);                                                         \
  if (for_calculation || data.contains(#name))                                 \
    settings.name = data[#name].cast<decltype(settings.name)>();
  SETTINGS_OPTIONS(READ)
#undef READ
  for (auto item : data)
    if (!names.count(py::cast<std::string>(item.first)))
      throw py::value_error("Unknown phase diagram setting: " +
                            py::cast<std::string>(item.first));
  return settings;
}
py::list vector_list(const Eigen::VectorXd &vector) {
  py::list values;
  for (double value : vector)
    values.append(value);
  return values;
}
py::list matrix_list(const Eigen::MatrixXd &matrix) {
  py::list rows;
  for (int i = 0; i < matrix.rows(); ++i)
    rows.append(vector_list(matrix.row(i).transpose()));
  return rows;
}
// Geometry views accept partial records; calculations require the saved
// equilibrium members. Keep the presence/type handling shared by both views.
template <typename Value>
void read_member(const py::dict &record, const char *key, Value &value,
                 bool required = false) {
  if (required || record.contains(key))
    value = record[key].cast<Value>();
}
std::vector<phase_diagrams::PhaseState>
saved_phase_states(py::handle records, bool for_calculation = true) {
  std::vector<phase_diagrams::PhaseState> states;
  for (auto item : py::reinterpret_borrow<py::iterable>(records)) {
    auto record = py::reinterpret_borrow<py::dict>(item);
    phase_diagrams::PhaseState state;
    state.amount = 1.;
#define READ(name) read_member(record, #name, state.name, for_calculation);
    state.id = record["id"].cast<int>();
    READ(amount)
    if (for_calculation) {
      if (!record.contains("composition") ||
          !record.contains("candidate_index"))
        throw py::value_error("Resuming requires saved phase compositions "
                              "and candidate_index.");
      READ(candidate_index)
      READ(composition)
      READ(name)
      read_member(record, "composition_basis", state.composition_basis);
    }
#undef READ
    states.push_back(std::move(state));
  }
  return states;
}
template <typename Point>
void read_coordinates(const py::dict &record, Point &point,
                      phase_diagrams::DiagramType type,
                      bool for_calculation = false) {
  using namespace phase_diagrams;
  point.pressure = record["pressure"].cast<double>();
  point.temperature = record["temperature"].cast<double>();
  const auto axes = diagram_axes(type);
  auto read = [&](const char *key, Coordinate axis, double &value) {
    if (!record.contains(key) &&
        (for_calculation || axes[0] == axis || axes[1] == axis))
      throw py::value_error(std::string("Records in this diagram require ") +
                            key + ".");
    read_member(record, key, value);
  };
  read("entropy", Coordinate::S, point.entropy);
  read("volume", Coordinate::V, point.volume);
  read("composition_coordinate", Coordinate::X, point.composition_coordinate);
}
phase_diagrams::BoundaryPoint
saved_boundary_point(const py::dict &record, phase_diagrams::DiagramType type,
                     bool for_calculation = true) {
  phase_diagrams::BoundaryPoint point;
  read_coordinates(record, point, type, for_calculation);
  if (for_calculation) {
    point.phases = saved_phase_states(record["phases"]);
#define READ(name) point.name = record[#name].cast<decltype(point.name)>();
    READ(mass_balance_error)
    READ(minimum_affinity)
    READ(residual)
#undef READ
  }
  return point;
}
py::list
phase_states_dict(const std::vector<phase_diagrams::PhaseState> &states) {
  py::list records;
  for (auto &state : states) {
    py::dict record;
#define SAVE(name) record[#name] = py::cast(state.name);
    SAVE(id)
    SAVE(candidate_index)
    SAVE(name)
    SAVE(amount)
#undef SAVE
    record["composition"] = vector_list(state.composition);
    if (state.composition_basis)
      record["composition_basis"] = matrix_list(*state.composition_basis);
    records.append(record);
  }
  return records;
}
py::dict equilibrium_state_dict(const phase_diagrams::State &state) {
  py::dict record;
#define SAVE(name) record[#name] = py::cast(state.name);
  SAVE(pressure)
  SAVE(temperature)
  SAVE(entropy)
  SAVE(volume)
  SAVE(composition_coordinate)
  SAVE(success)
  SAVE(is_field_verification)
  SAVE(outside_model_domain)
  SAVE(message)
  SAVE(excluded_phases)
  SAVE(gibbs)
  SAVE(mass_balance_error)
  SAVE(minimum_affinity)
  SAVE(equilibrium_error)
#undef SAVE
  record["phases"] = phase_states_dict(state.phases);
  return record;
}
py::dict boundary_point_dict(const phase_diagrams::BoundaryPoint &point) {
  py::dict item;
#define SAVE(name) item[#name] = py::cast(point.name);
  SAVE(pressure)
  SAVE(temperature)
  SAVE(entropy)
  SAVE(volume)
  SAVE(composition_coordinate)
  SAVE(mass_balance_error)
  SAVE(minimum_affinity)
  SAVE(residual)
#undef SAVE
  item["phases"] = phase_states_dict(point.phases);
  return item;
}
py::dict
validation_dict(const phase_diagrams::ThermodynamicValidation &validation) {
  py::dict data;
#define SAVE(name) data[#name] = py::cast(validation.name);
  SAVE(scope)
  SAVE(checked_pairs)
  SAVE(skipped_pairs)
  SAVE(violations)
#undef SAVE
  return data;
}
phase_diagrams::ThermodynamicValidation
saved_validation(const py::dict &data, bool for_calculation = false) {
  phase_diagrams::ThermodynamicValidation validation;
  if (for_calculation || data.contains("thermodynamic_validation")) {
    auto record = data["thermodynamic_validation"].cast<py::dict>();
#define READ(name)                                                             \
  validation.name = record[#name].cast<decltype(validation.name)>();
    READ(scope)
    READ(checked_pairs)
    READ(skipped_pairs)
    READ(violations)
#undef READ
  }
  return validation;
}
constexpr std::array<const char *, 14> diagnostic_categories{
    "other",
    "trace",
    "boundary_note",
    "sample",
    "field",
    "boundary",
    "junction",
    "critical_endpoint",
    "missing_assemblage",
    "thermodynamic",
    "equilibrium",
    "property",
    "spacing",
    "search"};
static_assert(
    static_cast<std::size_t>(phase_diagrams::DiagnosticCategory::count) ==
    diagnostic_categories.size());

py::list diagnostic_records(const phase_diagrams::Diagnostics &diagnostics) {
  py::list records;
  for (const auto &diagnostic : diagnostics) {
    py::dict record;
    record["category"] =
        diagnostic_categories.at(static_cast<std::size_t>(diagnostic.category));
    record["message"] = diagnostic.message;
    if (diagnostic.entity_id >= 0)
      record["entity_id"] = diagnostic.entity_id;
    if (diagnostic.grid_point[0] >= 0)
      record["grid_point"] = diagnostic.grid_point;
    records.append(record);
  }
  return records;
}
phase_diagrams::Diagnostics saved_diagnostics(const py::handle &items) {
  phase_diagrams::Diagnostics diagnostics;
  for (auto item : items.cast<py::list>()) {
    if (!py::isinstance<py::dict>(item))
      throw py::value_error("Diagnostics require category/message records.");
    const auto record = item.cast<py::dict>();
    const auto category = record["category"].cast<std::string>();
    const auto found = std::find(diagnostic_categories.begin(),
                                 diagnostic_categories.end(), category);
    if (found == diagnostic_categories.end())
      throw py::value_error("Unknown diagnostic category: " + category);
    phase_diagrams::Diagnostic diagnostic{
        static_cast<phase_diagrams::DiagnosticCategory>(
            found - diagnostic_categories.begin()),
        record.contains("entity_id") ? record["entity_id"].cast<int>() : -1,
        record["message"].cast<std::string>()};
    if (record.contains("grid_point"))
      diagnostic.grid_point = record["grid_point"].cast<std::array<int, 2>>();
    if (record.contains("entity_id") && diagnostic.entity_id < 0)
      throw py::value_error("Diagnostic entity_id must be nonnegative.");
    if (record.contains("grid_point") &&
        (diagnostic.grid_point[0] < 0 || diagnostic.grid_point[1] < 0))
      throw py::value_error(
          "Diagnostic grid_point must contain nonnegative indices.");
    diagnostics.push_back(std::move(diagnostic));
  }
  return diagnostics;
}
std::vector<std::string>
diagnostic_messages(const phase_diagrams::Diagnostics &diagnostics) {
  std::vector<std::string> messages;
  for (const auto &diagnostic : diagnostics)
    messages.push_back(diagnostic.text());
  return messages;
}
template <typename T, typename Format>
auto diagnostic_view(phase_diagrams::Diagnostics T::*member, Format format) {
  return [member, format](const T &value) { return format(value.*member); };
}

py::dict result_dict(const phase_diagrams::Result &result) {
  py::dict data;
  data["schema_version"] = 6;
  data["diagram_type"] = diagram_name(result.section.type);
  data["composition_start"] = py::cast(result.composition_start);
  data["composition_end"] = py::cast(result.section.composition_end);
  data["composition_range"] = py::cast(result.section.composition_range);
  data["entropy_range"] = py::cast(result.section.entropy_range);
  data["volume_range"] = py::cast(result.section.volume_range);
  if (result.section.fixed_coordinate) {
    data["fixed_coordinate"] =
        coordinate_name(*result.section.fixed_coordinate);
    data["fixed_value"] = result.section.fixed_value;
  }
#define SAVE(name) data[#name] = py::cast(result.name);
  SAVE(pressure_range)
  SAVE(temperature_range)
  SAVE(phase_names)
  data["diagnostics"] = diagnostic_records(result.diagnostics);
  data["coordinate_diagnostics"] =
      diagnostic_records(result.coordinate_diagnostics);
  data["spacing_diagnostics"] = diagnostic_records(result.spacing_diagnostics);
  data["spacing_resolved"] = result.spacing_diagnostics.empty();
  SAVE(resolved)
  SAVE(equilibrium_solves)
  SAVE(minimization_calls)
#undef SAVE
  data["thermodynamic_validation"] =
      validation_dict(result.thermodynamic_validation);
  data["settings"] = settings_dict(result.settings);
  py::list verifications;
  for (const auto &verification : result.field_verifications) {
    py::dict record;
#define SAVE(name) record[#name] = py::cast(verification.name);
    SAVE(phases)
    SAVE(boundary_ids)
    SAVE(node_ids)
    SAVE(success)
    SAVE(graph_signature)
    SAVE(message)
    SAVE(checked_junctions)
    SAVE(checked_boundary_points)
    SAVE(checked_crossings)
    SAVE(maximum_mass_balance_error)
    SAVE(maximum_reaction_error)
    SAVE(minimum_affinity)
#undef SAVE
    verifications.append(record);
  }
  data["field_verifications"] = verifications;
  py::list excluded;
  for (auto &ring : result.excluded_regions)
    excluded.append(matrix_list(ring));
  data["excluded_regions"] = excluded;
  py::list fields, nodes, boundaries, samples;
  for (auto &field : result.fields) {
    py::dict record;
#define SAVE(name) record[#name] = py::cast(field.name);
    SAVE(id)
    SAVE(phases)
    SAVE(sample_indices)
#undef SAVE
    fields.append(record);
  }
  for (auto &node : result.nodes) {
    py::dict record;
#define SAVE(name) record[#name] = py::cast(node.name);
    SAVE(id)
    SAVE(pressure)
    SAVE(temperature)
    SAVE(entropy)
    SAVE(volume)
    SAVE(composition_coordinate)
    SAVE(kind)
    SAVE(zero_phases)
    SAVE(assemblage)
    SAVE(gibbs_variance)
    SAVE(pt_nullity)
    SAVE(incident_lines)
    SAVE(is_verified_critical_endpoint)
#undef SAVE
    record["critical_mode"] = vector_list(node.critical_mode);
    nodes.append(record);
  }
  for (auto &line : result.boundaries) {
    py::dict record;
#define SAVE(name) record[#name] = py::cast(line.name);
    SAVE(id)
    SAVE(zero_phase)
    SAVE(assemblage)
    SAVE(side_a)
    SAVE(side_b)
    SAVE(is_solution_replacement)
    SAVE(start_node)
    SAVE(end_node)
    SAVE(termination)
#undef SAVE
    record["reaction_kind"] = reaction_name(line.reaction_kind);
    py::list points;
    for (auto &point : line.points) {
      points.append(boundary_point_dict(point));
    }
    record["points"] = points;
    py::list curves;
    for (const auto &curve : line.limiting_curves) {
      py::dict saved;
      saved["assemblage"] = py::cast(curve.assemblage);
      py::list curve_points;
      for (const auto &point : curve.points)
        curve_points.append(equilibrium_state_dict(point));
      saved["points"] = curve_points;
      curves.append(saved);
    }
    record["limiting_curves"] = curves;
    boundaries.append(record);
  }
  for (auto &state : result.samples)
    samples.append(equilibrium_state_dict(state));
  data["fields"] = fields;
  data["nodes"] = nodes;
  data["boundaries"] = boundaries;
  data["samples"] = samples;
  return data;
}
// Geometry-only JSON remains sufficient for plotting. Resuming also requires
// the accepted phase compositions/amounts retained by Result::to_dict().
phase_diagrams::Result saved_result(const py::dict &data,
                                    bool for_calculation = false) {
  using namespace phase_diagrams;
  if ((for_calculation || data.contains("schema_version")) &&
      (!data.contains("schema_version") ||
       data["schema_version"].cast<int>() != 6))
    throw py::value_error("Phase diagrams require schema version 6.");
  Result result;
  result.thermodynamic_validation = saved_validation(data, for_calculation);
  if (for_calculation || data.contains("field_verifications"))
    for (auto item : data["field_verifications"].cast<py::list>()) {
      const auto record = item.cast<py::dict>();
      FieldVerification verification;
#define READ(name)                                                             \
  verification.name = record[#name].cast<decltype(verification.name)>();
      READ(phases)
      READ(boundary_ids)
      READ(node_ids)
      READ(success)
      READ(graph_signature)
      READ(message)
      READ(checked_junctions)
      READ(checked_boundary_points)
      READ(checked_crossings)
      READ(maximum_mass_balance_error)
      READ(maximum_reaction_error)
      READ(minimum_affinity)
#undef READ
      result.field_verifications.push_back(std::move(verification));
    }
  if (for_calculation || data.contains("diagram_type"))
    result.section.type =
        diagram_type(data["diagram_type"].cast<std::string>());
#define READ(name)                                                             \
  read_member(data, #name, result.section.name, for_calculation);
  READ(composition_end)
  READ(composition_range)
  READ(entropy_range)
  READ(volume_range)
#undef READ
  read_member(data, "composition_start", result.composition_start,
              for_calculation);
  if (data.contains("fixed_coordinate")) {
    result.section.fixed_coordinate =
        coordinate_type(data["fixed_coordinate"].cast<std::string>());
    result.section.fixed_value = data["fixed_value"].cast<double>();
  }
  if (for_calculation || data.contains("settings"))
    result.settings =
        saved_settings(data["settings"].cast<py::dict>(), for_calculation);
  read_member(data, "resolved", result.resolved, for_calculation);
  if (for_calculation || data.contains("diagnostics"))
    result.diagnostics = saved_diagnostics(data["diagnostics"]);
  if (for_calculation || data.contains("coordinate_diagnostics"))
    result.coordinate_diagnostics =
        saved_diagnostics(data["coordinate_diagnostics"]);
  if (for_calculation || data.contains("spacing_diagnostics"))
    result.spacing_diagnostics = saved_diagnostics(data["spacing_diagnostics"]);
  result.pressure_range = data["pressure_range"].cast<std::array<double, 2>>();
  result.temperature_range =
      data["temperature_range"].cast<std::array<double, 2>>();
  read_member(data, "phase_names", result.phase_names, for_calculation);
  read_member(data, "excluded_regions", result.excluded_regions,
              for_calculation);
  auto saved_state = [&](const py::dict &value) {
    State state;
    state.success = value["success"].cast<bool>();
    read_coordinates(value, state, result.section.type, for_calculation);
#define READ(name) read_member(value, #name, state.name, for_calculation);
    READ(is_field_verification)
    READ(outside_model_domain)
    READ(excluded_phases)
#undef READ
    state.phases = saved_phase_states(value["phases"], for_calculation);
    if (for_calculation) {
#define READ(name) state.name = value[#name].cast<decltype(state.name)>();
      READ(message)
      READ(gibbs)
      READ(mass_balance_error)
      READ(minimum_affinity)
      READ(equilibrium_error)
#undef READ
    }
    return state;
  };
  if (for_calculation || data.contains("nodes"))
    for (auto item : data["nodes"]) {
      auto value = py::reinterpret_borrow<py::dict>(item);
      Node node;
      if (!for_calculation)
        node.gibbs_variance = node.pt_nullity = -1; // unknown unless supplied
      node.id = value["id"].cast<int>();
      read_coordinates(value, node, result.section.type, for_calculation);
      read_member(value, "incident_lines", node.incident_lines,
                  for_calculation);
      read_member(value, "is_verified_critical_endpoint",
                  node.is_verified_critical_endpoint, for_calculation);
    // Polygon construction also needs invariant metadata when plotting a
    // saved dictionary, without requiring all resumable equilibrium data.
#define READ(name) read_member(value, #name, node.name, for_calculation);
      READ(kind)
      READ(assemblage)
      READ(gibbs_variance)
      READ(pt_nullity)
      READ(zero_phases)
#undef READ
      if (for_calculation)
        read_member(value, "critical_mode", node.critical_mode, true);
      result.nodes.push_back(node);
    }
  for (auto item : data["boundaries"]) {
    auto value = py::reinterpret_borrow<py::dict>(item);
    Boundary line;
    // Loop constraints need the reaction metadata for dictionary plotting too.
#define READ(name) read_member(value, #name, line.name, for_calculation);
    READ(assemblage)
    READ(id)
    READ(zero_phase)
    READ(termination)
#undef READ
#define READ(name) read_member(value, #name, line.name, for_calculation);
    READ(start_node)
    READ(end_node)
    READ(side_a)
    READ(side_b)
    READ(is_solution_replacement)
#undef READ
    if (for_calculation || value.contains("reaction_kind")) {
      const auto kind = value["reaction_kind"].cast<std::string>();
      if (kind == "phase_out")
        line.reaction_kind = ReactionKind::phase_out;
      else if (kind == "replacement")
        line.reaction_kind = ReactionKind::replacement;
      else if (kind != "unknown")
        throw py::value_error("Unknown reaction kind: " + kind);
    }
    for (auto point : value["points"]) {
      line.points.push_back(
          saved_boundary_point(py::reinterpret_borrow<py::dict>(point),
                               result.section.type, for_calculation));
    }
    if (for_calculation) {
      auto curves = value["limiting_curves"].cast<py::list>();
      if (curves.size() != 2)
        throw py::value_error("A reaction boundary needs two limiting curves.");
      for (std::size_t side = 0; side < 2; ++side) {
        auto curve = curves[side].cast<py::dict>();
        line.limiting_curves[side].assemblage =
            curve["assemblage"].cast<std::vector<int>>();
        for (auto point : curve["points"])
          line.limiting_curves[side].points.push_back(
              saved_state(py::reinterpret_borrow<py::dict>(point)));
      }
    }
    result.boundaries.push_back(std::move(line));
  }
  if (for_calculation || data.contains("fields"))
    for (auto item : data["fields"]) {
      auto value = py::reinterpret_borrow<py::dict>(item);
      Field field;
      field.id = value["id"].cast<int>();
      field.phases = value["phases"].cast<std::vector<int>>();
      if (for_calculation)
        field.sample_indices = value["sample_indices"].cast<std::vector<int>>();
      result.fields.push_back(field);
    }
  if (for_calculation || data.contains("samples"))
    for (auto item : data["samples"]) {
      result.samples.push_back(
          saved_state(py::reinterpret_borrow<py::dict>(item)));
    }
  if (for_calculation) {
    read_member(data, "equilibrium_solves", result.equilibrium_solves, true);
    read_member(data, "minimization_calls", result.minimization_calls, true);
  }
  return result;
}
phase_diagrams::Result calculation_result(const py::object &object) {
  if (py::isinstance<phase_diagrams::Result>(object))
    return object.cast<phase_diagrams::Result>();
  if (py::isinstance<py::dict>(object))
    return saved_result(object.cast<py::dict>(), true);
  throw py::type_error(
      "Supply a PhaseDiagramResult or its full saved JSON dictionary.");
}

py::dict grid_solution_dict(const phase_diagrams::GridSolution &solution) {
  auto data = boundary_point_dict(solution.state);
  data["n_moles"] = solution.n_moles;
  data["gibbs"] = solution.gibbs;
  data["molar_fractions"] = vector_list(solution.molar_fractions.matrix());
  py::list bases, compositions;
  for (const auto &basis : solution.phase_bases)
    bases.append(matrix_list(basis));
  for (const auto &composition : solution.phase_compositions)
    compositions.append(vector_list(composition));
  data["phase_bases"] = bases;
  data["phase_compositions"] = compositions;
  return data;
}
py::dict property_grid_dict(const phase_diagrams::PropertyGrid &grid) {
  py::dict data;
  data["format"] = "burnman_cpp.property_grid";
  data["schema_version"] = 2;
  data["context"] = result_dict(grid.context);
  data["x"] = vector_list(grid.x);
  data["y"] = vector_list(grid.y);
  data["relaxed"] = grid.relaxed;
  data["resolved"] = grid.resolved;
  data["field_numbers"] = py::cast(grid.field_numbers).attr("tolist")();
  data["success"] = py::cast(grid.success).attr("tolist")();
  data["equilibrium_success"] =
      py::cast(grid.equilibrium_success).attr("tolist")();
  data["diagnostics"] = diagnostic_records(grid.diagnostics);
  data["equilibrium_diagnostics"] =
      diagnostic_records(grid.equilibrium_diagnostics);
  data["equilibrium_solves"] = grid.equilibrium_solves;
  data["thermodynamic_validation"] =
      validation_dict(grid.thermodynamic_validation);
  py::dict values;
  for (const auto &[name, matrix] : grid.values) {
    py::list rows;
    for (Eigen::Index j = 0; j < matrix.rows(); ++j) {
      py::list row;
      for (Eigen::Index i = 0; i < matrix.cols(); ++i)
        if (std::isnan(matrix(j, i)))
          row.append(py::none());
        else if (std::isinf(matrix(j, i)))
          row.append(matrix(j, i) > 0. ? "+inf" : "-inf");
        else
          row.append(matrix(j, i));
      rows.append(row);
    }
    values[py::str(name)] = rows;
  }
  data["values"] = values;
  data["solutions"] = py::none();
  if (!grid.solutions.empty()) {
    py::list rows;
    for (Eigen::Index j = 0; j < grid.y.size(); ++j) {
      py::list row;
      for (Eigen::Index i = 0; i < grid.x.size(); ++i) {
        const auto &solution =
            grid.solutions.at(static_cast<std::size_t>(j * grid.x.size() + i));
        if (solution)
          row.append(grid_solution_dict(*solution));
        else
          row.append(py::none());
      }
      rows.append(row);
    }
    data["solutions"] = rows;
  }
  return data;
}
phase_diagrams::PropertyGrid saved_property_grid(const py::dict &data) {
  using namespace phase_diagrams;
  if (!data.contains("format") ||
      data["format"].cast<std::string>() != "burnman_cpp.property_grid" ||
      !data.contains("schema_version") ||
      data["schema_version"].cast<int>() != 2)
    throw py::value_error(
        "Unsupported property grid format or schema version.");
  PropertyGrid grid;
  grid.thermodynamic_validation = saved_validation(data, true);
  grid.context = saved_result(data["context"].cast<py::dict>(), true);
  grid.diagram_type = grid.context.section.type;
  grid.x = data["x"].cast<Eigen::VectorXd>();
  grid.y = data["y"].cast<Eigen::VectorXd>();
  grid.relaxed = data["relaxed"].cast<bool>();
  grid.resolved = data["resolved"].cast<bool>();
  grid.field_numbers = data["field_numbers"].cast<Eigen::MatrixXi>();
  grid.success = data["success"].cast<decltype(grid.success)>();
  grid.equilibrium_success =
      data["equilibrium_success"].cast<decltype(grid.equilibrium_success)>();
  grid.diagnostics = saved_diagnostics(data["diagnostics"]);
  grid.equilibrium_diagnostics =
      saved_diagnostics(data["equilibrium_diagnostics"]);
  grid.equilibrium_solves = data["equilibrium_solves"].cast<int>();
  grid.values = data["values"].cast<std::map<std::string, Eigen::MatrixXd>>();
  auto shape = [&](const auto &matrix) {
    return matrix.rows() == grid.y.size() && matrix.cols() == grid.x.size();
  };
  if (!grid.x.size() || !grid.y.size() || !grid.x.allFinite() ||
      !grid.y.allFinite() || !shape(grid.field_numbers) ||
      !shape(grid.success) || !shape(grid.equilibrium_success))
    throw py::value_error("Saved property grid arrays do not match its shape.");
  for (const auto &axis : {grid.x, grid.y})
    for (Eigen::Index i = 1; i < axis.size(); ++i)
      if (axis[i] <= axis[i - 1])
        throw py::value_error("Saved grid coordinates must be increasing.");
  for (const auto &[name, matrix] : grid.values)
    if (!shape(matrix))
      throw py::value_error("Saved property array has an invalid shape: " +
                            name);
  if (!data["solutions"].is_none()) {
    const auto rows = data["solutions"].cast<py::list>();
    if (static_cast<Eigen::Index>(rows.size()) != grid.y.size())
      throw py::value_error("Saved solutions do not match the grid shape.");
    for (Eigen::Index j = 0; j < grid.y.size(); ++j) {
      const auto row = rows[static_cast<std::size_t>(j)].cast<py::list>();
      if (static_cast<Eigen::Index>(row.size()) != grid.x.size())
        throw py::value_error("Saved solutions do not match the grid shape.");
      for (Eigen::Index i = 0; i < grid.x.size(); ++i) {
        const auto item = row[static_cast<std::size_t>(i)];
        if (item.is_none()) {
          if (grid.equilibrium_success(j, i))
            throw py::value_error(
                "A successful grid point is missing its solution.");
          grid.solutions.push_back(std::nullopt);
          continue;
        }
        if (!grid.equilibrium_success(j, i) || grid.field_numbers(j, i) < 0)
          throw py::value_error("A saved solution has no successful "
                                "equilibrium or valid field number.");
        auto record = item.cast<py::dict>();
        GridSolution solution;
        solution.max_phase_instances =
            grid.context.settings.max_phase_instances;
        solution.n_moles = record["n_moles"].cast<double>();
        solution.gibbs = record["gibbs"].cast<double>();
        if (!std::isfinite(solution.gibbs))
          throw py::value_error("Saved grid Gibbs energy must be finite.");
        solution.molar_fractions =
            record["molar_fractions"].cast<Eigen::ArrayXd>();
        auto &point = solution.state;
        point = saved_boundary_point(record, grid.diagram_type);
        solution.phase_bases =
            record["phase_bases"].cast<std::vector<Eigen::MatrixXd>>();
        solution.phase_compositions =
            record["phase_compositions"].cast<std::vector<Eigen::VectorXd>>();
        if (point.phases.empty() ||
            point.phases.size() != solution.phase_bases.size() ||
            point.phases.size() != solution.phase_compositions.size())
          throw py::value_error(
              "Saved grid solution has inconsistent phase arrays.");
        grid.solutions.push_back(std::move(solution));
      }
    }
  }
  return grid;
}
} // namespace
void bind_phase_diagram(py::module_ &m) {
  m.def("_phase_diagram_context",
        [](const py::dict &data) { return result_dict(saved_result(data)); });
  m.def("phase_diagram_diagnostic_messages", [](const py::list &records) {
    return diagnostic_messages(saved_diagnostics(records));
  });
  m.def("_refresh_phase_diagram_diagnostics", [](const py::dict &data) {
    auto result = saved_result(data);
    phase_diagrams::detail::refresh_diagram_diagnostics(
        result, phase_diagrams::field_polygons(result));
    py::dict refreshed;
    refreshed["diagnostics"] = diagnostic_records(result.diagnostics);
    refreshed["coordinate_diagnostics"] =
        diagnostic_records(result.coordinate_diagnostics);
    refreshed["resolved"] = result.diagnostics.empty();
    return refreshed;
  });
  using namespace phase_diagrams;
  py::class_<Settings>(m, "PhaseDiagramSettings",
                       "Initial equilibrium sampling, reaction continuation "
                       "and numerical acceptance criteria for a diagram.")
      .def(py::init<>())
      .def("to_dict", &settings_dict,
           "Return all settings as a JSON-compatible dictionary.")
      .def_static(
          "from_dict",
          [](const py::dict &data) { return saved_settings(data); },
          py::arg("data"),
          "Restore numerical settings; omitted entries retain defaults.")
#define OPTION(name) .def_readwrite(#name, &Settings::name)
          SETTINGS_OPTIONS(OPTION)
#undef OPTION
#undef SETTINGS_OPTIONS
      ;
  py::class_<PhaseState>(m, "PhaseState",
                         "One mineral/solution instance: amount in moles and "
                         "proportions in its original endmember basis.")
      .def_readonly("id", &PhaseState::id)
      .def_readonly("candidate_index", &PhaseState::candidate_index)
      .def_readonly("name", &PhaseState::name)
      .def_readonly("amount", &PhaseState::amount)
      .def_property_readonly("composition",
                             [](const PhaseState &s) { return s.composition; })
      .def_property_readonly("composition_basis", [](const PhaseState &s) {
        return s.composition_basis;
      });
  py::class_<State>(m, "EquilibriumState")
#define PROPERTY(name) .def_readonly(#name, &State::name)
      PROPERTY(pressure) PROPERTY(temperature) PROPERTY(entropy)
          PROPERTY(volume) PROPERTY(success) PROPERTY(composition_coordinate)
              PROPERTY(is_field_verification) PROPERTY(message) PROPERTY(phases)
                  PROPERTY(gibbs) PROPERTY(mass_balance_error)
                      PROPERTY(minimum_affinity) PROPERTY(equilibrium_error)
                          PROPERTY(excluded_phases)
                              PROPERTY(outside_model_domain)
#undef PROPERTY
                                  ;
  py::class_<BoundaryPoint>(m, "BoundaryPoint")
#define PROPERTY(name) .def_readonly(#name, &BoundaryPoint::name)
      PROPERTY(pressure) PROPERTY(temperature) PROPERTY(entropy)
          PROPERTY(volume) PROPERTY(phases) PROPERTY(composition_coordinate)
              PROPERTY(mass_balance_error) PROPERTY(minimum_affinity)
                  PROPERTY(residual)
#undef PROPERTY
                      ;
  py::class_<BoundaryCurve>(m, "PhaseBoundaryCurve",
                            "One field-side limiting curve, with one "
                            "equilibrium and its own P,T,S,V per point.")
      .def_readonly("assemblage", &BoundaryCurve::assemblage)
      .def_readonly("points", &BoundaryCurve::points);
  py::class_<Boundary>(
      m, "PhaseBoundary",
      "Solved reaction curve. The boundary assemblage retains "
      "the zero-amount phase; side_a/side_b label its right/left "
      "neighbours in calculation-axis order.")
#define PROPERTY(name) .def_readonly(#name, &Boundary::name)
      PROPERTY(id) PROPERTY(zero_phase) PROPERTY(start_node) PROPERTY(end_node)
          PROPERTY(assemblage) PROPERTY(side_a) PROPERTY(side_b)
              PROPERTY(is_solution_replacement) PROPERTY(points)
                  PROPERTY(termination) PROPERTY(limiting_curves)
#undef PROPERTY
                      .def_property_readonly(
                          "reaction_kind", [](const Boundary &line) {
                            return reaction_name(line.reaction_kind);
                          });
  py::class_<Node>(m, "PhaseDiagramNode")
#define PROPERTY(name) .def_readonly(#name, &Node::name)
      PROPERTY(id) PROPERTY(gibbs_variance) PROPERTY(pt_nullity) PROPERTY(kind)
          PROPERTY(composition_coordinate) PROPERTY(pressure)
              PROPERTY(temperature) PROPERTY(entropy) PROPERTY(volume)
                  PROPERTY(zero_phases) PROPERTY(assemblage)
                      PROPERTY(incident_lines) PROPERTY(critical_mode)
                          PROPERTY(is_verified_critical_endpoint)
#undef PROPERTY
                              ;
  py::class_<Field>(m, "PhaseField",
                    "Samples sharing one assemblage, which may occupy several "
                    "disconnected fields. Its id is not the plot Field number.")
      .def_readonly("id", &Field::id)
      .def_readonly("phases", &Field::phases)
      .def_readonly("sample_indices", &Field::sample_indices);
  py::class_<Result>(
      m, "PhaseDiagramResult",
      "Equilibrium samples, traced reaction boundaries, junctions "
      "and diagnostics. resolved means no outstanding blocking diagnostics "
      "from implemented checks; finite sampling cannot guarantee "
      "field discovery. coordinate_diagnostics are informational precision "
      "limits and do not block resolution.")
      .def("to_dict", &result_dict,
           "Return a complete JSON-compatible result, including continuation "
           "states, settings and EOS-domain diagnostics.")
      .def_static(
          "from_dict",
          [](const py::dict &data) { return saved_result(data, true); },
          py::arg("data"),
          "Restore a complete saved calculation; candidate models are supplied "
          "separately when new equilibrium solves are needed.")
      .def_property_readonly(
          "settings", [](const Result &result) { return result.settings; },
          py::return_value_policy::copy)
      .def_property_readonly(
          "diagram_type",
          [](const Result &r) { return diagram_name(r.section.type); })
      .def_property_readonly(
          "composition_end",
          [](const Result &r) { return r.section.composition_end; })
      .def_property_readonly(
          "composition_range",
          [](const Result &r) { return r.section.composition_range; })
      .def_property_readonly(
          "entropy_range",
          [](const Result &r) { return r.section.entropy_range; })
      .def_property_readonly(
          "volume_range",
          [](const Result &r) { return r.section.volume_range; })
      .def_property_readonly(
          "fixed_coordinate",
          [](const Result &r) -> py::object {
            return r.section.fixed_coordinate
                       ? py::cast(coordinate_name(*r.section.fixed_coordinate))
                       : py::none();
          })
      .def_property_readonly(
          "fixed_value", [](const Result &r) { return r.section.fixed_value; })
      .def_property_readonly("coordinate_ranges", &Result::coordinate_ranges)
#define PROPERTY(name) .def_readonly(#name, &Result::name)
          PROPERTY(pressure_range) PROPERTY(temperature_range)
              PROPERTY(composition_start) PROPERTY(phase_names)
                  PROPERTY(samples) PROPERTY(fields) PROPERTY(boundaries)
                      PROPERTY(nodes) PROPERTY(resolved)
                          PROPERTY(equilibrium_solves)
                              PROPERTY(minimization_calls)
                                  PROPERTY(excluded_regions)
#undef PROPERTY
      .def_property_readonly(
          "diagnostics",
          diagnostic_view(&Result::diagnostics, diagnostic_messages))
      .def_property_readonly(
          "coordinate_diagnostics",
          diagnostic_view(&Result::coordinate_diagnostics, diagnostic_messages))
      .def_property_readonly(
          "spacing_diagnostics",
          diagnostic_view(&Result::spacing_diagnostics, diagnostic_messages))
      .def_property_readonly(
          "diagnostic_records",
          diagnostic_view(&Result::diagnostics, diagnostic_records))
      .def_property_readonly(
          "coordinate_diagnostic_records",
          diagnostic_view(&Result::coordinate_diagnostics, diagnostic_records))
      .def_property_readonly(
          "spacing_resolved",
          [](const Result &r) { return r.spacing_diagnostics.empty(); })
      .def_property_readonly("thermodynamic_validation", [](const Result &r) {
        return validation_dict(r.thermodynamic_validation);
      });
  py::class_<FieldPolygon>(m, "PhaseFieldPolygon",
                           "One plotted field: a closed outer coordinate ring "
                           "and any holes, with its inferred assemblage and "
                           "one-based field_number.")
      .def_readonly("field_number", &FieldPolygon::field_number)
      .def_readonly("field_id", &FieldPolygon::field_id)
      .def_readonly("n_phases", &FieldPolygon::n_phases)
      .def_readonly("sample_index", &FieldPolygon::sample_index)
      .def_readonly("has_open_boundary", &FieldPolygon::has_open_boundary)
      .def_readonly("is_provisional", &FieldPolygon::is_provisional)
      .def_readonly("has_unresolved_interior",
                    &FieldPolygon::has_unresolved_interior)
      .def_readonly("has_verified_reaction_perimeter",
                    &FieldPolygon::has_verified_reaction_perimeter)
      .def_readonly("has_closed_reaction_ring",
                    &FieldPolygon::has_closed_reaction_ring)
      .def_readonly("outside_model_domain", &FieldPolygon::outside_model_domain)
      .def_readonly("phases", &FieldPolygon::phases)
      .def_readonly("boundary_ids", &FieldPolygon::boundary_ids)
      .def_readonly("area", &FieldPolygon::area)
      .def_readonly("label_clearance", &FieldPolygon::label_clearance)
      .def("contains_rectangle", &FieldPolygon::contains_rectangle,
           py::arg("centre"), py::arg("half_size"),
           "Whether a label rectangle fits strictly inside the field, avoiding "
           "all holes. Coordinates and positive half-sizes use native diagram "
           "units; boundary contact returns False.",
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("vertices",
                             [](const FieldPolygon &p) { return p.vertices; })
      .def_property_readonly("holes",
                             [](const FieldPolygon &p) { return p.holes; })
      .def_property_readonly("label_position", [](const FieldPolygon &p) {
        return p.label_position;
      });
  py::class_<FieldPolygons>(m, "PhaseDiagramPolygons")
      .def_readonly("polygons", &FieldPolygons::polygons)
      .def_property_readonly(
          "diagnostics",
          diagnostic_view(&FieldPolygons::diagnostics, diagnostic_messages))
      .def_property_readonly(
          "coordinate_diagnostics",
          diagnostic_view(&FieldPolygons::coordinate_diagnostics,
                          diagnostic_messages))
      .def_property_readonly(
          "diagnostic_records",
          diagnostic_view(&FieldPolygons::diagnostics, diagnostic_records))
      .def_property_readonly(
          "coordinate_diagnostic_records",
          diagnostic_view(&FieldPolygons::coordinate_diagnostics,
                          diagnostic_records))
      .def_readonly("boundary_sides", &FieldPolygons::boundary_sides)
      .def_readonly("boundary_segments", &FieldPolygons::boundary_segments)
      .def_readonly("boundary_lines", &FieldPolygons::boundary_lines)
      .def_readonly("boundary_nodes", &FieldPolygons::boundary_nodes);
  py::class_<GridSolution>(m, "GridSolution")
      .def_readonly("state", &GridSolution::state)
      .def_readonly("n_moles", &GridSolution::n_moles)
      .def_readonly("gibbs", &GridSolution::gibbs)
      .def_readonly("molar_fractions", &GridSolution::molar_fractions)
      .def_readonly("phase_bases", &GridSolution::phase_bases)
      .def_readonly("phase_compositions", &GridSolution::phase_compositions)
      .def("to_dict", &grid_solution_dict);
  py::class_<PropertyGrid>(m, "PhaseDiagramPropertyGrid")
      .def("bind_phases", &bind_reconstruction_phases, py::arg("phases"),
           "Bind private original model copies for reconstruct_assemblage(). "
           "Validate candidate order/names without changing saved equilibria.")
      .def_property_readonly(
          "diagram_type",
          [](const PropertyGrid &r) { return diagram_name(r.diagram_type); })
      .def_property_readonly(
          "bulk_composition",
          [](const PropertyGrid &r) { return r.context.composition_start; },
          "Saved elemental mole amounts (starting bulk for X sections).")
      .def_readonly("x", &PropertyGrid::x)
      .def_readonly("y", &PropertyGrid::y)
      .def_readonly("values", &PropertyGrid::values)
      .def_readonly("field_numbers", &PropertyGrid::field_numbers)
      .def_readonly("success", &PropertyGrid::success)
      .def_readonly("equilibrium_success", &PropertyGrid::equilibrium_success)
      .def_readonly("relaxed", &PropertyGrid::relaxed)
      .def_readonly("resolved", &PropertyGrid::resolved)
      .def_property_readonly(
          "diagnostics",
          diagnostic_view(&PropertyGrid::diagnostics, diagnostic_messages))
      .def_property_readonly(
          "equilibrium_diagnostics",
          diagnostic_view(&PropertyGrid::equilibrium_diagnostics,
                          diagnostic_messages))
      .def_property_readonly(
          "equilibrium_diagnostic_records",
          diagnostic_view(&PropertyGrid::equilibrium_diagnostics,
                          diagnostic_records))
      .def_property_readonly(
          "diagnostic_records",
          diagnostic_view(&PropertyGrid::diagnostics, diagnostic_records))
      .def_readonly("context", &PropertyGrid::context)
      .def_property_readonly("thermodynamic_validation",
                             [](const PropertyGrid &r) {
                               return validation_dict(
                                   r.thermodynamic_validation);
                             })
      .def_property_readonly(
          "has_solutions",
          [](const PropertyGrid &r) { return !r.solutions.empty(); })
      .def(
          "solution_at",
          [](const PropertyGrid &r, Eigen::Index row, Eigen::Index col) {
            if (row < 0 || col < 0 || row >= r.y.size() || col >= r.x.size())
              throw py::index_error("Grid index out of range.");
            if (r.solutions.empty())
              throw py::value_error("Grid has no saved solutions; calculate "
                                    "with store_solutions=True.");
            auto saved = r.solutions.at(
                static_cast<std::size_t>(row * r.x.size() + col));
            if (saved) {
              saved->reconstruction_phases = r.reconstruction_phases;
              saved->max_phase_instances =
                  r.context.settings.max_phase_instances;
            }
            return saved;
          },
          py::arg("row"), py::arg("col"))
      .def("to_dict", &property_grid_dict)
      .def_static("from_dict", &saved_property_grid, py::arg("data"))
      .def_readonly("equilibrium_solves", &PropertyGrid::equilibrium_solves);
  m.def(
      "reconstruct_assemblage",
      [](const GridSolution &saved, py::object phases,
         types::AveragingType averaging_scheme) {
        auto models = saved.reconstruction_phases;
        if (!phases.is_none())
          models = std::make_shared<std::vector<std::shared_ptr<Material>>>(
              phases.cast<std::vector<std::shared_ptr<Material>>>());
        if (!models)
          throw py::value_error(
              "No candidate models are bound to this saved solution. "
              "Load the grid with phases=original_candidates or pass phases= "
              "to reconstruct_assemblage().");
        py::gil_scoped_release release;
        auto assemblage =
            reconstruct_assemblage(saved, *models, saved.max_phase_instances);
        assemblage->set_averaging_scheme(averaging_scheme);
        return assemblage;
      },
      py::arg("saved"), py::arg("phases") = py::none(), py::kw_only(),
      py::arg("averaging_scheme") = types::AveragingType::VRH,
      R"doc(Reconstruct an independent Assemblage from a retained GridSolution.

Restore its saved P,T, mole amounts, compositions and exact active bases.
No equilibrium solves or relaxation run. Each call returns private phase/model
copies, including separate coexisting instances. Newly calculated grids bind
their original models automatically. For loaded grids, supply phases= when
calling load_property_grid(), or pass the original candidates here explicitly.
averaging_scheme controls elastic averaging and defaults to VRH.
)doc");
  py::class_<PropertyGridder>(m, "_PropertyGridder")
      .def(py::init([](py::object previous,
                       std::vector<std::shared_ptr<Material>> phases,
                       Eigen::VectorXd x, Eigen::VectorXd y) {
        auto saved = calculation_result(previous);
        py::gil_scoped_release release;
        return std::make_unique<PropertyGridder>(
            std::move(saved), std::move(phases), std::move(x), std::move(y));
      }))
      .def("evaluate", &PropertyGridder::evaluate, py::arg("properties"),
           py::arg("x"), py::arg("y"), py::arg("relaxed"),
           py::arg("store_solutions"), py::call_guard<py::gil_scoped_release>())
      .def("retry", &PropertyGridder::retry, py::arg("grid"),
           py::call_guard<py::gil_scoped_release>())
      .def("audit", &PropertyGridder::audit, py::arg("grid"),
           py::arg("reverse") = false, py::arg("reset") = false,
           py::call_guard<py::gil_scoped_release>());
  m.def("_evaluate_property_grid", &evaluate_property_grid, py::arg("grid"),
        py::arg("phases"), py::arg("properties"), py::arg("relaxed"),
        py::arg("averaging_scheme") = types::AveragingType::VRH,
        py::call_guard<py::gil_scoped_release>());
  m.def("thermodynamic_property_names", &thermodynamic_property_names);
  m.def("assemblage_properties", &assemblage_properties, py::arg("assemblage"),
        py::arg("properties"), py::kw_only(), py::arg("relaxed") = true,
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "_phase_diagram_grid",
      [](py::object previous,
         const std::vector<std::shared_ptr<Material>> &phases,
         const std::vector<std::string> &properties, const Eigen::VectorXd &x,
         const Eigen::VectorXd &y, bool relaxed, bool store_solutions) {
        auto saved = calculation_result(previous);
        py::gil_scoped_release release;
        return phase_diagram_grid(saved, phases, properties, x, y, relaxed,
                                  store_solutions);
      },
      py::arg("result"), py::arg("phases"), py::arg("properties"), py::arg("x"),
      py::arg("y"), py::arg("relaxed"), py::arg("store_solutions") = false);
  m.def(
      "phase_diagram_field_polygons",
      [](py::object object, double tolerance, bool close_domain) {
        if (py::isinstance<Result>(object)) {
          const auto &result = object.cast<const Result &>();
          py::gil_scoped_release release;
          return field_polygons(result, tolerance, close_domain);
        }
        if (!py::isinstance<py::dict>(object))
          throw py::type_error(
              "Supply a PhaseDiagramResult or its saved JSON dictionary.");
        auto result = saved_result(object.cast<py::dict>());
        py::gil_scoped_release release;
        return field_polygons(result, tolerance, close_domain);
      },
      py::arg("result"), py::arg("tolerance") = 1.e-8,
      py::arg("close_domain") = true,
      R"doc(Reconstruct closed assemblage fields from saved equilibrium boundaries.

Accept a native result or saved dictionary; no equilibrium calculations are run.
A loop follows connected reaction arcs back to its start. Its coordinate ring
repeats the first point at the end; a field can have an outer ring and inner holes.
Saved junctions establish connectivity. Crossings of straight drawing segments
(chords) do not create thermodynamic junctions or automatically subdivide fields.

Labels use perimeter reaction assemblages and saved equilibria. Inspect each
polygon's is_provisional/has_open_boundary flags and the returned diagnostics.
has_verified_reaction_perimeter records a thermodynamic label certificate;
has_closed_reaction_ring records the fallback of a complete compatible ring;
has_unresolved_interior independently records coordinate uncertainty.
field_number is the one-based number used by plots, keys and field diagnostics.
tolerance is a distance after rescaling both calculation axes to 0..1.
close_domain permits the frame to close edge fields.
)doc");
  m.def(
      "verify_phase_diagram_field_perimeters",
      [](py::object object,
         const std::vector<std::shared_ptr<Material>> &phases) {
        auto result = calculation_result(object);
        py::gil_scoped_release release;
        return verify_field_perimeters(result, phases);
      },
      py::arg("result"), py::arg("phases"),
      "Verify field identities through stable bounding reactions and common "
      "junctions, testing possible crossings by corrected equilibria. "
      "Verified fields' precision limits are retained in "
      "coordinate_diagnostics "
      "and do not set resolved=False.");
  m.def(
      "verify_phase_diagram_critical_endpoints",
      [](py::object object,
         const std::vector<std::shared_ptr<Material>> &phases) {
        auto result = calculation_result(object);
        py::gil_scoped_release release;
        return verify_critical_endpoints(result, phases);
      },
      py::arg("result"), py::arg("phases"),
      R"doc(Test whether a dangling solvus reaction ends at a genuine critical point.

Use the original models to re-solve the coalescing solution compositions and
limiting univariant states. Return updated certificates/diagnostics, preserving
the saved traces, samples and unrelated messages. A line ending on the drawing
alone does not establish critical behaviour. Certificates are retained for later
plotting without additional solves.
)doc");
  m.def(
      "stable_equilibrium", &stable_equilibrium, py::arg("composition"),
      py::arg("phases"), py::arg("pressure"), py::arg("temperature"),
      py::arg("settings") = Settings{},
      "Find an equilibrium assemblage at P (Pa) and T (K) for elemental mole "
      "amounts. Search the candidate minerals/solutions, refine reaction "
      "equilibrium, and test affinities against competing phases. Return phase "
      "amounts/compositions and validation errors, or a failure message.",
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "stable_equilibrium",
      [](const types::FormulaMap &bulk,
         const std::vector<std::shared_ptr<Material>> &phases,
         std::optional<double> pressure, std::optional<double> temperature,
         std::optional<double> entropy, std::optional<double> volume,
         const Settings &settings, const std::array<double, 2> &pressure_range,
         const std::array<double, 2> &temperature_range) {
        std::string axes;
        Eigen::Vector2d position;
        const std::array<std::optional<double>, 4> values{pressure, temperature,
                                                          entropy, volume};
        for (std::size_t i = 0; i < values.size(); ++i)
          if (values[i]) {
            if (axes.size() < 2)
              position[static_cast<Eigen::Index>(axes.size())] = *values[i];
            axes += std::string("PTSV")[i];
          }
        if (axes.size() != 2)
          throw py::value_error("Specify exactly two of pressure, temperature, "
                                "entropy and volume.");
        const auto type = diagram_type(axes);
        py::gil_scoped_release release;
        return stable_equilibrium_at(bulk, phases, position, type, settings,
                                     pressure_range, temperature_range);
      },
      py::arg("composition"), py::arg("phases"), py::kw_only(),
      py::arg("pressure") = py::none(), py::arg("temperature") = py::none(),
      py::arg("entropy") = py::none(), py::arg("volume") = py::none(),
      py::arg("settings") = Settings{},
      py::arg("pressure_range") = std::array<double, 2>{0., 150.e9},
      py::arg("temperature_range") = std::array<double, 2>{1., 6000.},
      R"doc(Discover a stable assemblage at one point, without tracing a diagram.

Specify exactly two of pressure (Pa), temperature (K), entropy (total J/K),
and volume (total m^3). Use the same elemental mole scale for bulk, S and V.

This is the full phase-discovery calculation used for initial diagram grid
points: the candidate assemblage is not prescribed. P/T search ranges guide
inverse S/V solves and supply the initial physical seed. Return an EquilibriumState
with success, phase amounts/compositions and validation diagnostics.

P-S, T-V and S-V searches start from mass-balanced phase amounts and minimise
total enthalpy, Helmholtz energy and internal energy respectively when phases
need to change. A full P-T equilibrium search is a fallback. Bulk and the selected
coordinates stay constrained while competing phases enter. A failed EOS endpoint
does not suppress attempts from a valid endpoint. Every accepted state must still
pass reaction, mass-balance, site-bound and candidate-affinity validation.
)doc");
  m.def(
      "_equilibrium_path",
      [](const types::FormulaMap &bulk,
         const std::vector<std::shared_ptr<Material>> &phases,
         const std::vector<Eigen::Vector2d> &coordinates,
         const std::string &variables, const Settings &settings,
         const std::array<double, 2> &pressure_range,
         const std::array<double, 2> &temperature_range,
         const std::vector<std::string> &properties, bool relaxed,
         types::AveragingType averaging_scheme) {
        const auto type = diagram_type(variables);
        auto result = [&] {
          py::gil_scoped_release release;
          return equilibrium_path(bulk, phases, coordinates, type, settings,
                                  pressure_range, temperature_range, properties,
                                  relaxed, averaging_scheme);
        }();
        return py::make_tuple(result.states, result.values,
                              result.property_diagnostics,
                              result.equilibrium_solves);
      },
      py::arg("composition"), py::arg("phases"), py::arg("coordinates"),
      py::arg("variables"), py::arg("settings"), py::arg("pressure_range"),
      py::arg("temperature_range"), py::arg("properties"), py::arg("relaxed"),
      py::arg("averaging_scheme"));
  m.def(
      "phase_diagram",
      [](const types::FormulaMap &composition,
         const std::vector<std::shared_ptr<Material>> &phases,
         std::optional<std::array<double, 2>> pressure_range,
         std::optional<std::array<double, 2>> temperature_range,
         const Settings &settings, const std::string &diagram,
         const types::FormulaMap &composition_end,
         const std::array<double, 2> &composition_range,
         std::optional<double> pressure, std::optional<double> temperature,
         std::optional<std::array<double, 2>> entropy_range,
         std::optional<std::array<double, 2>> volume_range,
         std::optional<double> entropy, std::optional<double> volume) {
        CompositionSection section;
        section.type = diagram_type(diagram);
        section.composition_end = composition_end;
        section.composition_range = composition_range;
        const auto axes = diagram_axes(section.type);
        std::array<std::optional<std::array<double, 2>>, 4> ranges{
            pressure_range, temperature_range, entropy_range, volume_range};
        std::array<std::optional<double>, 4> values{pressure, temperature,
                                                    entropy, volume};
        for (std::size_t k = 0; k < 4; ++k) {
          const auto axis = static_cast<Coordinate>(k);
          const bool plotted = axis == axes[0] || axis == axes[1];
          if (values[k]) {
            if (ranges[k])
              throw py::value_error(
                  "Supply a coordinate value or its range, not both.");
            if (plotted || !has_composition_axis(section.type) ||
                section.fixed_coordinate)
              throw py::value_error(
                  "X diagrams fix exactly one coordinate outside their axes.");
            section.fixed_coordinate = axis;
            section.fixed_value = *values[k];
            ranges[k] = std::array<double, 2>{*values[k], *values[k]};
          }
          if (plotted && !ranges[k])
            throw py::value_error(
                "Specify both diagram coordinate ranges (missing " +
                coordinate_name(axis) + ").");
        }
        if (ranges[2])
          section.entropy_range = *ranges[2];
        if (ranges[3])
          section.volume_range = *ranges[3];
        if (!section.fixed_coordinate && has_composition_axis(section.type))
          throw py::value_error("Specify both diagram axes and one fixed "
                                "thermodynamic coordinate.");
        pressure_range = ranges[0].value_or(std::array<double, 2>{0., 150.e9});
        temperature_range =
            ranges[1].value_or(std::array<double, 2>{1., 6000.});
        py::gil_scoped_release release;
        return phase_diagram(composition, phases, *pressure_range,
                             *temperature_range, settings, section);
      },
      py::arg("composition"), py::arg("phases"),
      py::arg("pressure_range") = py::none(),
      py::arg("temperature_range") = py::none(),
      py::arg("settings") = Settings{}, py::kw_only(),
      py::arg("diagram") = "PT",
      py::arg("composition_end") = types::FormulaMap{},
      py::arg("composition_range") = std::array<double, 2>{0., 1.},
      py::arg("pressure") = py::none(), py::arg("temperature") = py::none(),
      py::arg("entropy_range") = py::none(),
      py::arg("volume_range") = py::none(), py::arg("entropy") = py::none(),
      py::arg("volume") = py::none(),
      R"doc(Calculate a mineral-assemblage diagram for a closed elemental bulk.

composition gives elemental mole amounts; phases supplies the candidate mineral
and solution models. Initial equilibrium sampling discovers assemblage changes.
Reaction continuation then follows phase-appearance/disappearance and solvus
boundaries, testing reaction equilibrium, mass balance and competing affinities.
Use phase_diagram_field_polygons() or plot_phase_diagram() to identify closed fields.

diagram selects PT, PS, PV, TS, TV, SV, PX, TX, SX or VX. Native units are Pa, K,
total J/K and total m^3. S and V scale with the supplied bulk amount. An X section
uses bulk(X)=(1-X)*composition+X*composition_end, without automatic normalisation,
and fixes one other thermodynamic coordinate, e.g. temperature for a P-X diagram.
Unplotted P/T ranges guide inverse seed searches (defaults 0-150 GPa, 1-6000 K).

settings controls seed density, tracing and acceptance tolerances. In the returned
result, resolved means that the implemented checks have no outstanding messages;
finite sampling cannot prove discovery of every disconnected stability field.
)doc");
  m.def(
      "refine_phase_diagram",
      [](const types::FormulaMap &bulk,
         const std::vector<std::shared_ptr<Material>> &phases,
         py::object previous, py::object settings,
         const std::optional<std::array<int, 2>> &resolution,
         bool reciprocal_volume,
         const std::optional<std::array<std::array<double, 2>, 2>> &region) {
        auto saved = calculation_result(previous);
        auto opts =
            settings.is_none() ? saved.settings : settings.cast<Settings>();
        std::optional<LineResolution> line_resolution;
        if (region && !resolution)
          throw py::value_error("A refinement region requires resolution.");
        if (resolution)
          line_resolution =
              LineResolution{*resolution, reciprocal_volume, region};
        py::gil_scoped_release release;
        return refine_phase_diagram(bulk, phases, saved, opts, line_resolution);
      },
      py::arg("composition"), py::arg("phases"), py::arg("previous"),
      py::arg("settings") = py::none(), py::kw_only(),
      py::arg("resolution") = py::none(), py::arg("reciprocal_volume") = false,
      py::arg("region") = py::none(),
      R"doc(Refine the equilibrium sampling of saved reaction lines.

Supply the original bulk, candidate order and phase-instance settings. previous
is a native result or full saved dictionary; omitted settings reuse its settings.
Connections, labels, nodes, samples and original points are preserved, including
unresolved problems. Refinement does not repair, retrace or relabel the diagram.
Without resolution, no equilibrium calculations are performed.

resolution gives point counts in calculation-axis order: 101 means 100 divisions.
It limits spacing along reaction lines, rather than adding an assemblage seed
grid. reciprocal_volume requests uniform density spacing. region gives a window
in native diagram units; counts then subdivide its spans. The complete diagram
and its existing diagnostics are retained. Unmet spacing is reported separately
in spacing_diagnostics.
)doc");
  m.def("water_fluid", py::overload_cast<>(&minerals::water_fluid));
}
} // namespace burnman::python
