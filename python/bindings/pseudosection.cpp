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

#include "burnman/tools/pseudosection.hpp"
#include "bindings.hpp"
#include "burnman/minerals/water.hpp"
#include <set>
namespace burnman::python {
namespace {
std::string coordinate_name(pseudosections::Coordinate axis) {
  return std::string(1, std::string("PTSVX")[static_cast<std::size_t>(axis)]);
}
pseudosections::Coordinate coordinate_type(const std::string &name) {
  auto i = std::string("PTSVX").find(name);
  if (name.size() != 1 || i == std::string::npos)
    throw py::value_error("Unknown coordinate: " + name);
  return static_cast<pseudosections::Coordinate>(i);
}
std::string diagram_name(pseudosections::DiagramType type) {
  auto axes = pseudosections::diagram_axes(type);
  return coordinate_name(axes[0]) + coordinate_name(axes[1]);
}
pseudosections::DiagramType diagram_type(std::string name) {
  name.erase(std::remove(name.begin(), name.end(), '-'), name.end());
  for (int i = 0; i < 10; ++i) {
    auto type = static_cast<pseudosections::DiagramType>(i);
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
  X(required_eos_phases)

py::dict settings_dict(const pseudosections::Settings &settings) {
  py::dict data;
#define SAVE(name) data[#name] = py::cast(settings.name);
  SETTINGS_OPTIONS(SAVE)
#undef SAVE
  return data;
}
pseudosections::Settings saved_settings(const py::dict &data) {
  pseudosections::Settings settings;
  std::set<std::string> names;
#define READ(name)                                                             \
  names.insert(#name);                                                         \
  if (data.contains(#name))                                                    \
    settings.name = data[#name].cast<decltype(settings.name)>();
  SETTINGS_OPTIONS(READ)
#undef READ
  for (auto item : data)
    if (!names.count(py::cast<std::string>(item.first)))
      throw py::value_error("Unknown pseudosection setting: " +
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
py::list
phase_states_dict(const std::vector<pseudosections::PhaseState> &states) {
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
    records.append(record);
  }
  return records;
}
py::dict boundary_point_dict(const pseudosections::BoundaryPoint &point) {
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
py::dict result_dict(const pseudosections::Result &result) {
  py::dict data;
  data["schema_version"] = 3;
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
  SAVE(diagnostics)
  SAVE(resolved)
  SAVE(equilibrium_solves)
  SAVE(minimization_calls)
#undef SAVE
  // Preserve the SI range keys used by the original example JSON files.
  data["calculation_pressure_range_Pa"] = data["pressure_range"];
  data["temperature_range_K"] = data["temperature_range"];
  data["settings"] = settings_dict(result.settings);
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
    py::list points;
    for (auto &point : line.points) {
      points.append(boundary_point_dict(point));
    }
    record["points"] = points;
    boundaries.append(record);
  }
  for (auto &state : result.samples) {
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
    samples.append(record);
  }
  data["fields"] = fields;
  data["nodes"] = nodes;
  data["boundaries"] = boundaries;
  data["samples"] = samples;
  return data;
}
// Geometry-only JSON remains sufficient for plotting. Resuming also requires
// the accepted phase compositions/amounts retained by Result::to_dict().
pseudosections::Result saved_result(const py::dict &data, bool full = false) {
  using namespace pseudosections;
  Result result;
  if (data.contains("diagram_type"))
    result.section.type =
        diagram_type(data["diagram_type"].cast<std::string>());
  if (data.contains("composition_start"))
    result.composition_start =
        data["composition_start"].cast<types::FormulaMap>();
  if (data.contains("composition_end"))
    result.section.composition_end =
        data["composition_end"].cast<types::FormulaMap>();
  if (data.contains("composition_range"))
    result.section.composition_range =
        data["composition_range"].cast<std::array<double, 2>>();
  if (data.contains("entropy_range"))
    result.section.entropy_range =
        data["entropy_range"].cast<std::array<double, 2>>();
  if (data.contains("volume_range"))
    result.section.volume_range =
        data["volume_range"].cast<std::array<double, 2>>();
  if (data.contains("fixed_coordinate")) {
    result.section.fixed_coordinate =
        coordinate_type(data["fixed_coordinate"].cast<std::string>());
    result.section.fixed_value = data["fixed_value"].cast<double>();
  }
  auto extensive = [&](const py::dict &record, const char *key,
                       Coordinate axis) {
    if (record.contains(key))
      return record[key].cast<double>();
    const auto axes = diagram_axes(result.section.type);
    if (axes[0] == axis || axes[1] == axis)
      throw py::value_error(std::string("Records in this diagram require ") +
                            key + ".");
    return 0.;
  };
  auto composition_coordinate = [&](const py::dict &record) {
    if (record.contains("composition_coordinate"))
      return record["composition_coordinate"].cast<double>();
    if (has_composition_axis(result.section.type))
      throw py::value_error("X records require composition_coordinate.");
    return 0.;
  };
  if (data.contains("settings"))
    result.settings = saved_settings(data["settings"].cast<py::dict>());
  else {
    // Legacy pyrolite results recorded their model policy at the top level.
    if (data.contains("excludes_invalid_eos"))
      result.settings.exclude_invalid_eos =
          data["excludes_invalid_eos"].cast<bool>();
    if (data.contains("active_solution_faces"))
      result.settings.active_solution_faces =
          data["active_solution_faces"].cast<bool>();
    if (data.contains("required_eos_phases"))
      result.settings.required_eos_phases =
          data["required_eos_phases"].cast<std::vector<std::string>>();
  }
  if (data.contains("resolved"))
    result.resolved = data["resolved"].cast<bool>();
  if (data.contains("diagnostics"))
    result.diagnostics = data["diagnostics"].cast<std::vector<std::string>>();
  result.pressure_range =
      data.contains("pressure_range")
          ? data["pressure_range"].cast<std::array<double, 2>>()
          : data["calculation_pressure_range_Pa"].cast<std::array<double, 2>>();
  result.temperature_range =
      data.contains("temperature_range")
          ? data["temperature_range"].cast<std::array<double, 2>>()
          : data["temperature_range_K"].cast<std::array<double, 2>>();
  if (data.contains("phase_names"))
    result.phase_names = data["phase_names"].cast<std::vector<std::string>>();
  if (data.contains("excluded_regions"))
    result.excluded_regions =
        data["excluded_regions"].cast<std::vector<Eigen::MatrixXd>>();
  auto phases = [&](py::handle records) {
    std::vector<PhaseState> out;
    for (auto item : py::reinterpret_borrow<py::iterable>(records)) {
      auto value = py::reinterpret_borrow<py::dict>(item);
      PhaseState phase;
      phase.id = value["id"].cast<int>();
      phase.amount =
          value.contains("amount") ? value["amount"].cast<double>() : 1.;
      if (full) {
        if (!value.contains("composition") ||
            !value.contains("candidate_index"))
          throw py::value_error("Resuming requires saved phase compositions "
                                "and candidate_index.");
        phase.candidate_index = value["candidate_index"].cast<int>();
        phase.composition = value["composition"].cast<Eigen::VectorXd>();
        if (value.contains("name"))
          phase.name = value["name"].cast<std::string>();
      }
      out.push_back(std::move(phase));
    }
    return out;
  };
  if (data.contains("nodes"))
    for (auto item : data["nodes"]) {
      auto value = py::reinterpret_borrow<py::dict>(item);
      Node node;
      node.id = value["id"].cast<int>();
      node.pressure = value["pressure"].cast<double>();
      node.temperature = value["temperature"].cast<double>();
      node.composition_coordinate = composition_coordinate(value);
      node.entropy = extensive(value, "entropy", Coordinate::S);
      node.volume = extensive(value, "volume", Coordinate::V);
      if (value.contains("incident_lines"))
        node.incident_lines = value["incident_lines"].cast<std::vector<int>>();
      if (full) {
        node.kind = value["kind"].cast<std::string>();
        node.assemblage = value["assemblage"].cast<std::vector<int>>();
        node.zero_phases = value["zero_phases"].cast<std::vector<int>>();
        node.gibbs_variance = value["gibbs_variance"].cast<int>();
        node.pt_nullity = value["pt_nullity"].cast<int>();
        if (value.contains("critical_mode"))
          node.critical_mode = value["critical_mode"].cast<Eigen::VectorXd>();
      }
      result.nodes.push_back(node);
    }
  for (auto item : data["boundaries"]) {
    auto value = py::reinterpret_borrow<py::dict>(item);
    Boundary line;
    if (full) {
      line.id = value["id"].cast<int>();
      line.zero_phase = value["zero_phase"].cast<int>();
      line.assemblage = value["assemblage"].cast<std::vector<int>>();
      line.termination = value["termination"].cast<std::string>();
    }
    if (value.contains("start_node"))
      line.start_node = value["start_node"].cast<int>();
    if (value.contains("end_node"))
      line.end_node = value["end_node"].cast<int>();
    if (value.contains("side_a"))
      line.side_a = value["side_a"].cast<std::vector<int>>();
    if (value.contains("side_b"))
      line.side_b = value["side_b"].cast<std::vector<int>>();
    if (value.contains("is_solution_replacement"))
      line.is_solution_replacement =
          value["is_solution_replacement"].cast<bool>();
    for (auto point : value["points"]) {
      auto record = py::reinterpret_borrow<py::dict>(point);
      BoundaryPoint p;
      p.pressure = record["pressure"].cast<double>();
      p.temperature = record["temperature"].cast<double>();
      p.composition_coordinate = composition_coordinate(record);
      p.entropy = extensive(record, "entropy", Coordinate::S);
      p.volume = extensive(record, "volume", Coordinate::V);
      if (full) {
        p.phases = phases(record["phases"]);
        p.mass_balance_error = record["mass_balance_error"].cast<double>();
        p.minimum_affinity = record["minimum_affinity"].cast<double>();
        p.residual = record["residual"].cast<double>();
      }
      line.points.push_back(p);
    }
    result.boundaries.push_back(std::move(line));
  }
  if (data.contains("fields"))
    for (auto item : data["fields"]) {
      auto value = py::reinterpret_borrow<py::dict>(item);
      Field field;
      field.id = value["id"].cast<int>();
      field.phases = value["phases"].cast<std::vector<int>>();
      if (full)
        field.sample_indices = value["sample_indices"].cast<std::vector<int>>();
      result.fields.push_back(field);
    }
  if (data.contains("samples"))
    for (auto item : data["samples"]) {
      auto value = py::reinterpret_borrow<py::dict>(item);
      State state;
      state.success = value["success"].cast<bool>();
      state.pressure = value["pressure"].cast<double>();
      if (value.contains("is_field_verification"))
        state.is_field_verification =
            value["is_field_verification"].cast<bool>();
      if (value.contains("outside_model_domain"))
        state.outside_model_domain = value["outside_model_domain"].cast<bool>();
      state.temperature = value["temperature"].cast<double>();
      state.composition_coordinate = composition_coordinate(value);
      state.entropy = extensive(value, "entropy", Coordinate::S);
      state.volume = extensive(value, "volume", Coordinate::V);
      if (value.contains("excluded_phases"))
        state.excluded_phases =
            value["excluded_phases"].cast<std::vector<std::string>>();
      state.phases = phases(value["phases"]);
      if (full) {
        state.message = value["message"].cast<std::string>();
        state.gibbs = value["gibbs"].cast<double>();
        state.mass_balance_error = value["mass_balance_error"].cast<double>();
        state.minimum_affinity = value["minimum_affinity"].cast<double>();
        state.equilibrium_error = value["equilibrium_error"].cast<double>();
      }
      result.samples.push_back(std::move(state));
    }
  if (full) {
    if (data.contains("equilibrium_solves"))
      result.equilibrium_solves = data["equilibrium_solves"].cast<int>();
    if (data.contains("minimization_calls"))
      result.minimization_calls = data["minimization_calls"].cast<int>();
  }
  return result;
}
} // namespace
void bind_pseudosection(py::module_ &m) {
  using namespace pseudosections;
  py::class_<Settings>(m, "PseudosectionSettings")
      .def(py::init<>())
      .def("to_dict", &settings_dict,
           "Return all settings as a JSON-compatible dictionary.")
      .def_static("from_dict", &saved_settings, py::arg("data"))
#define OPTION(name) .def_readwrite(#name, &Settings::name)
          SETTINGS_OPTIONS(OPTION)
#undef OPTION
#undef SETTINGS_OPTIONS
      ;
  py::class_<PhaseState>(m, "PhaseState")
      .def_readonly("id", &PhaseState::id)
      .def_readonly("candidate_index", &PhaseState::candidate_index)
      .def_readonly("name", &PhaseState::name)
      .def_readonly("amount", &PhaseState::amount)
      .def_property_readonly("composition",
                             [](const PhaseState &s) { return s.composition; });
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
  py::class_<Boundary>(m, "PhaseBoundary")
#define PROPERTY(name) .def_readonly(#name, &Boundary::name)
      PROPERTY(id) PROPERTY(zero_phase) PROPERTY(start_node) PROPERTY(end_node)
          PROPERTY(assemblage) PROPERTY(side_a) PROPERTY(side_b)
              PROPERTY(is_solution_replacement) PROPERTY(points)
                  PROPERTY(termination)
#undef PROPERTY
                      ;
  py::class_<Node>(m, "PhaseDiagramNode")
#define PROPERTY(name) .def_readonly(#name, &Node::name)
      PROPERTY(id) PROPERTY(gibbs_variance) PROPERTY(pt_nullity) PROPERTY(kind)
          PROPERTY(composition_coordinate) PROPERTY(pressure)
              PROPERTY(temperature) PROPERTY(entropy) PROPERTY(volume)
                  PROPERTY(zero_phases) PROPERTY(assemblage)
                      PROPERTY(incident_lines) PROPERTY(critical_mode)
#undef PROPERTY
                          ;
  py::class_<Field>(m, "PhaseField")
      .def_readonly("id", &Field::id)
      .def_readonly("phases", &Field::phases)
      .def_readonly("sample_indices", &Field::sample_indices);
  py::class_<Result>(m, "PseudosectionResult")
      .def("to_dict", &result_dict,
           "Return a complete JSON-compatible result, including continuation "
           "states, settings and EOS-domain diagnostics.")
      .def_static(
          "from_dict",
          [](const py::dict &data) { return saved_result(data, true); },
          py::arg("data"))
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
                      PROPERTY(nodes) PROPERTY(diagnostics) PROPERTY(resolved)
                          PROPERTY(equilibrium_solves)
                              PROPERTY(minimization_calls)
                                  PROPERTY(excluded_regions)
#undef PROPERTY
      ;
  py::class_<FieldPolygon>(m, "PhaseFieldPolygon")
      .def_readonly("field_id", &FieldPolygon::field_id)
      .def_readonly("n_phases", &FieldPolygon::n_phases)
      .def_readonly("sample_index", &FieldPolygon::sample_index)
      .def_readonly("has_open_boundary", &FieldPolygon::has_open_boundary)
      .def_readonly("outside_model_domain", &FieldPolygon::outside_model_domain)
      .def_readonly("phases", &FieldPolygon::phases)
      .def_readonly("source_regions", &FieldPolygon::source_regions)
      .def_readonly("area", &FieldPolygon::area)
      .def_readonly("label_clearance", &FieldPolygon::label_clearance)
      .def("contains_rectangle", &FieldPolygon::contains_rectangle,
           py::arg("centre"), py::arg("half_size"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("vertices",
                             [](const FieldPolygon &p) { return p.vertices; })
      .def_property_readonly("holes",
                             [](const FieldPolygon &p) { return p.holes; })
      .def_property_readonly("label_position", [](const FieldPolygon &p) {
        return p.label_position;
      });
  py::class_<FieldPolygons>(m, "PseudosectionPolygons")
      .def_readonly("polygons", &FieldPolygons::polygons)
      .def_readonly("diagnostics", &FieldPolygons::diagnostics)
      .def_readonly("boundary_segments", &FieldPolygons::boundary_segments)
      .def_readonly("boundary_nodes", &FieldPolygons::boundary_nodes);
  py::class_<ContourSettings>(m, "PseudosectionContourSettings")
      .def(py::init<>())
      .def_readwrite("seed_grid", &ContourSettings::seed_grid)
      .def_readwrite("step", &ContourSettings::step)
      .def_readwrite("min_step", &ContourSettings::min_step)
      .def_readwrite("max_trace_steps", &ContourSettings::max_trace_steps);
  py::class_<ContourLine>(m, "PseudosectionContourLine")
      .def_readonly("field_id", &ContourLine::field_id)
      .def_readonly("phases", &ContourLine::phases)
      .def_readonly("points", &ContourLine::points)
      .def_readonly("closed", &ContourLine::closed)
      .def_readonly("termination", &ContourLine::termination);
  py::class_<ContourResult>(m, "PseudosectionContourResult")
      .def_property_readonly(
          "diagram_type",
          [](const ContourResult &r) { return diagram_name(r.diagram_type); })
      .def_readonly("lines", &ContourResult::lines)
      .def_readonly("resolved", &ContourResult::resolved)
      .def_readonly("diagnostics", &ContourResult::diagnostics)
      .def_readonly("equilibrium_solves", &ContourResult::equilibrium_solves)
      .def("to_dict", [](const ContourResult &r) {
        py::dict data;
        data["schema_version"] = 1;
        data["diagram_type"] = diagram_name(r.diagram_type);
        data["resolved"] = r.resolved;
        data["diagnostics"] = r.diagnostics;
        data["equilibrium_solves"] = r.equilibrium_solves;
        py::list lines;
        for (const auto &line : r.lines) {
          py::dict record;
          record["field_id"] = line.field_id;
          record["phases"] = line.phases;
          record["closed"] = line.closed;
          record["termination"] = line.termination;
          py::list points;
          for (const auto &point : line.points)
            points.append(boundary_point_dict(point));
          record["points"] = points;
          lines.append(record);
        }
        data["lines"] = lines;
        return data;
      });
  m.def(
      "pseudosection_contours",
      [](py::object previous,
         const std::vector<std::shared_ptr<Material>> &phases,
         py::object constraint, const ContourSettings &settings,
         const std::optional<types::FormulaMap> &composition) {
        Result saved;
        if (py::isinstance<Result>(previous))
          saved = previous.cast<Result>();
        else if (py::isinstance<py::dict>(previous))
          saved = saved_result(previous.cast<py::dict>(), true);
        else
          throw py::type_error("Supply a PseudosectionResult or its full saved "
                               "JSON dictionary.");
        if (composition) {
          if (!saved.composition_start.empty() &&
              saved.composition_start != *composition)
            throw py::value_error("composition must match the saved bulk.");
          saved.composition_start = *composition;
        }
        using C = equilibration::EqualityConstraint;
        ContourConstraintFactory factory;
        if (py::isinstance<C>(constraint)) {
          auto native = constraint.cast<std::shared_ptr<C>>();
          factory = [native](const Assemblage &, const auto &, const auto &) {
            return native->clone();
          };
        } else if (PyCallable_Check(constraint.ptr())) {
          factory = [constraint](const Assemblage &a, const auto &prm,
                                 const auto &ids) -> std::unique_ptr<C> {
            py::gil_scoped_acquire acquire;
            // Callbacks receive independent copies; they cannot mutate the
            // native warm state or retain a reference to a temporary layout.
            auto context = std::make_shared<Assemblage>(a);
            py::object value = constraint(
                context, py::cast(prm, py::return_value_policy::copy), ids);
            if (value.is_none())
              return nullptr;
            if (!py::isinstance<C>(value))
              throw py::type_error("Constraint factory must return an "
                                   "EqualityConstraint or None.");
            return value.cast<std::shared_ptr<C>>()->clone();
          };
        } else
          throw py::type_error("constraint must be an EqualityConstraint or a "
                               "callable factory.");
        py::gil_scoped_release release;
        return pseudosection_contours(saved, phases, factory, settings);
      },
      py::arg("result"), py::arg("phases"), py::arg("constraint"),
      py::kw_only(), py::arg("settings") = ContourSettings{},
      py::arg("composition") = py::none(),
      "Trace any native equality constraint through saved closed fields. "
      "Accept a constraint or factory(assemblage, parameters, phase_ids); "
      "return None from the factory in fields where it is undefined. "
      "Build PhaseFraction/PhaseComposition/LinearX constraints in a factory "
      "using that field's parameter layout. The factory is called once per "
      "field; all seeding, solves and continuation run in C++. Supply the same "
      "candidate models in their original order. composition supplies the "
      "elemental bulk for legacy JSON without composition_start. Increase "
      "settings.seed_grid to check disconnected contour discovery.");
  m.def(
      "pseudosection_field_polygons",
      [](py::object object, double tolerance, bool close_domain,
         bool merge_fields) {
        if (py::isinstance<Result>(object)) {
          const auto &result = object.cast<const Result &>();
          py::gil_scoped_release release;
          return field_polygons(result, tolerance, close_domain, merge_fields);
        }
        if (!py::isinstance<py::dict>(object))
          throw py::type_error(
              "Supply a PseudosectionResult or its saved JSON dictionary.");
        auto result = saved_result(object.cast<py::dict>());
        py::gil_scoped_release release;
        return field_polygons(result, tolerance, close_domain, merge_fields);
      },
      py::arg("result"), py::arg("tolerance") = 1.e-8,
      py::arg("close_domain") = true, py::arg("merge_fields") = true,
      "Build closed field polygons in C++, including holes. Merge adjacent "
      "identical assemblages by default; source_regions preserves raw face "
      "indices. Accept a native result or saved JSON dictionary.");
  m.def("stable_equilibrium", &stable_equilibrium, py::arg("composition"),
        py::arg("phases"), py::arg("pressure"), py::arg("temperature"),
        py::arg("settings") = Settings{},
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "pseudosection",
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
        if (!section.fixed_coordinate && has_composition_axis(section.type)) {
          for (std::size_t k = 0; k < 4; ++k)
            if (static_cast<Coordinate>(k) != axes[0] && ranges[k] &&
                (*ranges[k])[0] == (*ranges[k])[1]) {
              if (section.fixed_coordinate)
                throw py::value_error(
                    "X diagrams require exactly one fixed coordinate.");
              section.fixed_coordinate = static_cast<Coordinate>(k);
              section.fixed_value = (*ranges[k])[0];
            }
          if (!section.fixed_coordinate)
            throw py::value_error(
                "Specify both diagram axes and one fixed thermodynamic "
                "coordinate; PX "
                "requires fixed temperature and TX fixed pressure by default.");
        }
        pressure_range = ranges[0].value_or(std::array<double, 2>{0., 150.e9});
        temperature_range =
            ranges[1].value_or(std::array<double, 2>{1., 6000.});
        py::gil_scoped_release release;
        return pseudosection(composition, phases, *pressure_range,
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
      "Trace any pair of P,T,S,V,X with native equilibrate continuation. "
      "S (J/K) and V (m^3) are totals on the supplied bulk amount scale. "
      "X diagrams fix one other thermodynamic coordinate and use "
      "bulk(X)=(1-X)*composition+X*composition_end. Unplotted P/T ranges "
      "bound inverse seed searches (defaults 0-150 GPa and 1-6000 K).");
  m.def(
      "refine_pseudosection",
      [](const types::FormulaMap &bulk,
         const std::vector<std::shared_ptr<Material>> &phases,
         py::object previous, py::object settings) {
        Result saved;
        if (py::isinstance<Result>(previous))
          saved = previous.cast<Result>();
        else if (py::isinstance<py::dict>(previous))
          saved = saved_result(previous.cast<py::dict>(), true);
        else
          throw py::type_error("Supply a PseudosectionResult or its full saved "
                               "JSON dictionary.");
        auto opts =
            settings.is_none() ? saved.settings : settings.cast<Settings>();
        py::gil_scoped_release release;
        return refine_pseudosection(bulk, phases, saved, opts);
      },
      py::arg("composition"), py::arg("phases"), py::arg("previous"),
      py::arg("settings") = py::none(),
      "Resume unfinished phase lines from saved accepted states, using the "
      "same bulk and candidate models. Omitted settings reuse the saved "
      "calculation settings, including its EOS policy. All continuation runs "
      "in C++.");
  m.def("water_fluid", py::overload_cast<>(&minerals::water_fluid));
}
} // namespace burnman::python
