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
#include <charconv>
#include <set>
namespace burnman::python {
namespace {
std::vector<std::string>
normalize_field_diagnostics(std::vector<std::string> messages) {
  const std::string region = "Region ";
  const std::string sampled = "No closed region resolved for sampled field ";
  for (auto &message : messages) {
    if (message.compare(0, region.size(), region) == 0) {
      const auto end = message.find(' ', region.size());
      if (end == std::string::npos)
        continue;
      int index = -1;
      auto parsed = std::from_chars(message.data() + region.size(),
                                    message.data() + end, index);
      if (parsed.ec == std::errc{} && parsed.ptr == message.data() + end &&
          index >= 0 && index < std::numeric_limits<int>::max())
        message = "Field " + std::to_string(index + 1) + message.substr(end);
    } else if (message.compare(0, sampled.size(), sampled) == 0)
      message.replace(0, sampled.size(),
                      "No closed region resolved for sampled assemblage ");
  }
  return messages;
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
  X(required_eos_phases)

py::dict settings_dict(const phase_diagrams::Settings &settings) {
  py::dict data;
#define SAVE(name) data[#name] = py::cast(settings.name);
  SETTINGS_OPTIONS(SAVE)
#undef SAVE
  return data;
}
phase_diagrams::Settings saved_settings(const py::dict &data) {
  phase_diagrams::Settings settings;
  std::set<std::string> names;
#define READ(name)                                                             \
  names.insert(#name);                                                         \
  if (data.contains(#name))                                                    \
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
    records.append(record);
  }
  return records;
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
py::dict result_dict(const phase_diagrams::Result &result) {
  py::dict data;
  data["schema_version"] = 4;
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
  SAVE(spacing_diagnostics)
  data["spacing_resolved"] = result.spacing_diagnostics.empty();
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
phase_diagrams::Result saved_result(const py::dict &data, bool full = false) {
  using namespace phase_diagrams;
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
    result.diagnostics = normalize_field_diagnostics(
        data["diagnostics"].cast<std::vector<std::string>>());
  if (data.contains("spacing_diagnostics"))
    result.spacing_diagnostics =
        data["spacing_diagnostics"].cast<std::vector<std::string>>();
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
      if (!full)
        node.gibbs_variance = node.pt_nullity = -1; // unknown unless supplied
      node.id = value["id"].cast<int>();
      node.pressure = value["pressure"].cast<double>();
      node.temperature = value["temperature"].cast<double>();
      node.composition_coordinate = composition_coordinate(value);
      node.entropy = extensive(value, "entropy", Coordinate::S);
      node.volume = extensive(value, "volume", Coordinate::V);
      if (value.contains("incident_lines"))
        node.incident_lines = value["incident_lines"].cast<std::vector<int>>();
      if (value.contains("is_verified_critical_endpoint"))
        node.is_verified_critical_endpoint =
            value["is_verified_critical_endpoint"].cast<bool>();
      // Polygon construction also needs invariant metadata when plotting a
      // saved dictionary, without requiring all resumable equilibrium data.
      if (full || value.contains("kind"))
        node.kind = value["kind"].cast<std::string>();
      if (full || value.contains("assemblage"))
        node.assemblage = value["assemblage"].cast<std::vector<int>>();
      if (full || value.contains("gibbs_variance"))
        node.gibbs_variance = value["gibbs_variance"].cast<int>();
      if (full || value.contains("pt_nullity"))
        node.pt_nullity = value["pt_nullity"].cast<int>();
      if (full || value.contains("zero_phases"))
        node.zero_phases = value["zero_phases"].cast<std::vector<int>>();
      if (full) {
        if (value.contains("critical_mode"))
          node.critical_mode = value["critical_mode"].cast<Eigen::VectorXd>();
      }
      result.nodes.push_back(node);
    }
  for (auto item : data["boundaries"]) {
    auto value = py::reinterpret_borrow<py::dict>(item);
    Boundary line;
    if (full || value.contains("assemblage"))
      line.assemblage = value["assemblage"].cast<std::vector<int>>();
    // Loop constraints need the reaction metadata for dictionary plotting too.
    if (full || value.contains("id"))
      line.id = value["id"].cast<int>();
    if (full || value.contains("zero_phase"))
      line.zero_phase = value["zero_phase"].cast<int>();
    if (full) {
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
  detail::separate_spacing_diagnostics(result);
  return result;
}
py::dict grid_solution_dict(const phase_diagrams::GridSolution &solution) {
  auto data = boundary_point_dict(solution.state);
  data["n_moles"] = solution.n_moles;
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
  data["schema_version"] = 1;
  data["context"] = result_dict(grid.context);
  data["x"] = vector_list(grid.x);
  data["y"] = vector_list(grid.y);
  data["relaxed"] = grid.relaxed;
  data["resolved"] = grid.resolved;
  data["field_numbers"] = py::cast(grid.field_numbers).attr("tolist")();
  data["success"] = py::cast(grid.success).attr("tolist")();
  data["equilibrium_success"] =
      py::cast(grid.equilibrium_success).attr("tolist")();
  data["diagnostics"] = grid.diagnostics;
  data["equilibrium_diagnostics"] = grid.equilibrium_diagnostics;
  data["equilibrium_solves"] = grid.equilibrium_solves;
  py::dict values;
  for (const auto &[name, matrix] : grid.values) {
    py::list rows;
    for (Eigen::Index j = 0; j < matrix.rows(); ++j) {
      py::list row;
      for (Eigen::Index i = 0; i < matrix.cols(); ++i)
        if (std::isfinite(matrix(j, i)))
          row.append(matrix(j, i));
        else
          row.append(py::none());
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
      data["schema_version"].cast<int>() != 1)
    throw py::value_error(
        "Unsupported property grid format or schema version.");
  PropertyGrid grid;
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
  grid.diagnostics = data["diagnostics"].cast<std::vector<std::string>>();
  grid.equilibrium_diagnostics =
      data["equilibrium_diagnostics"].cast<std::vector<std::string>>();
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
      const auto row = rows[static_cast<py::ssize_t>(j)].cast<py::list>();
      if (static_cast<Eigen::Index>(row.size()) != grid.x.size())
        throw py::value_error("Saved solutions do not match the grid shape.");
      for (Eigen::Index i = 0; i < grid.x.size(); ++i) {
        const auto item = row[static_cast<py::ssize_t>(i)];
        if (item.is_none()) {
          if (grid.equilibrium_success(j, i))
            throw py::value_error(
                "A successful grid point is missing its solution.");
          grid.solutions.push_back(std::nullopt);
          continue;
        }
        if (!grid.equilibrium_success(j, i) || grid.field_numbers(j, i) <= 0)
          throw py::value_error(
              "A saved solution has no successful equilibrium or field.");
        auto record = item.cast<py::dict>();
        GridSolution solution;
        solution.n_moles = record["n_moles"].cast<double>();
        solution.molar_fractions =
            record["molar_fractions"].cast<Eigen::ArrayXd>();
        auto &point = solution.state;
#define READ(name) point.name = record[#name].cast<decltype(point.name)>();
        READ(pressure)
        READ(temperature)
        READ(entropy)
        READ(volume)
        READ(composition_coordinate) READ(mass_balance_error)
            READ(minimum_affinity) READ(residual)
#undef READ
                for (auto entry : record["phases"].cast<py::list>()) {
          auto p = entry.cast<py::dict>();
          PhaseState phase;
#define READ(name) phase.name = p[#name].cast<decltype(phase.name)>();
          READ(id)
          READ(candidate_index)
          READ(name)
          READ(amount)
          READ(composition)
#undef READ
              point.phases.push_back(std::move(phase));
        }
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
  m.def("_normalize_phase_diagram_diagnostics", &normalize_field_diagnostics);
  using namespace phase_diagrams;
  py::class_<Settings>(m, "PhaseDiagramSettings",
                       "Initial equilibrium sampling, reaction continuation "
                       "and numerical acceptance criteria for a diagram.")
      .def(py::init<>())
      .def("to_dict", &settings_dict,
           "Return all settings as a JSON-compatible dictionary.")
      .def_static(
          "from_dict", &saved_settings, py::arg("data"),
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
  py::class_<Boundary>(
      m, "PhaseBoundary",
      "Solved reaction curve. The boundary assemblage retains "
      "the zero-amount phase; side_a/side_b label its right/left "
      "neighbours in calculation-axis order.")
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
      "and diagnostics. resolved means no outstanding diagnostics "
      "from implemented checks; finite sampling cannot guarantee "
      "field discovery.")
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
                      PROPERTY(nodes) PROPERTY(diagnostics) PROPERTY(resolved)
                          PROPERTY(spacing_diagnostics)
                              PROPERTY(equilibrium_solves)
                                  PROPERTY(minimization_calls)
                                      PROPERTY(excluded_regions)
#undef PROPERTY
      .def_property_readonly("spacing_resolved", [](const Result &r) {
        return r.spacing_diagnostics.empty();
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
      .def_readonly("outside_model_domain", &FieldPolygon::outside_model_domain)
      .def_readonly("phases", &FieldPolygon::phases)
      .def_readonly("source_regions", &FieldPolygon::source_regions)
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
      .def_readonly("diagnostics", &FieldPolygons::diagnostics)
      .def_readonly("boundary_sides", &FieldPolygons::boundary_sides)
      .def_readonly("boundary_segments", &FieldPolygons::boundary_segments)
      .def_readonly("boundary_lines", &FieldPolygons::boundary_lines)
      .def_readonly("boundary_nodes", &FieldPolygons::boundary_nodes);
  py::class_<ContourSettings>(m, "PhaseDiagramContourSettings")
      .def(py::init<>())
      .def_readwrite("seed_grid", &ContourSettings::seed_grid)
      .def_readwrite("step", &ContourSettings::step)
      .def_readwrite("min_step", &ContourSettings::min_step)
      .def_readwrite("max_trace_steps", &ContourSettings::max_trace_steps);
  py::class_<ContourLine>(m, "PhaseDiagramContourLine")
      .def_readonly("field_number", &ContourLine::field_number)
      .def_readonly("field_id", &ContourLine::field_id)
      .def_readonly("phases", &ContourLine::phases)
      .def_readonly("points", &ContourLine::points)
      .def_readonly("closed", &ContourLine::closed)
      .def_readonly("termination", &ContourLine::termination);
  py::class_<ContourResult>(m, "PhaseDiagramContourResult")
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
          record["field_number"] = line.field_number;
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
  py::class_<GridSolution>(m, "GridSolution")
      .def_readonly("state", &GridSolution::state)
      .def_readonly("n_moles", &GridSolution::n_moles)
      .def_readonly("molar_fractions", &GridSolution::molar_fractions)
      .def_readonly("phase_bases", &GridSolution::phase_bases)
      .def_readonly("phase_compositions", &GridSolution::phase_compositions)
      .def("to_dict", &grid_solution_dict);
  py::class_<PropertyGrid>(m, "PhaseDiagramPropertyGrid")
      .def_property_readonly(
          "diagram_type",
          [](const PropertyGrid &r) { return diagram_name(r.diagram_type); })
      .def_readonly("x", &PropertyGrid::x)
      .def_readonly("y", &PropertyGrid::y)
      .def_readonly("values", &PropertyGrid::values)
      .def_readonly("field_numbers", &PropertyGrid::field_numbers)
      .def_readonly("success", &PropertyGrid::success)
      .def_readonly("equilibrium_success", &PropertyGrid::equilibrium_success)
      .def_readonly("relaxed", &PropertyGrid::relaxed)
      .def_readonly("resolved", &PropertyGrid::resolved)
      .def_readonly("diagnostics", &PropertyGrid::diagnostics)
      .def_readonly("equilibrium_diagnostics",
                    &PropertyGrid::equilibrium_diagnostics)
      .def_readonly("context", &PropertyGrid::context)
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
            return r.solutions.at(
                static_cast<std::size_t>(row * r.x.size() + col));
          },
          py::arg("row"), py::arg("col"))
      .def("to_dict", &property_grid_dict)
      .def_static("from_dict", &saved_property_grid, py::arg("data"))
      .def_readonly("equilibrium_solves", &PropertyGrid::equilibrium_solves);
  m.def("_evaluate_property_grid", &evaluate_property_grid, py::arg("grid"),
        py::arg("phases"), py::arg("properties"), py::arg("relaxed"),
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
        Result saved;
        if (py::isinstance<Result>(previous))
          saved = previous.cast<Result>();
        else if (py::isinstance<py::dict>(previous))
          saved = saved_result(previous.cast<py::dict>(), true);
        else
          throw py::type_error(
              "Supply a PhaseDiagramResult or saved JSON dictionary.");
        py::gil_scoped_release release;
        return phase_diagram_grid(saved, phases, properties, x, y, relaxed,
                                  store_solutions);
      },
      py::arg("result"), py::arg("phases"), py::arg("properties"), py::arg("x"),
      py::arg("y"), py::arg("relaxed"), py::arg("store_solutions") = false);
  m.def(
      "phase_diagram_contours",
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
          throw py::type_error("Supply a PhaseDiagramResult or its full saved "
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
        return phase_diagram_contours(saved, phases, factory, settings);
      },
      py::arg("result"), py::arg("phases"), py::arg("constraint"),
      py::kw_only(), py::arg("settings") = ContourSettings{},
      py::arg("composition") = py::none(),
      R"doc(Calculate a property contour or mineral-composition isopleth within saved fields.

constraint fixes the contour value, for example a garnet composition or total
volume. Supply a native equality constraint or a factory called once per field
as factory(assemblage, parameters, phase_ids). Build phase-dependent constraints
using that field's parameter layout; return None where the quantity is undefined.

Use the original candidate models in their original order. The calculation holds
each saved field's assemblage and follows equilibrium to its outer ring or holes;
it does not repeat phase discovery or validate the source field identification.
Increase settings.seed_grid to assess discovery of disconnected contours.
composition supplies the elemental bulk for legacy JSON without composition_start.
)doc");
  m.def(
      "phase_diagram_field_polygons",
      [](py::object object, double tolerance, bool close_domain,
         bool merge_fields) {
        if (py::isinstance<Result>(object)) {
          const auto &result = object.cast<const Result &>();
          py::gil_scoped_release release;
          return field_polygons(result, tolerance, close_domain, merge_fields);
        }
        if (!py::isinstance<py::dict>(object))
          throw py::type_error(
              "Supply a PhaseDiagramResult or its saved JSON dictionary.");
        auto result = saved_result(object.cast<py::dict>());
        py::gil_scoped_release release;
        return field_polygons(result, tolerance, close_domain, merge_fields);
      },
      py::arg("result"), py::arg("tolerance") = 1.e-8,
      py::arg("close_domain") = true, py::arg("merge_fields") = true,
      R"doc(Reconstruct closed assemblage fields from saved equilibrium boundaries.

Accept a native result or saved dictionary; no equilibrium calculations are run.
A loop follows connected reaction arcs back to its start. Its coordinate ring
repeats the first point at the end; a field can have an outer ring and inner holes.
Saved junctions establish connectivity. Crossings of straight drawing segments
(chords) do not create thermodynamic junctions or automatically subdivide fields.

Labels use perimeter reaction assemblages and saved equilibria. Inspect each
polygon's is_provisional/has_open_boundary flags and the returned diagnostics.
field_number is the one-based number used by plots, keys and field diagnostics.
tolerance is a distance after rescaling both calculation axes to 0..1.
close_domain permits the frame to close edge fields. merge_fields is retained
for compatibility and has no effect.
)doc");
  m.def(
      "verify_phase_diagram_critical_endpoints",
      [](py::object object,
         const std::vector<std::shared_ptr<Material>> &phases) {
        auto result = py::isinstance<Result>(object)
                          ? object.cast<Result>()
                          : saved_result(object.cast<py::dict>(), true);
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
)doc");
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
        Result saved;
        if (py::isinstance<Result>(previous))
          saved = previous.cast<Result>();
        else if (py::isinstance<py::dict>(previous))
          saved = saved_result(previous.cast<py::dict>(), true);
        else
          throw py::type_error("Supply a PhaseDiagramResult or its full saved "
                               "JSON dictionary.");
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
      R"doc(Continue saved reaction lines and optionally refine their equilibrium sampling.

Supply the original bulk, candidate order and phase-instance settings. previous
is a native result or full saved dictionary; omitted settings reuse its settings.
Recovery can extend incomplete reactions, trim overshoots and reconcile duplicate
records. With resolution supplied, closed fields still lacking trusted labels
can receive additional interior equilibrium samples.

resolution gives point counts in calculation-axis order: 101 means 100 divisions.
It limits spacing along reaction lines, rather than adding an assemblage seed
grid. reciprocal_volume requests uniform density spacing. region gives a window
in native diagram units; counts then subdivide its spans. Suspect arcs can be
retraced during recovery. The complete diagram is retained, and unmet spacing
or unresolved connections/labels remain in the returned diagnostics.
)doc");
  m.def("water_fluid", py::overload_cast<>(&minerals::water_fluid));
}
} // namespace burnman::python
