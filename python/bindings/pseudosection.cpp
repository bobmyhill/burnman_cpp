#include "burnman/tools/pseudosection.hpp"
#include "bindings.hpp"
#include "burnman/minerals/water.hpp"
#include <set>
namespace burnman::python {
namespace {
#define SETTINGS_OPTIONS(X)                                                    \
  X(pressure_seeds)                                                            \
  X(temperature_seeds)                                                         \
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
py::dict result_dict(const pseudosections::Result &result) {
  py::dict data;
  data["schema_version"] = 1;
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
  auto phases = [](const std::vector<pseudosections::PhaseState> &states) {
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
  };
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
      py::dict item;
#define SAVE(name) item[#name] = py::cast(point.name);
      SAVE(pressure)
      SAVE(temperature)
      SAVE(mass_balance_error)
      SAVE(minimum_affinity)
      SAVE(residual)
#undef SAVE
      item["phases"] = phases(point.phases);
      points.append(item);
    }
    record["points"] = points;
    boundaries.append(record);
  }
  for (auto &state : result.samples) {
    py::dict record;
#define SAVE(name) record[#name] = py::cast(state.name);
    SAVE(pressure)
    SAVE(temperature)
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
    record["phases"] = phases(state.phases);
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
      PROPERTY(pressure) PROPERTY(temperature) PROPERTY(success)
          PROPERTY(is_field_verification) PROPERTY(message) PROPERTY(phases)
              PROPERTY(gibbs) PROPERTY(mass_balance_error)
                  PROPERTY(minimum_affinity) PROPERTY(equilibrium_error)
                      PROPERTY(excluded_phases) PROPERTY(outside_model_domain)
#undef PROPERTY
                          ;
  py::class_<BoundaryPoint>(m, "BoundaryPoint")
#define PROPERTY(name) .def_readonly(#name, &BoundaryPoint::name)
      PROPERTY(pressure) PROPERTY(temperature) PROPERTY(phases)
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
          PROPERTY(pressure) PROPERTY(temperature) PROPERTY(zero_phases)
              PROPERTY(assemblage) PROPERTY(incident_lines)
                  PROPERTY(critical_mode)
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
#define PROPERTY(name) .def_readonly(#name, &Result::name)
          PROPERTY(pressure_range) PROPERTY(temperature_range)
              PROPERTY(phase_names) PROPERTY(samples) PROPERTY(fields)
                  PROPERTY(boundaries) PROPERTY(nodes) PROPERTY(diagnostics)
                      PROPERTY(resolved) PROPERTY(equilibrium_solves)
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
  m.def("pseudosection", &pseudosection, py::arg("composition"),
        py::arg("phases"), py::arg("pressure_range"),
        py::arg("temperature_range"), py::arg("settings") = Settings{},
        py::call_guard<py::gil_scoped_release>());
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
