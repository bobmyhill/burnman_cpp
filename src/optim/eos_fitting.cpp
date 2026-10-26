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

#include "burnman/optim/eos_fitting.hpp"
#include "burnman/core/material.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace burnman::optim {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Params = types::MineralParams;

template <typename Visit> void coefficients(double &value, Visit &visit) {
  visit(value);
}
template <typename Values, typename Visit>
void coefficients(Values &values, Visit &visit) {
  for (auto &value : values)
    coefficients(value, visit);
}

// Resolve Python parameter names without changing the typed parameter API.
// References are used only during each call, so replacing a coefficient array
// cannot leave the fitting model with dangling pointers.
void parameter(Params &p, std::string name,
               const std::function<void(double &)> &visit) {
  if (name == "n")
    name = "napfu";
  if (name == "Debye_0")
    name = "debye_0";
  if (name == "Kprime_prime_0" || name == "Kprime_prime_S")
    name = "Kdprime_0";
  if (name == "K_S")
    name = "K_0";
  if (name == "Kprime_S")
    name = "Kprime_0";
  if (p.equation_of_state == types::EOSType::AA) {
    if (name == "a" || name == "b" || name == "Theta" || name == "F" ||
        name == "lmda")
      name = "aa_" + name;
  }
  if (p.equation_of_state == types::EOSType::BroshCalphad) {
    if (name == "a")
      name = "brosh_a";
    if (name == "c")
      name = "brosh_c";
    if (name == "b" || name == "delta") {
      auto &first = name == "b" ? p.b_0 : p.delta_0;
      auto &second = name == "b" ? p.b_1 : p.delta_1;
      if (!first || !second)
        throw std::invalid_argument("Unset fit parameter: " + name);
      visit(*first);
      visit(*second);
      return;
    }
  }
  using Member = std::optional<double> Params::*;
  static const std::unordered_map<std::string, Member> scalars = {
#define FIELD(field) {#field, &Params::field}
      FIELD(napfu),
      FIELD(molar_mass),
      FIELD(Z),
      FIELD(P_0),
      FIELD(T_0),
      FIELD(E_0),
      FIELD(F_0),
      FIELD(V_0),
      FIELD(K_0),
      FIELD(Kprime_0),
      FIELD(Kdprime_0),
      FIELD(G_0),
      FIELD(Gprime_0),
      FIELD(Kprime_inf),
      FIELD(Gprime_inf),
      FIELD(S_0),
      FIELD(H_0),
      FIELD(Cv),
      FIELD(debye_0),
      FIELD(grueneisen_0),
      FIELD(q_0),
      FIELD(a_0),
      FIELD(dKdT_0),
      FIELD(m),
      FIELD(eta_s_0),
      FIELD(T_einstein),
      FIELD(bel_0),
      FIELD(gel),
      FIELD(grueneisen_prime),
      FIELD(grueneisen_n),
      FIELD(theta),
      FIELD(xi_0),
      FIELD(c_1),
      FIELD(c_2),
      FIELD(q_1),
      FIELD(q_2),
      FIELD(a_anh),
      FIELD(b_anh),
      FIELD(c_anh),
      FIELD(m_anh),
      FIELD(mu_anh),
      FIELD(sigma_anh),
      FIELD(Tel_0),
      FIELD(zeta_0),
      FIELD(xi),
      FIELD(eta),
      FIELD(el_V_0),
      FIELD(delta_0),
      FIELD(delta_1),
      FIELD(b_0),
      FIELD(b_1),
      FIELD(theta_0),
      FIELD(delta_Gr),
      FIELD(delta_Sr),
      FIELD(delta_Cpr),
      FIELD(cork_T),
      FIELD(cork_P),
      FIELD(Debye_n),
      FIELD(Cv_0),
      FIELD(T_min),
      FIELD(T_max),
      FIELD(P_max)
#undef FIELD
  };
  const auto found = scalars.find(name);
  if (found != scalars.end()) {
    auto &value = p.*found->second;
    if (!value)
      throw std::invalid_argument("Unset fit parameter: " + name);
    visit(*value);
    return;
  }
  if (name == "Cp") {
    if (!p.Cp)
      throw std::invalid_argument("Unset fit parameter: Cp");
    visit(p.Cp->a);
    visit(p.Cp->b);
    visit(p.Cp->c);
    visit(p.Cp->d);
    return;
  }
#define ARRAY(field)                                                           \
  if (name == #field) {                                                        \
    if (!p.field)                                                              \
      throw std::invalid_argument("Unset fit parameter: " + name);             \
    coefficients(*p.field, visit);                                             \
    return;                                                                    \
  }
  ARRAY(a);
  ARRAY(aa_a);
  ARRAY(aa_b);
  ARRAY(aa_Theta);
  ARRAY(aa_F);
  ARRAY(aa_lmda);
  ARRAY(brosh_a);
  ARRAY(brosh_c);
  ARRAY(spin_a);
  ARRAY(spin_b);
  ARRAY(dcorkdT);
  ARRAY(c_coeffs);
#undef ARRAY
  throw std::invalid_argument("Unsupported fit parameter: " + name);
}

std::string property_flag(const std::any &flag) {
  return flag.has_value() ? std::any_cast<std::string>(flag) : "V";
}
void check_point(const Vector &x) {
  if (x.size() != 3 || !x.allFinite())
    throw std::invalid_argument(
        "MineralFit requires finite P-T-property triples.");
}
} // namespace
Eigen::VectorXd default_mle_tolerances(Material &material,
                                       const std::vector<std::string> &flags) {
  static const std::unordered_set<std::string> energy_flags = {
      "H",           "energy",          "molar_internal_energy",
      "helmholtz",   "molar_helmholtz", "gibbs",
      "molar_gibbs", "enthalpy",        "molar_enthalpy"};
  material.set_state(1.e5, 300.);
  Eigen::VectorXd tolerances(static_cast<Eigen::Index>(flags.size()));
  for (Eigen::Index i = 0; i < tolerances.size(); ++i) {
    const auto &flag = flags[static_cast<std::size_t>(i)];
    if (energy_flags.count(flag))
      tolerances[i] = 1.;
    else if (flag == "P" || flag == "pressure")
      tolerances[i] = 100.;
    else
      tolerances[i] = 1.e-5 * material.get_property(flag);
  }
  if (!(tolerances.array() > 0.).all())
    throw std::invalid_argument("All MLE tolerances should be > 0.");
  return tolerances;
}

MineralFit::MineralFit(std::shared_ptr<Mineral> mineral,
                       const Matrix &observations,
                       const std::vector<Matrix> &covariances,
                       const std::vector<std::string> &property_flags,
                       const std::vector<std::string> &parameters,
                       const Vector &tolerances,
                       const std::optional<Vector> &differences,
                       const std::optional<Matrix> &parameter_bounds)
    : m(std::move(mineral)), fit_params(parameters), bounds(parameter_bounds) {
  if (!m)
    throw std::invalid_argument("MineralFit requires a mineral.");
  data = observations;
  data_covariances = covariances;
  mle_tolerances = tolerances;
  for (const auto &flag : property_flags)
    flags.emplace_back(flag);
  const Vector params = get_params();
  delta_params = differences
                     ? *differences
                     : (params.array() * 1.e-5 + 1.e-10).matrix().eval();
  validate();
}

Vector MineralFit::get_params() const {
  std::vector<double> values;
  for (const auto &name : fit_params)
    parameter(m->params, name, [&](double &value) { values.push_back(value); });
  return Eigen::Map<const Vector>(values.data(),
                                  static_cast<Eigen::Index>(values.size()));
}

void MineralFit::set_params(const Vector &values) {
  if (values.size() != get_params().size() || !values.allFinite())
    throw std::invalid_argument(
        "Fit parameters must have the expected size and be finite.");
  Vector clipped = values;
  if (bounds)
    clipped = clipped.cwiseMax(bounds->col(0)).cwiseMin(bounds->col(1));
  Eigen::Index i = 0;
  for (const auto &name : fit_params)
    parameter(m->params, name, [&](double &value) { value = clipped[i++]; });
  m->reset_cache();
}

Vector MineralFit::function(const Vector &x, const std::any &flag) {
  check_point(x);
  m->set_state(x[0], x[1]);
  return Eigen::Vector3d(x[0], x[1], m->get_property(property_flag(flag)));
}

Vector MineralFit::normal(const Vector &x, const std::any &flag) {
  check_point(x);
  const double P = x[0], T = x[1];
  const std::string name = property_flag(flag);
  double dPdp, dpdT;
  if (name == "V" || name == "H" || name == "S" || name == "gibbs") {
    m->set_state(P, T);
    const double V = m->get_molar_volume();
    const double alpha = name == "gibbs" ? 0. : m->get_thermal_expansivity();
    if (name == "V") {
      dPdp = -m->get_isothermal_bulk_modulus_reuss() / V;
      dpdT = alpha * V;
    } else if (name == "H") {
      dPdp = 1. / ((1. - T * alpha) * V);
      dpdT = m->get_molar_heat_capacity_p();
    } else if (name == "S") {
      dPdp = -1. / (alpha * V);
      dpdT = m->get_molar_heat_capacity_p() / T;
    } else {
      dPdp = 1. / V;
      dpdT = -m->get_molar_entropy();
    }
  } else {
    constexpr double dP = 1.e5, dT = 1.;
    const double high_P = function(Eigen::Vector3d(P + dP, T, 0.), flag)[2];
    const double low_P = function(Eigen::Vector3d(P - dP, T, 0.), flag)[2];
    dPdp = 2. * dP / (high_P - low_P);
    const double high_T = function(Eigen::Vector3d(P, T + dT, 0.), flag)[2];
    const double low_T = function(Eigen::Vector3d(P, T - dT, 0.), flag)[2];
    dpdT = (high_T - low_T) / (2. * dT);
  }
  return Eigen::Vector3d(-1., -dPdp * dpdT, dPdp).normalized();
}

void MineralFit::validate() const {
  if (data.cols() != 3 || flags.size() != static_cast<std::size_t>(data.rows()))
    throw std::invalid_argument(
        "MineralFit requires P-T-property triples and one flag per datum.");
  if (bounds && (bounds->rows() != get_params().size() || bounds->cols() != 2 ||
                 bounds->array().isNaN().any() ||
                 (bounds->col(0).array() > bounds->col(1).array()).any()))
    throw std::invalid_argument(
        "Bounds must have one ordered lower/upper pair per parameter.");
  NonLinearModel::validate();
}

MineralFit fit_PTp_data(std::shared_ptr<Mineral> mineral,
                        const std::vector<std::string> &fit_params,
                        const PropertyFlags &property_flags, const Matrix &data,
                        const std::vector<Matrix> &data_covariances,
                        const Vector &mle_tolerances,
                        const std::optional<Vector> &delta_params,
                        const std::optional<Matrix> &bounds,
                        const NonlinearFitOptions &options) {
  if (!mineral || data.cols() != 3 || data.rows() == 0)
    throw std::invalid_argument(
        "Require a mineral and nonempty P-T-property triples.");
  const auto *single_flag = std::get_if<std::string>(&property_flags);
  const std::vector<std::string> flags =
      single_flag ? std::vector<std::string>(
                        static_cast<std::size_t>(data.rows()), *single_flag)
                  : std::get<std::vector<std::string>>(property_flags);
  if (flags.size() != static_cast<std::size_t>(data.rows()))
    throw std::invalid_argument(
        "The number of flags must equal the number of data.");
  const Vector tolerances = mle_tolerances.size()
                                ? mle_tolerances
                                : default_mle_tolerances(*mineral, flags);
  std::vector<Matrix> covariances = data_covariances;
  if (covariances.empty()) {
    Matrix covariance = Matrix::Zero(3, 3);
    covariance(0, 0) = 1.;
    covariances.assign(static_cast<std::size_t>(data.rows()), covariance);
  }
  MineralFit model(std::move(mineral), data, covariances, flags, fit_params,
                   tolerances, delta_params, bounds);
  nonlinear_least_squares_fit(model, options);
  return model;
}

MineralFit fit_PTV_data(std::shared_ptr<Mineral> mineral,
                        const std::vector<std::string> &fit_params,
                        const Matrix &data,
                        const std::vector<Matrix> &data_covariances,
                        const std::optional<Vector> &delta_params,
                        const std::optional<Matrix> &bounds,
                        const NonlinearFitOptions &options) {
  return fit_PTp_data(std::move(mineral), fit_params, std::string("V"), data,
                      data_covariances, {}, delta_params, bounds, options);
}
} // namespace burnman::optim
