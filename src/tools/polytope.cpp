/* Native cddlib/GMP polytope tools, GPL v3 or later. */
#include "burnman/tools/polytope.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

// Keep cddlib's arithmetic macros private to this translation unit.
#define GMPRATIONAL
// cdd.h needs the set_type declaration from setoper.h first.
// clang-format off
#if __has_include(<cddlib/setoper.h>)
#include <cddlib/setoper.h>
#include <cddlib/cdd.h>
#else
#include <setoper.h>
#include <cdd.h>
#endif
// clang-format on

namespace burnman::polytope {
namespace {
struct CddRuntime {
  std::mutex mutex;
  CddRuntime() { dd_set_global_constants(); }
  ~CddRuntime() { dd_free_global_constants(); }
};
CddRuntime &runtime() {
  static CddRuntime instance;
  return instance;
}
using MatrixHandle = std::unique_ptr<dd_MatrixType, decltype(&dd_FreeMatrix)>;
using PolyhedronHandle =
    std::unique_ptr<dd_PolyhedraType, decltype(&dd_FreePolyhedra)>;

// Continued fractions recover e.g. 1/3 and decimal chemical proportions.
// Exact binary conversion alone can make redundant mass balances inconsistent.
void set_rational(mytype target, double value, double tolerance) {
  mpq_set_d(target, value);
  if (tolerance == 0.0)
    return;
  const double sign = value < 0.0 ? -1.0 : 1.0;
  double remainder = std::abs(value);
  long numerator0 = 0, numerator1 = 1, denominator0 = 1, denominator1 = 0;
  constexpr long max_denominator = 1000000000L;
  for (int iteration = 0; iteration < 64; ++iteration) {
    double integral = std::floor(remainder);
    if (!std::isfinite(integral) ||
        integral > static_cast<double>(std::numeric_limits<long>::max() / 2))
      return;
    long coefficient = static_cast<long>(integral);
    if ((numerator1 &&
         coefficient >
             (std::numeric_limits<long>::max() - numerator0) / numerator1) ||
        (denominator1 &&
         coefficient > (max_denominator - denominator0) / denominator1))
      return;
    long numerator = coefficient * numerator1 + numerator0;
    long denominator = coefficient * denominator1 + denominator0;
    if (!denominator || denominator > max_denominator)
      return;
    if (std::abs(sign * static_cast<double>(numerator) /
                     static_cast<double>(denominator) -
                 value) <= tolerance) {
      mpq_set_si(target, sign < 0 ? -numerator : numerator,
                 static_cast<unsigned long>(denominator));
      mpq_canonicalize(target);
      return;
    }
    numerator0 = numerator1;
    numerator1 = numerator;
    denominator0 = denominator1;
    denominator1 = denominator;
    double fractional = remainder - integral;
    if (fractional == 0.0)
      return;
    remainder = 1.0 / fractional;
  }
}
Eigen::MatrixXd rows_to_matrix(const std::vector<Eigen::VectorXd> &rows,
                               Eigen::Index cols) {
  Eigen::MatrixXd result(static_cast<Eigen::Index>(rows.size()), cols);
  for (std::size_t i = 0; i < rows.size(); ++i)
    result.row(static_cast<Eigen::Index>(i)) = rows[i];
  return result;
}
} // namespace

MaterialPolytope::MaterialPolytope(const Eigen::MatrixXd &equalities,
                                   const Eigen::MatrixXd &inequalities,
                                   double rational_tolerance)
    : equalities_(equalities), inequalities_(inequalities) {
  if (!std::isfinite(rational_tolerance) || rational_tolerance < 0.0 ||
      rational_tolerance > 1.e-6) {
    throw std::invalid_argument(
        "Rational tolerance must be finite and between zero and 1e-6.");
  }
  Eigen::Index cols =
      equalities.rows() ? equalities.cols() : inequalities.cols();
  if (cols < 2 || (equalities.rows() && equalities.cols() != cols) ||
      (inequalities.rows() && inequalities.cols() != cols) ||
      !equalities.allFinite() || !inequalities.allFinite() ||
      equalities.rows() + inequalities.rows() == 0) {
    throw std::invalid_argument(
        "Polytope constraints require finite [constant, coefficients...] rows "
        "of matching width.");
  }
  // cddlib has global arithmetic state; serialize only the cddlib work.
  auto &cdd = runtime();
  std::lock_guard<std::mutex> lock(cdd.mutex);
  MatrixHandle input(
      dd_CreateMatrix(equalities.rows() + inequalities.rows(), cols),
      dd_FreeMatrix);
  if (!input)
    throw std::bad_alloc();
  input->representation = dd_Inequality;
  input->numbtype = dd_Rational;
  for (Eigen::Index row = 0; row < input->rowsize; ++row) {
    bool equality = row < equalities.rows();
    if (equality)
      set_addelem(input->linset, row + 1);
    for (Eigen::Index col = 0; col < cols; ++col) {
      set_rational(input->matrix[row][col],
                   equality ? equalities(row, col)
                            : inequalities(row - equalities.rows(), col),
                   rational_tolerance);
    }
  }
  dd_ErrorType error = dd_NoError;
  PolyhedronHandle polyhedron(dd_DDMatrix2Poly(input.get(), &error),
                              dd_FreePolyhedra);
  if (error != dd_NoError || !polyhedron) {
    throw std::runtime_error("cddlib vertex enumeration failed (error " +
                             std::to_string(error) + ").");
  }
  MatrixHandle generators(dd_CopyGenerators(polyhedron.get()), dd_FreeMatrix);
  if (!generators)
    throw std::runtime_error("cddlib returned no generator matrix.");
  std::vector<Eigen::VectorXd> vertices, rays, lines;
  for (Eigen::Index row = 0; row < generators->rowsize; ++row) {
    Eigen::VectorXd point(cols - 1);
    double scale = dd_get_d(generators->matrix[row][0]);
    for (Eigen::Index col = 1; col < cols; ++col) {
      point[col - 1] = dd_get_d(generators->matrix[row][col]);
    }
    if (set_member(row + 1, generators->linset))
      lines.push_back(point);
    else if (scale == 0.0)
      rays.push_back(point);
    else
      vertices.push_back(point / scale);
  }
  vertices_ = rows_to_matrix(vertices, cols - 1);
  rays_ = rows_to_matrix(rays, cols - 1);
  lineality_ = rows_to_matrix(lines, cols - 1);
  occupancies_ = vertices_;
  empty_ = polyhedron->IsEmpty == dd_TRUE;
}

void MaterialPolytope::map_to_site_occupancies(
    const Eigen::MatrixXd &occupancies) {
  if (occupancies.rows() != vertices_.cols() || !occupancies.allFinite()) {
    throw std::invalid_argument(
        "Occupancy mapping must match the polytope coordinates.");
  }
  occupancies_ = vertices_ * occupancies;
}

MaterialPolytope
solution_polytope_from_endmember_occupancies(const Eigen::MatrixXd &occupancies,
                                             double rational_tolerance) {
  if (!occupancies.rows() || !occupancies.cols() || !occupancies.allFinite() ||
      occupancies.minCoeff() < -1.e-12) {
    throw std::invalid_argument(
        "Endmember occupancies must be finite, nonnegative and nonempty.");
  }
  Eigen::MatrixXd equalities = Eigen::MatrixXd::Ones(1, occupancies.rows() + 1);
  equalities(0, 0) = -1.0;
  Eigen::MatrixXd inequalities =
      Eigen::MatrixXd::Zero(occupancies.cols(), occupancies.rows() + 1);
  inequalities.rightCols(occupancies.rows()) = occupancies.transpose();
  MaterialPolytope polytope(equalities, inequalities, rational_tolerance);
  if (!polytope.is_bounded()) {
    throw std::invalid_argument(
        "Endmembers must give a bounded, independent site-occupancy basis.");
  }
  polytope.map_to_site_occupancies(occupancies);
  return polytope;
}

MaterialPolytope composite_polytope_at_constrained_composition(
    const Assemblage &composite, const types::FormulaMap &composition,
    double rational_tolerance) {
  if (!composite.get_n_phases())
    throw std::invalid_argument("An assemblage must contain phases.");
  auto elements = composite.get_elements();
  for (const auto &[element, amount] : composition) {
    if (!std::isfinite(amount) || amount < 0.0) {
      throw std::invalid_argument(
          "Bulk composition must contain finite nonnegative amounts.");
    }
    if (amount != 0.0 && std::find(elements.begin(), elements.end(), element) ==
                             elements.end()) {
      throw std::invalid_argument(
          "Bulk composition contains an element absent from the assemblage: " +
          element);
    }
  }
  Eigen::Index n = composite.get_n_endmembers(), rows = 0;
  for (Eigen::Index i = 0; i < composite.get_n_phases(); ++i) {
    auto phase = composite.get_phase(static_cast<std::size_t>(i));
    if (auto solution = std::dynamic_pointer_cast<Solution>(phase))
      rows += solution->get_endmember_occupancies().cols();
    else if (std::dynamic_pointer_cast<Mineral>(phase))
      ++rows;
    else
      throw std::invalid_argument("Polytope tools require a flat assemblage of "
                                  "minerals and solutions.");
  }
  Eigen::MatrixXd equalities =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(elements.size()), n + 1);
  equalities.rightCols(n) = composite.get_stoichiometric_matrix().transpose();
  for (std::size_t i = 0; i < elements.size(); ++i) {
    auto amount = composition.find(elements[i]);
    if (amount != composition.end())
      equalities(static_cast<Eigen::Index>(i), 0) = -amount->second;
  }
  Eigen::MatrixXd inequalities = Eigen::MatrixXd::Zero(rows, n + 1);
  Eigen::Index offset = 0, row = 0;
  for (Eigen::Index i = 0; i < composite.get_n_phases(); ++i) {
    auto phase = composite.get_phase(static_cast<std::size_t>(i));
    if (auto solution = std::dynamic_pointer_cast<Solution>(phase)) {
      auto occupancies = solution->get_endmember_occupancies();
      inequalities.block(row, offset + 1, occupancies.cols(),
                         occupancies.rows()) = occupancies.matrix().transpose();
      row += occupancies.cols();
      offset += occupancies.rows();
    } else {
      inequalities(row++, ++offset) = 1.0;
    }
  }
  return MaterialPolytope(equalities, inequalities, rational_tolerance);
}
GibbsLPResult gibbs_linear_program(const Eigen::MatrixXd &a,
                                   const Eigen::VectorXd &g,
                                   const Eigen::VectorXd &b) {
  if (a.rows() != g.size() || a.cols() != b.size() || !a.rows() || !a.cols() ||
      !a.allFinite() || !g.allFinite() || !b.allFinite())
    throw std::invalid_argument(
        "Gibbs LP requires matching finite stoichiometry, energies and bulk.");
  auto &cdd = runtime();
  std::lock_guard<std::mutex> lock(cdd.mutex);
  MatrixHandle input(dd_CreateMatrix(a.rows(), a.cols() + 1), dd_FreeMatrix);
  input->representation = dd_Inequality;
  input->numbtype = dd_Rational;
  input->objective = dd_LPmax;
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    set_rational(input->matrix[i][0], g[i], 0.);
    for (Eigen::Index j = 0; j < a.cols(); ++j)
      set_rational(input->matrix[i][j + 1], -a(i, j), 1.e-12);
  }
  for (Eigen::Index j = 0; j < b.size(); ++j)
    set_rational(input->rowvec[j + 1], b[j], 1.e-12);
  dd_ErrorType error = dd_NoError;
  std::unique_ptr<dd_LPType, decltype(&dd_FreeLPData)> lp(
      dd_Matrix2LP(input.get(), &error), dd_FreeLPData);
  if (!lp || error != dd_NoError)
    throw std::runtime_error("cddlib Gibbs LP construction failed.");
  dd_LPSolve(lp.get(), dd_DualSimplex, &error);
  if (error == dd_NoError &&
      (lp->LPS == dd_DualInconsistent || lp->LPS == dd_StrucDualInconsistent))
    throw InfeasibleBulk("Bulk composition cannot be represented by the "
                         "available phase compositions.");
  if (error != dd_NoError || lp->LPS != dd_Optimal)
    throw std::runtime_error("Bulk composition is infeasible, or Gibbs LP is "
                             "unbounded (cddlib status " +
                             std::to_string(lp->LPS) + ").");
  GibbsLPResult result;
  result.gibbs = dd_get_d(lp->optvalue);
  result.chemical_potentials.resize(a.cols());
  result.amounts = Eigen::VectorXd::Zero(a.rows());
  for (Eigen::Index j = 0; j < a.cols(); ++j)
    result.chemical_potentials[j] = dd_get_d(lp->sol[j + 1]);
  for (Eigen::Index j = 1; j <= a.cols(); ++j) {
    auto row = lp->nbindex[j + 1];
    if (row > 0 && row <= a.rows())
      result.amounts[row - 1] = dd_get_d(lp->dsol[j]);
  }
  // Fail explicitly if an unexpected cddlib dual convention is encountered.
  if ((a.transpose() * result.amounts - b).norm() > 1.e-8 * (1 + b.norm()) ||
      result.amounts.minCoeff() < -1.e-10)
    throw std::runtime_error("cddlib Gibbs LP dual mass balance failed.");
  return result;
}
} // namespace burnman::polytope
