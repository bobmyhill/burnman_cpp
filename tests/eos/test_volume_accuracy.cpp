#include "burnman/eos/make_eos.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE("Volume solves scale accuracy with molar volume", "[eos][roots]") {
  auto type = GENERATE(types::EOSType::BM3, types::EOSType::Vinet,
                       types::EOSType::MGD3);
  auto reference_volume = GENERATE(1.124e-6, 1.124e-5, 9.0e-4);
  auto temperature = GENERATE(300.0, 1000.0);
  auto ratio = GENERATE(0.91, 0.67);
  types::MineralParams params;
  params.V_0 = reference_volume;
  params.K_0 = 161.0e9;
  params.Kprime_0 = 3.8;
  params.molar_mass = 0.0403;
  params.napfu = 2;
  params.debye_0 = 773.0;
  params.grueneisen_0 = 1.5;
  params.q_0 = 1.5;
  auto eos = eos::make_eos(type);
  eos->validate_parameters(params);
  const double target_volume = ratio * reference_volume;
  const double pressure =
      eos->compute_pressure(temperature, target_volume, params);
  const double solved_volume =
      eos->compute_volume(pressure, temperature, params);
  CAPTURE(type, reference_volume, temperature, ratio);
  CHECK_THAT(solved_volume, WithinRel(target_volume, 2.0e-13));
  CHECK_THAT(eos->compute_pressure(temperature, solved_volume, params),
             WithinRel(pressure, 2.0e-13));
}
