#include "Teuchos_UnitTestHarness.hpp"
#include <Omega_h_kokkos.hpp>

#include "IdealGasFunctions.hpp"

namespace {

  /***********************************************************************
                              hyperelasticCauchyStress()
  ***********************************************************************/

  TEUCHOS_UNIT_TEST( IdealGasModel, hyperelasticCauchyStress_0 )
  {
    // Tests for when (gamma-1)=0
    const Plato::Scalar tol = 1e-12;
    const Plato::Scalar internal_energy = 1.0;
    const Plato::Scalar mass_density = 1.0;
    const Plato::Scalar gamma = 1.0;
    Plato::Scalar K_gold = 0.0;
    Plato::Scalar K_calc = IdealGasFunctions::hyperelasticCauchyStress(
                       internal_energy, mass_density, gamma);
    TEST_FLOATING_EQUALITY(K_gold, K_calc, tol);
  }


  TEUCHOS_UNIT_TEST( IdealGasModel, hyperelasticCauchyStress_1 )
  {
    // Tests for when (gamma-1)=1
    const Plato::Scalar tol = 1e-12;
    const Plato::Scalar internal_energy = 1.0;
    const Plato::Scalar mass_density = 1.0;
    const Plato::Scalar gamma = 2.0;
    Plato::Scalar K_gold = 1.0;
    Plato::Scalar K_calc = IdealGasFunctions::hyperelasticCauchyStress(
                       internal_energy, mass_density, gamma);
    TEST_FLOATING_EQUALITY(K_gold, K_calc, tol);
  }

/*
  TEUCHOS_UNIT_TEST( IdealGasModel, hyperelasticCauchyStress_negative_pressure_0 )
  {
    // Tests for when the pressure is negative !!!SHOULD FAIL!!!
    const Plato::Scalar internal_energy = 1.0;
    const Plato::Scalar mass_density = -1.0;
    const Plato::Scalar gamma = 2.0;
    TEST_THROW(IdealGasFunctions::hyperelasticCauchyStress(
                      internal_energy, mass_density, gamma),
               std::logic_error);
  }


  TEUCHOS_UNIT_TEST( IdealGasModel, hyperelasticCauchyStress_no_energy_floor )
  {
    // Tests for when the pressure is negative !!!SHOULD FAIL!!!
    const Plato::Scalar internal_energy = -1.0;
    const Plato::Scalar mass_density = 1.0;
    const Plato::Scalar gamma = 2.0;
    TEST_THROW(IdealGasFunctions::hyperelasticCauchyStress(
                      internal_energy, mass_density, gamma),
               std::logic_error);
  }
*/


} // namespace


namespace {

  /***********************************************************************
                              waveModuli()
  ***********************************************************************/

  TEUCHOS_UNIT_TEST( IdealGasModel, waveModuli_0 )
  {
    // Tests for when (gamma-1)=0
    const Plato::Scalar tol = 1e-12;
    const Plato::Scalar internal_energy = 1.0;
    const Plato::Scalar mass_density = 1.0;
    const Plato::Scalar gamma = 1.0;
    Plato::Scalar K_gold = 0.0;
    Plato::Scalar K_calc = IdealGasFunctions::waveModuli(internal_energy, mass_density, gamma);
    TEST_FLOATING_EQUALITY(K_gold, K_calc, tol);
  }

  TEUCHOS_UNIT_TEST( IdealGasModel, waveModuli_1 )
  {
    // Tests for when (gamma-1)=1
    const Plato::Scalar tol = 1e-12;
    const Plato::Scalar internal_energy = 1.0;
    const Plato::Scalar mass_density = 1.0;
    const Plato::Scalar gamma = 2.0;
    Plato::Scalar K_gold = 2.0;
    Plato::Scalar K_calc = IdealGasFunctions::waveModuli(internal_energy, mass_density, gamma);
    TEST_FLOATING_EQUALITY(K_gold, K_calc, tol);
  }

/*
  TEUCHOS_UNIT_TEST( IdealGasModel, waveModuli_negative_bulkmod )
  {
    // Tests for when the bulk modulus is negative !!!SHOULD FAIL!!!
    const Plato::Scalar tol = 1e-12;
    const Plato::Scalar internal_energy = 1.0;
    const Plato::Scalar mass_density = -1.0;
    const Plato::Scalar gamma = 2.0;
    TEST_THROW(IdealGasFunctions::waveModuli(internal_energy, mass_density, gamma),
               std::logic_error);
  }
*/

  TEUCHOS_UNIT_TEST( IdealGasModel, waveModuli_no_energy_floor )
  {
    // Tests for when the internal energy is negative and that it's being
    // picked up by the internal energy floor.
    const Plato::Scalar tol = 1e-12;
    const Plato::Scalar internal_energy = -1.0;
    const Plato::Scalar mass_density = 1.0;
    const Plato::Scalar gamma = 2.0;
    Plato::Scalar K_gold = 0.0;
    Plato::Scalar K_calc = IdealGasFunctions::waveModuli(internal_energy, mass_density, gamma);
//    TEST_THROW(IdealGasFunctions::waveModuli(internal_energy, mass_density, gamma),
//               std::logic_error);
    TEST_ASSERT(K_calc - K_gold < tol);
  }

} // namespace
