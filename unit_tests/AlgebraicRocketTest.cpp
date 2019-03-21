/*
 * AlgebraicRocketTest.cpp
 *
 *  Created on: Oct 25, 2018
 */

#include <iostream>
#include <fstream>

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/Plato_Cylinder.hpp"
#include "plato/Plato_RocketMocks.hpp"
#include "plato/Plato_AlgebraicRocketModel.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"
#include "plato/Plato_LevelSetOnExternalMesh.hpp"

namespace AlgebraicRocketTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketAnalyticalCylinder)
{
  const Plato::Scalar tInitialRadius = 0.075; // m
  const Plato::Scalar tMaxRadius = 0.15; // m
  const Plato::Scalar tLength = 0.65; // m
  const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
  Plato::ProblemParams tParams =
          Plato::RocketMocks::set_constant_burn_rate_problem(tMaxRadius, tLength, tInitialRadius, tRefBurnRate);

  auto tCylinder = std::make_shared<Plato::Cylinder>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs tRocketInputs;
  Plato::AlgebraicRocketModel tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetCylinder)
{
  const Plato::Scalar tInitialRadius = 0.075; // m
  const Plato::Scalar tMaxRadius = 0.15; // m
  const Plato::Scalar tLength = 0.65; // m
  const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
  Plato::ProblemParams tParams =
          Plato::RocketMocks::set_constant_burn_rate_problem(tMaxRadius, tLength, tInitialRadius, tRefBurnRate);

  auto tCylinder = std::make_shared<Plato::LevelSetCylinderInBox>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs tRocketInputs;
  assert(tRocketInputs.mNumTimeSteps == tParams.mNumTimeSteps);
  Plato::AlgebraicRocketModel tDriver(tRocketInputs, tCylinder);
  tDriver.solve();

  std::ofstream tFile;
  tFile.open("example.txt");
  auto tOutput = tDriver.getThrustProfile();
  for(size_t i = 0; i < tOutput.size(); i++)
  {
      tFile << std::setprecision(18) << tOutput[i] << "\n";
  }
  tFile.close();
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetCylinderLinearBurnRate)
{
  const Plato::Scalar tInitialRadius = 0.025; // m (1in)
  const Plato::Scalar tLength = 0.65; // m
  const Plato::Scalar tMaxRadius = 0.1524; // m (6in)
  const Plato::Scalar tMaxRefBurnRate = 0.015;  // meters/seconds
  const Plato::Scalar tMinRefBurnRate = 0.015/6.;  // meters/seconds
  const Plato::Scalar tRefBurnRateSlopeWithRadius = (tMinRefBurnRate-tMaxRefBurnRate)/(tMaxRadius-tInitialRadius);
  const Plato::Scalar tCenterBurnRate = tMaxRefBurnRate - tRefBurnRateSlopeWithRadius*tInitialRadius;
  Plato::ProblemParams tParams =
          Plato::RocketMocks::set_linear_burn_rate_problem(tMaxRadius, tLength, tInitialRadius, tCenterBurnRate, tRefBurnRateSlopeWithRadius);

  auto tCylinder = std::make_shared<Plato::LevelSetCylinderInBox>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs tRocketInputs;
  Plato::AlgebraicRocketModel tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}

//TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetOnExternalMesh)
//{
//  auto tLevelSetGeometry = std::make_shared<Plato::LevelSetOnExternalMesh>("tetWithAllFields.vtu");
//  tLevelSetGeometry->initialize();
//
//  const Plato::AlgebraicRocketInputs tRocketInputs;
//  Plato::AlgebraicRocketModel tDriver(tRocketInputs, tLevelSetGeometry);
//  tDriver.solve();
//}

} // namespace AlgebraicRocketTest
