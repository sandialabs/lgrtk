/*
 * AlgebraicRocketTest.cpp
 *
 *  Created on: Oct 25, 2018
 */

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
  Plato::ProblemParams tParams = Plato::RocketMocks::setupConstantBurnRateCylinder(tMaxRadius, tLength, tInitialRadius, tRefBurnRate);

  auto tCylinder = std::make_shared<Plato::Cylinder<Plato::Scalar>>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;
  Plato::AlgebraicRocketModel<double> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetCylinder)
{
  const Plato::Scalar tInitialRadius = 0.075; // m
  const Plato::Scalar tMaxRadius = 0.15; // m
  const Plato::Scalar tLength = 0.65; // m
  const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
  Plato::ProblemParams tParams = Plato::RocketMocks::setupConstantBurnRateCylinder(tMaxRadius, tLength, tInitialRadius, tRefBurnRate);

  auto tCylinder = std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;
  Plato::AlgebraicRocketModel<Plato::Scalar> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
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
  Plato::ProblemParams tParams = Plato::RocketMocks::setupLinearBurnRateCylinder(tMaxRadius, tLength, tInitialRadius, tCenterBurnRate, tRefBurnRateSlopeWithRadius);

  auto tCylinder = std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;
  Plato::AlgebraicRocketModel<Plato::Scalar> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}

//TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetOnExternalMesh)
//{
//  auto tLevelSetGeometry = std::make_shared<Plato::LevelSetOnExternalMesh<Plato::Scalar>>("tetWithAllFields.vtu");
//  tLevelSetGeometry->initialize();
//
//  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;
//  Plato::AlgebraicRocketModel<Plato::Scalar> tDriver(tRocketInputs, tLevelSetGeometry);
//  tDriver.solve();
//}

} // namespace AlgebraicRocketTest
