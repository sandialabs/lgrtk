/*
 * AlgebraicRocketTest.cpp
 *
 *  Created on: Oct 25, 2018
 */

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/Plato_Cylinder.hpp"
#include "plato/Plato_AlgebraicRocketModel.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"

Plato::ProblemParams setupConstantBurnRateCylinder(Plato::Scalar aRadius, Plato::Scalar aLength, Plato::Scalar aRefBurnRate)
{
	Plato::ProblemParams tParams;
	tParams.mGeometry.push_back(aRadius);
	tParams.mGeometry.push_back(aLength);
	tParams.mRefBurnRate.push_back(aRefBurnRate);
	return tParams;
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketAnalyticalCylinder)
{
  const Plato::Scalar tRadius = 0.075; // m
  const Plato::Scalar tLength = 0.65; // m
  const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
  Plato::ProblemParams tParams = setupConstantBurnRateCylinder(tRadius, tLength, tRefBurnRate);

  auto tCylinder = std::make_shared<Plato::Cylinder<Plato::Scalar>>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;
  Plato::AlgebraicRocketModel<double> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetCylinder)
{
  const Plato::Scalar tRadius = 0.075; // m
  const Plato::Scalar tLength = 0.65; // m
  const Plato::Scalar tRefBurnRate = 0.005;  // meters/seconds
  Plato::ProblemParams tParams = setupConstantBurnRateCylinder(tRadius, tLength, tRefBurnRate);

  auto tCylinder = std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>();
  tCylinder->initialize(tParams);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;
  Plato::AlgebraicRocketModel<Plato::Scalar> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}
