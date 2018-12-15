/*
 * AlgebraicRocketTest.cpp
 *
 *  Created on: Oct 25, 2018
 *      Author: drnoble
 */

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/Plato_AlgebraicRocketModel.hpp"
#include "plato/Plato_Cylinder.hpp"
#include "plato/Plato_LevelSetCylinderInBox.hpp"

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketAnalyticalCylinder)
{
  const Plato::Scalar tChamberRadius = 0.075; // m
  const Plato::Scalar tChamberLength = 0.65; // m
  auto tCylinder = std::make_shared<Plato::Cylinder<Plato::Scalar>>(tChamberRadius, tChamberLength);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;

  Plato::AlgebraicRocketModel<double> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetCylinder)
{
  const Plato::Scalar tChamberRadius = 0.075; // m
  const Plato::Scalar tChamberLength = 0.65; // m
  auto tCylinder = std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>(tChamberRadius, tChamberLength);
  const Plato::AlgebraicRocketInputs<Plato::Scalar> tRocketInputs;

  Plato::AlgebraicRocketModel<double> tDriver(tRocketInputs, tCylinder);
  tDriver.solve();
}
