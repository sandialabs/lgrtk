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
  const Plato::Scalar ChamberRadius = 0.075; // m
  const Plato::Scalar ChamberLength = 0.65; // m
  auto cylinder = std::make_shared<Plato::Cylinder<Plato::Scalar>>(ChamberRadius, ChamberLength);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> rocketInputs;

  Plato::AlgebraicRocketModel<double> tDriver(rocketInputs, cylinder);
  tDriver.solve();
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetCylinder)
{
  const Plato::Scalar ChamberRadius = 0.075; // m
  const Plato::Scalar ChamberLength = 0.65; // m
  auto cylinder = std::make_shared<Plato::LevelSetCylinderInBox<Plato::Scalar>>(ChamberRadius, ChamberLength);

  const Plato::AlgebraicRocketInputs<Plato::Scalar> rocketInputs;

  Plato::AlgebraicRocketModel<double> tDriver(rocketInputs, cylinder);
  tDriver.solve();
}
