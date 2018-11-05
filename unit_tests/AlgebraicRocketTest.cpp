/*
 * AlgebraicRocketTest.cpp
 *
 *  Created on: Oct 25, 2018
 *      Author: drnoble
 */

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/Plato_AlgebraicRocketModel.hpp"

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocket)
{
    Plato::AlgebraicRocketModel<double> tDriver;
    tDriver.solve();
}
