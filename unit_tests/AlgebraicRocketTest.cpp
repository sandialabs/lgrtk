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
#include "plato/Plato_BuildMesh.hpp"
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

/*
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ReadData)
{
    Omega_h::Library tOmega_h_Lib(nullptr, nullptr, MPI_COMM_WORLD);
    Omega_h::Mesh tMesh(&tOmega_h_Lib);
    constexpr Plato::OrdinalType tSpatialDim = 3;
    std::string tConnInputFile("./ParcMesh/tetTT.txt");
    std::string tCoordsInputFile("./ParcMesh/tetV.txt");
    constexpr Plato::Scalar tSharpCornerAngle = Omega_h::PI / static_cast<Plato::Scalar>(4);
    Plato::build_mesh_from_text_files<tSpatialDim>(tConnInputFile, tCoordsInputFile, tSharpCornerAngle, tMesh);

    const Plato::OrdinalType tNumNodes = tMesh.nverts();
    auto tReadSignDistField = Plato::read_data("./ParcMesh/nodalInnerDistanceField.txt", tNumNodes);
    auto tSignDistField = Plato::transform(tReadSignDistField);
    const Plato::OrdinalType tNumElems = tMesh.nelems();
    auto tReadBurnRate = Plato::read_data("./ParcMesh/elementalMaterialField.txt", tNumElems);
    auto tBurnRate = Plato::transform(tReadBurnRate);

    Omega_h::vtk::Writer tWriter("parc_mesh", &tMesh, tSpatialDim);
    tMesh.add_tag(Omega_h::VERT, "LevelSet", 1 NUM_COMPONENTS, Omega_h::Reals(tSignDistField.write()));
    tMesh.add_tag(Omega_h::REGION, "BurnRate", 1 NUM_COMPONENTS, Omega_h::Reals(tBurnRate.write()));
    auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, tSpatialDim);
    tWriter.write(0.0 time, tTags);
}
*/

/*
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AlgebraicRocketLevelSetOnExternalMesh)
{
    std::string tConnInputFile("./ParcMesh/tetTT.txt");
    std::string tCoordsInputFile("./ParcMesh/tetV.txt");
    auto tGeometry = std::make_shared < Plato::LevelSetOnExternalMesh > (tCoordsInputFile, tConnInputFile);
    Plato::ProblemParams tParams;
    tGeometry->initialize(tParams);

    std::string tLevelSetFile("./ParcMesh/nodalInnerDistanceField.txt");
    tGeometry->readNodalLevelSet(tLevelSetFile);
    std::string tBurnRateFile("./ParcMesh/elementalMaterialField.txt");
    tGeometry->readElementBurnRate(tBurnRateFile);

    const Plato::AlgebraicRocketInputs tRocketInputs;
    Plato::AlgebraicRocketModel tDriver(tRocketInputs, tGeometry);
    tDriver.solve();
}
*/

} // namespace AlgebraicRocketTest
