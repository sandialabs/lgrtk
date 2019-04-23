/*
 * PlatoCenterGravityTest.cpp
 *
 *  Created on: Apr 15, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "PlatoTestHelpers.hpp"
#include "plato/Plato_Diagnostics.hpp"
#include "plato/Plato_CenterGravityCriterion.hpp"

namespace CenterGravityTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeSimplexCentroid2D)
{
    // CREATE MESH FOR UNIT TEST
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // CREATE CONFIGURATION WORKSET
    auto tNumCells = tMesh->nelems();
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // TEST FUNCTION
    Plato::ScalarMultiVectorT<Plato::Scalar> tCellCentroids("Cell Centroids", tNumCells, tSpaceDim);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::compute_simplex_centroid<tSpaceDim>(aCellOrdinal, tConfigWS, tCellCentroids);
    },"Compute Centroids");

    // TEST OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {2.0/3.0, 1.0/3.0}, {1.0/3.0, 2.0/3.0} };
    auto tHostCellCentroids = Kokkos::create_mirror(tCellCentroids);
    Kokkos::deep_copy(tHostCellCentroids, tCellCentroids);
    for(Plato::OrdinalType tCellOrdinal = 0; tCellOrdinal < tNumCells; tCellOrdinal++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCellOrdinal][tDim], tHostCellCentroids(tCellOrdinal, tDim), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeSimplexCentroid3D)
{
    // CREATE MESH FOR UNIT TEST
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // CREATE CONFIGURATION WORKSET
    auto tNumCells = tMesh->nelems();
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // TEST FUNCTION
    Plato::ScalarMultiVectorT<Plato::Scalar> tCellCentroids("Cell Centroids", tNumCells, tSpaceDim);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::compute_simplex_centroid<tSpaceDim>(aCellOrdinal, tConfigWS, tCellCentroids);
    },"Compute Centroids");

    // TEST OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostCellCentroids = Kokkos::create_mirror(tCellCentroids);
    Kokkos::deep_copy(tHostCellCentroids, tCellCentroids);
    std::vector<std::vector<Plato::Scalar>> tGold =
        { {0.5, 0.75, 0.25}, {0.25, 0.75, 0.5}, {0.25, 0.5, 0.75}, {0.5, 0.25, 0.75}, {0.75, 0.25, 0.5}, {0.75, 0.5, 0.25} };
    for(Plato::OrdinalType tCellOrdinal = 0; tCellOrdinal < tNumCells; tCellOrdinal++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCellOrdinal][tDim], tHostCellCentroids(tCellOrdinal, tDim), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, StructuralMass_operator)
{
    // CREATE MESH FOR UNIT TEST
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    constexpr Plato::Scalar tDensity = 2.5;
    Plato::StructuralMass<tSpaceDim> tComputeStructuralMass(tDensity);

    // CREATE CONFIGURATION WORKSET
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // CREATE CONTROL WORKSET
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(0.25, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // CALL FUNCTION BEING TESTED
    Plato::Scalar tStructuralMass = 0;
    tComputeStructuralMass(tNumCells, tControlWS, tConfigWS, tStructuralMass);

    // TEST OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(0.625 /* gold */, tStructuralMass, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CenterGravity_ComputeInitialStructuralMass)
{
    // CREATE MESH FOR UNIT TEST
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::CenterGravityCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // CALL FUNCTION BEING TESTED
    tCriterion.computeInitialStructuralMass();

    // TEST OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tMassNormalizationMultiplier = tCriterion.getMassNormalizationMultiplier();
    TEST_FLOATING_EQUALITY(1.0 /* gold */, tMassNormalizationMultiplier, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CenterGravity_UpdateProblem)
{
    // CREATE MESH FOR UNIT TEST
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::CenterGravityCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // CREATE CONFIGURATION WORKSET
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // CREATE CONTROL WORKSET
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    Plato::fill(0.5, tState);
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // 1. CALL FUNCTION BEING TESTED
    Plato::ScalarVector tTarget("target cg", tSpaceDim);
    Kokkos::deep_copy(tTarget, 0.25);
    tCriterion.setTargetCenterGravity(tTarget);
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);

    // TEST PENALTY OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tAugLagPenalty = tCriterion.getAugLagPenalty();
    TEST_FLOATING_EQUALITY(0.375 /* gold */, tAugLagPenalty, tTolerance);

    // TEST LAGRANGE MULTIPLIERS
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
    {
        TEST_FLOATING_EQUALITY(0., tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CenterGravity_FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    using StateT = typename GradientZ::StateScalarType;
    using ConfigT = typename GradientZ::ConfigScalarType;
    using ResultT = typename GradientZ::ResultScalarType;
    using ControlT = typename GradientZ::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::CenterGravityCriterion<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::OrdinalType tSuperscriptLowerBound = -1;
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion, tSuperscriptLowerBound);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CenterGravity_FiniteDiff_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    using StateT = typename GradientZ::StateScalarType;
    using ConfigT = typename GradientZ::ConfigScalarType;
    using ResultT = typename GradientZ::ResultScalarType;
    using ControlT = typename GradientZ::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::CenterGravityCriterion<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::OrdinalType tSuperscriptLowerBound = -1;
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion, tSuperscriptLowerBound);
}

} // namespace CenterGravityTest
