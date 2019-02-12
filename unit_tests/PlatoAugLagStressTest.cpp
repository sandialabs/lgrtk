/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "plato/Plato_Diagnostics.hpp"
#include "plato/Plato_AugLagStressCriterion.hpp"

namespace AugLagStressTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CellDensity)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::ScalarMultiVector tCellControls("Control Workset", tNumCells, tNumNodesPerCell);
    auto tHostCellControls = Kokkos::create_mirror(tCellControls);
    tHostCellControls(0, 0) = 1.00;
    tHostCellControls(1, 0) = 0.93;
    tHostCellControls(0, 1) = 0.90;
    tHostCellControls(1, 1) = 1.00;
    tHostCellControls(0, 2) = 0.95;
    tHostCellControls(1, 2) = 0.89;
    tHostCellControls(0, 3) = 0.89;
    tHostCellControls(1, 3) = 0.91;
    Kokkos::deep_copy(tCellControls, tHostCellControls);

    Plato::ScalarVector tCellDensity("Cell Density", tNumCells);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    { tCellDensity(tCellOrdinal) = Plato::cell_density<tNumNodesPerCell>(tCellOrdinal, tCellControls); }, "Test cell density inline function");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.935, 0.9325};
    auto tHostCellDensity = Kokkos::create_mirror(tCellDensity);
    Kokkos::deep_copy(tHostCellDensity, tCellDensity);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellDensity(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.557692;
    tHostCauchyStress(0, 1) = 1.557692;
    tHostCauchyStress(1, 1) = 1.557692;
    tHostCauchyStress(0, 2) = 1.096154;
    tHostCauchyStress(1, 2) = 0.634615;
    tHostCauchyStress(0, 3) = 0.461538;
    tHostCauchyStress(1, 3) = 0.230769;
    tHostCauchyStress(0, 4) = 0.230769;
    tHostCauchyStress(1, 4) = 0.230769;
    tHostCauchyStress(0, 5) = 0.461538;
    tHostCauchyStress(1, 5) = 0.692308;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
            {
                tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.965377, 1.315587};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.457692;
    tHostCauchyStress(0, 1) = 1.557692;
    tHostCauchyStress(1, 1) = 1.557692;
    tHostCauchyStress(0, 2) = 1.096154;
    tHostCauchyStress(1, 2) = 0.634615;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
            {
                tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {2.350563425, 1.867844683};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.457692;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
            {
                tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.096154, 1.457692};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CriterionEval_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00166667, 0.0121221, 0.00166667, 0.0426097, 0.26095, 0.383705};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.702721, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterion<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FiniteDiff_CriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    Plato::AugLagStressCriterion<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FiniteDiff_CriterionGradZ_3D)
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
    Plato::AugLagStressCriterion<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FiniteDiff_CriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using StateT = Jacobian::StateScalarType;
    using ConfigT = Jacobian::ConfigScalarType;
    using ResultT = Jacobian::ResultScalarType;
    using ControlT = Jacobian::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterion<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_UpdateMultipliers1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    tMassMultipliers = tCriterion.getMassMultipliers();
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldMassMultipliers(tNumCells, 0.525);
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tHostMassMultipliers = Kokkos::create_mirror(tMassMultipliers);
    Kokkos::deep_copy(tHostMassMultipliers, tMassMultipliers);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldMassMultipliers[tIndex], tHostMassMultipliers(tIndex), tTolerance);
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_UpdateMultipliers2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    tMassMultipliers = tCriterion.getMassMultipliers();
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldMassMultipliers(tNumCells, 0.);
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
        {0.0211555, 0.0441683, 0.0943195, 0.0707151, 0.0772896, 0.054743};
    auto tHostMassMultipliers = Kokkos::create_mirror(tMassMultipliers);
    Kokkos::deep_copy(tHostMassMultipliers, tMassMultipliers);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldMassMultipliers[tIndex], tHostMassMultipliers(tIndex), tTolerance);
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

} // namespace AugLagStressTest
