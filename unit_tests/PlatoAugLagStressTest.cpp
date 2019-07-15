/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "plato/Plato_Diagnostics.hpp"
#include "plato/Plato_AugLagStressCriterion.hpp"
#include "plato/Plato_AugLagStressCriterionGeneral.hpp"
#include "plato/ScalarFunctionBase.hpp"
#include "plato/Eigenvalues.hpp"
#include "plato/Plato_AugLagStressCriterionQuadratic.hpp"
#include "plato/VonMisesLocalMeasure.hpp"
#include "plato/TensileEnergyDensityLocalMeasure.hpp"
#include "plato/WeightedSumFunction.hpp"
#include "plato/PhysicsScalarFunction.hpp"
#include "plato/MassPropertiesFunction.hpp"


namespace AugLagStressTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_Eigenvalue1D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 1;
    Plato::Eigenvalues<tSpaceDim> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0;
    tHostCauchyStrain(1, 0) = 0.5;
    tHostCauchyStrain(2, 0) = -1.0;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-8;
    std::vector<Plato::Scalar> tGold1 = {3.0};
    std::vector<Plato::Scalar> tGold2 = {0.5};
    std::vector<Plato::Scalar> tGold3 = {-1.0};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_Eigenvalue2D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    Plato::Eigenvalues<tSpaceDim> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 0.0;
    tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 1.6;
    tHostCauchyStrain(2, 0) = 0.0; tHostCauchyStrain(2, 1) = 0.0; tHostCauchyStrain(2, 2) = 1.6;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold1 = {3.0, 2.0};
    std::vector<Plato::Scalar> tGold2 = {1.16394103, -0.46394103};
    std::vector<Plato::Scalar> tGold3 = {-0.8, 0.8};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_Eigenvalue3D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::Eigenvalues<tSpaceDim> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 1.0;
    tHostCauchyStrain(0, 3) = 0.0; tHostCauchyStrain(0, 4) = 0.0; tHostCauchyStrain(0, 5) = 0.0;
    tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 0.8;
    tHostCauchyStrain(1, 3) = 1.1; tHostCauchyStrain(1, 4) = 1.5; tHostCauchyStrain(1, 5) = 0.3;
    tHostCauchyStrain(2, 0) = 1.64913808; tHostCauchyStrain(2, 1) = 0.61759347; tHostCauchyStrain(2, 2) = 0.33326845;
    tHostCauchyStrain(2, 3) = 0.65938917; tHostCauchyStrain(2, 4) = -0.1840644; tHostCauchyStrain(2, 5) = 1.55789418;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold1 = {3.0, 2.0, 1.0};
    std::vector<Plato::Scalar> tGold2 = {-0.28251642,  0.17136166,  1.61115476};
    std::vector<Plato::Scalar> tGold3 = {2.07094021, -0.07551018, 0.60456996};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvaluateVonMises)
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
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasure = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00140738, 0.00750674, 0.00140738, 0.0183732, 0.0861314, 0.122407};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.237233, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvaluateTensileEnergyDensity2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    //Plato::fill(0.0, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion.setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasure = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tCriterion.setLocalMeasureValueLimit(0.15);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {158.526064959, 4.77842781597};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(163.304492775, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvalTensileEnergyScalarFuncBase2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(1.0, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterion = 
    std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion->setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion->setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasure = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion->setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tCriterion->setLocalMeasureValueLimit(0.15);

    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFunc = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFunc->allocateValue(tCriterion);
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    auto tObjFuncVal = tWeightedSum.value(tState, tControl, 0.0);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(163.304492775, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyScalarFuncBaseGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<GradientZ>> tCriterionGradZ = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<GradientZ>>(*tMesh, tMeshSets, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>  tLocalMeasureGradZ = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFunc = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFunc->allocateValue(tCriterionResidual);
    tPhysicsScalarFunc->allocateGradientZ(tCriterionGradZ);

    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyScalarFuncBaseGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Jacobian>> tCriterionGradU = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFunc = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFunc->allocateValue(tCriterionResidual);
    tPhysicsScalarFunc->allocateGradientU(tCriterionGradU);
    
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyScalarFuncBaseGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Jacobian>> tCriterionGradU = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(*tMesh, tMeshSets, tDataMap);
    
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);
    
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFunc = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFunc->allocateValue(tCriterionResidual);
    tPhysicsScalarFunc->allocateGradientU(tCriterionGradU);
    
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvaluateTensileEnergyDensity3D)
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
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    //Plato::fill(0.0, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); }, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion.setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasure = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.0290519854871, 0.148180507407, 0.0290519854871,
                                         0.439464476224,  3.34517161484,    5.3805864727};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(9.37150704214, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterionQuadratic<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterionQuadratic<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    Plato::AugLagStressCriterionQuadratic<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    Plato::AugLagStressCriterionQuadratic<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);
    
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_UpdateMultipliers1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasure = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_UpdateMultipliers2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasure = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
        {0.041727, 0.050998, 0.122774, 0.12715, 0.130626, 0.0671748};
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
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
    std::vector<Plato::Scalar> tGold = {1.284867, 1.615385};
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

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_ComputeStructuralMass_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);
    tCriterion.setCellMaterialDensity(0.5);
    tCriterion.computeStructuralMass();

    // TEST STRUCTURAL MASS CALCULATION
    auto tStructMass = tCriterion.getMassNormalizationMultiplier();
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(0.5, tStructMass, tTolerance);
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
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00307405, 0.00917341, 0.00307405, 0.0200399, 0.0877981, 0.124073};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.247233, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradZ_2D)
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradU_2D)
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradZ_3D)
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradU_3D)
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
        {0.041727, 0.050998, 0.122774, 0.12715, 0.130626, 0.0671748};
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

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterionGeneral<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    Plato::AugLagStressCriterionGeneral<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradZ_3D)
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
    Plato::AugLagStressCriterionGeneral<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradU_3D)
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
    Plato::AugLagStressCriterionGeneral<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_computeStructuralMass)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterionGeneral<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // TEST FUNCTION
    tCriterion.computeStructuralMass();
    auto tStructuralMass = tCriterion.getMassNormalizationMultiplier();
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(1.0, tStructuralMass, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_UpdateProbelm1)
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
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_UpdateProblem2)
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
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
        {0.041727, 0.050998, 0.122774, 0.12715, 0.130626, 0.0671748};
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }

    auto tPenaltyMultiplier = tCriterion.getAugLagPenalty();
    TEST_FLOATING_EQUALITY(0.105, tPenaltyMultiplier, tTolerance);
}


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



TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, MassPlusTensileEnergy2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 1.0;
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<Residual>> tMassCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tTensileEnergyCriterion = 
    std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tTensileEnergyCriterion->setLagrangeMultipliers(tLagrangeMultipliers);
    tTensileEnergyCriterion->setAugLagPenalty(1.5);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasure = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tTensileEnergyCriterion->setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tTensileEnergyCriterion->setLocalMeasureValueLimit(0.15);

    // Append function one
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncMass = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncMass->allocateValue(tMassCriterion);
    
    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    // Append function two
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncTensileEnergy = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    const Plato::Scalar tTensileEnergyFunctionWeight = 0.5;
    tPhysicsScalarFuncTensileEnergy->allocateValue(tTensileEnergyCriterion);
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncTensileEnergy);
    tWeightedSum.appendFunctionWeight(tTensileEnergyFunctionWeight);

    auto tObjFuncVal = tWeightedSum.value(tState, tControl, 0.0);

    Plato::Scalar tMassGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim) 
                                   * tPseudoDensity * tMassFunctionWeight * tMaterialDensity;

    Plato::Scalar tTensileEnergyGoldValue = tTensileEnergyFunctionWeight * 163.304492775;

    Plato::Scalar tGoldWeightedSum = tMassGoldValue + tTensileEnergyGoldValue;

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldWeightedSum, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, MassPlusVonMises_GradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<GradientZ>> tCriterionGradZ = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<GradientZ>>(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const std::shared_ptr<Plato::VonMisesLocalMeasure<GradientZ>>  tLocalMeasureGradZ = 
        std::make_shared<Plato::VonMisesLocalMeasure<GradientZ>>(tCellStiffMatrix, "VonMises");
    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasurePODType = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // Append function one
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncVonMises = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncVonMises->allocateValue(tCriterionResidual);
    tPhysicsScalarFuncVonMises->allocateGradientZ(tCriterionGradZ);
    
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<GradientZ>> tMassCriterionGradZ = 
          std::make_shared<Plato::MassMoment<GradientZ>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterionGradZ->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradZ->setCalculationType("Mass");

    const std::shared_ptr<Plato::MassMoment<Residual>> tMassCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncMass = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncMass->allocateValue(tMassCriterion);
    tPhysicsScalarFuncMass->allocateGradientZ(tMassCriterionGradZ);
    
    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, MassPlusVonMises_GradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<GradientZ>> tCriterionGradZ = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<GradientZ>>(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const std::shared_ptr<Plato::VonMisesLocalMeasure<GradientZ>>  tLocalMeasureGradZ = 
        std::make_shared<Plato::VonMisesLocalMeasure<GradientZ>>(tCellStiffMatrix, "VonMises");
    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasurePODType = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // Append function one
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncVonMises = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncVonMises->allocateValue(tCriterionResidual);
    tPhysicsScalarFuncVonMises->allocateGradientZ(tCriterionGradZ);
    
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<GradientZ>> tMassCriterionGradZ = 
          std::make_shared<Plato::MassMoment<GradientZ>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterionGradZ->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradZ->setCalculationType("Mass");

    const std::shared_ptr<Plato::MassMoment<Residual>> tMassCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncMass = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncMass->allocateValue(tMassCriterion);
    tPhysicsScalarFuncMass->allocateGradientZ(tMassCriterionGradZ);
    
    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, MassPlusVonMises_GradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Jacobian>> tCriterionGradU = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const std::shared_ptr<Plato::VonMisesLocalMeasure<Jacobian>>  tLocalMeasureGradU = 
        std::make_shared<Plato::VonMisesLocalMeasure<Jacobian>>(tCellStiffMatrix, "VonMises");
    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasurePODType = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureGradU, tLocalMeasurePODType);

    // Append function one
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncVonMises = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncVonMises->allocateValue(tCriterionResidual);
    tPhysicsScalarFuncVonMises->allocateGradientU(tCriterionGradU);
    
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<Jacobian>> tMassCriterionGradU = 
          std::make_shared<Plato::MassMoment<Jacobian>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterionGradU->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradU->setCalculationType("Mass");

    const std::shared_ptr<Plato::MassMoment<Residual>> tMassCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncMass = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncMass->allocateValue(tMassCriterion);
    tPhysicsScalarFuncMass->allocateGradientU(tMassCriterionGradU);
    
    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, MassPlusVonMises_GradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::WeightedSumFunction<Plato::Mechanics<tSpaceDim>> tWeightedSum(*tMesh, tDataMap);

    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(*tMesh, tMeshSets, tDataMap);

    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Jacobian>> tCriterionGradU = 
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const std::shared_ptr<Plato::VonMisesLocalMeasure<Jacobian>>  tLocalMeasureGradU = 
        std::make_shared<Plato::VonMisesLocalMeasure<Jacobian>>(tCellStiffMatrix, "VonMises");
    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasurePODType = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureGradU, tLocalMeasurePODType);

    // Append function one
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncVonMises = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncVonMises->allocateValue(tCriterionResidual);
    tPhysicsScalarFuncVonMises->allocateGradientU(tCriterionGradU);
    
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::MassMoment<Jacobian>> tMassCriterionGradU = 
          std::make_shared<Plato::MassMoment<Jacobian>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterionGradU->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradU->setCalculationType("Mass");

    const std::shared_ptr<Plato::MassMoment<Residual>> tMassCriterion = 
          std::make_shared<Plato::MassMoment<Residual>>(*tMesh, tMeshSets, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const std::shared_ptr<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>> tPhysicsScalarFuncMass = 
          std::make_shared<Plato::PhysicsScalarFunction<Plato::Mechanics<tSpaceDim>>>(*tMesh, tDataMap);

    tPhysicsScalarFuncMass->allocateValue(tMassCriterion);
    tPhysicsScalarFuncMass->allocateGradientU(tMassCriterionGradU);
    
    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tWeightedSum);
}

} // namespace AugLagStressTest
