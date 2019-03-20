/*
 * StructuralDynamicsTest.cpp
 *
 *  Created on: Mar 2, 2018
 **/

#include <memory>
#include <cstdlib>

#include <iostream>
#include <fstream>

#include "PlatoTestHelpers.hpp"

#include "plato/Simp.hpp"
#include "plato/ExpVolume.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/ScalarFunction.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/DynamicCompliance.hpp"
#include "plato/StructuralDynamics.hpp"
#include "plato/HeavisideProjection.hpp"
#include "plato/ComplexRayleighDamping.hpp"
#include "plato/FrequencyResponseMisfit.hpp"
#include "plato/StructuralDynamicsOutput.hpp"
#include "plato/StructuralDynamicsProblem.hpp"
#include "plato/StructuralDynamicsResidual.hpp"
#include "plato/HyperbolicTangentProjection.hpp"
#include "plato/AdjointComplexRayleighDamping.hpp"
#include "plato/ComputeFrequencyResponseMisfit.hpp"
#include "plato/AdjointStructuralDynamicsResidual.hpp"

#include "Teuchos_UnitTestHarness.hpp"

/*#include <amgx_c.h>
#include <amgx_eig_c.h>

namespace Plato
{

namespace eigen
{



}

}*/

namespace PlatoUnitTests
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, LinearTetCubRuleDegreeOne_3D)
{
    const Plato::OrdinalType tSpatialDim = 3;
    Plato::LinearTetCubRuleDegreeOne<tSpatialDim> tTetCubRule;

    // ******************** TEST WEIGHT FUNCTION ********************
    Plato::Scalar tCubWeight = tTetCubRule.getCubWeight();
    Plato::Scalar tGoldScalar = 1.0 / 6.0;
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tCubWeight, tGoldScalar, tTolerance); 
    
    // ******************** TEST NUM CUBPOINTS FUNCTION ********************
    Plato::OrdinalType tNumCubPoints = tTetCubRule.getNumCubPoints();
    Plato::OrdinalType tGoldOrdinal = 1;
    TEST_EQUALITY(tGoldOrdinal, tNumCubPoints);

    // ******************** TEST CUBPOINTS COORDS FUNCTION ********************
    auto tCubPointsCoords = tTetCubRule.getCubPointsCoords();
    auto tHostCubPointsCoords = Kokkos::create_mirror(tCubPointsCoords);
    Kokkos::deep_copy(tHostCubPointsCoords, tCubPointsCoords);
    tGoldScalar = 0.25;
    TEST_FLOATING_EQUALITY(tHostCubPointsCoords(0), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostCubPointsCoords(1), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostCubPointsCoords(2), tGoldScalar, tTolerance);

    // ******************** TEST GET BASIS FUNCTIONS ********************
    auto tBasisFunctions = tTetCubRule.getBasisFunctions();
    auto tHostBasisFunctions = Kokkos::create_mirror(tBasisFunctions);
    Kokkos::deep_copy(tHostBasisFunctions, tBasisFunctions);
    tGoldScalar = 0.25;
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(0), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(1), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(2), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(3), tGoldScalar, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, LinearTetCubRuleDegreeOne_2D)
{
    const Plato::OrdinalType tSpatialDim = 2;
    Plato::LinearTetCubRuleDegreeOne<tSpatialDim> tTetCubRule;

    // ******************** TEST WEIGHT FUNCTION ********************
    Plato::Scalar tCubWeight = tTetCubRule.getCubWeight();
    Plato::Scalar tGoldScalar = 1.0 / 2.0;
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tCubWeight, tGoldScalar, tTolerance);
                     
    // ******************** TEST CUBPOINTS COORDS FUNCTION ********************
    const Plato::ScalarVector tCubPointsCoords = tTetCubRule.getCubPointsCoords();
    auto tHostCubPointsCoords = Kokkos::create_mirror(tCubPointsCoords);
    Kokkos::deep_copy(tHostCubPointsCoords, tCubPointsCoords);
    
    tGoldScalar = 1.0/3.0;
    TEST_FLOATING_EQUALITY(tHostCubPointsCoords(0), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostCubPointsCoords(1), tGoldScalar, tTolerance);

    // ******************** TEST GET BASIS FUNCTIONS ********************
    auto tBasisFunctions = tTetCubRule.getBasisFunctions();
    auto tHostBasisFunctions = Kokkos::create_mirror(tBasisFunctions);
    Kokkos::deep_copy(tHostBasisFunctions, tBasisFunctions);

    tGoldScalar = 1.0/3.0;
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(0), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(1), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(2), tGoldScalar, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, LinearTetCubRuleDegreeOne_1D)
{   
    const Plato::OrdinalType tSpatialDim = 1;
    Plato::LinearTetCubRuleDegreeOne<tSpatialDim> tTetCubRule;
    
    // ******************** TEST WEIGHT FUNCTION ********************
    Plato::Scalar tCubWeight = tTetCubRule.getCubWeight();
    Plato::Scalar tGoldScalar = 1.0;
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tCubWeight, tGoldScalar, tTolerance);
                     
    // ******************** TEST CUBPOINTS COORDS FUNCTION ********************
    const Plato::ScalarVector tCubPointsCoords = tTetCubRule.getCubPointsCoords();
    Plato::OrdinalType tGoldOrdinal = 1;
    TEST_EQUALITY(tGoldOrdinal, tCubPointsCoords.size());
    
    auto tHostCubPointsCoords = Kokkos::create_mirror(tCubPointsCoords);
    Kokkos::deep_copy(tHostCubPointsCoords, tCubPointsCoords);
    tGoldScalar = 1.0/2.0;    
    TEST_FLOATING_EQUALITY(tHostCubPointsCoords(0), tGoldScalar, tTolerance);

    // ******************** TEST GET BASIS FUNCTIONS ********************
    auto tBasisFunctions = tTetCubRule.getBasisFunctions();
    tGoldOrdinal = 2;
    TEST_EQUALITY(tGoldOrdinal, tBasisFunctions.size());
    auto tHostBasisFunctions = Kokkos::create_mirror(tBasisFunctions);
    Kokkos::deep_copy(tHostBasisFunctions, tBasisFunctions);

    tGoldScalar = 1.0/2.0;
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(0), tGoldScalar, tTolerance);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(1), tGoldScalar, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, HyperbolicTangentProjection)
{
    Plato::HyperbolicTangentProjection tProjection;
    
    // ******************** TEST APPLY FUNCTION ********************
    const Plato::Scalar tDensity = 0.5;
    Plato::Scalar tOutput = tProjection.apply(tDensity);
    Plato::Scalar tGoldValue = 0.5;
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tOutput, tGoldValue, tTolerance);
    
    // ******************** TEST SET TRESHOLD LEVEL PARAM FUNCTION ********************
    Plato::Scalar tEta = 0.1;
    tProjection.setTresholdLevelParameterEta(tEta);
    tOutput = tProjection.apply(tDensity);
    tGoldValue = 0.999619282;
    TEST_FLOATING_EQUALITY(tOutput, tGoldValue, tTolerance);
    
    // ******************** TEST SET CURVATURE PARAM FUNCTION ********************
    Plato::Scalar tBeta = 1;
    tProjection.setCurvatureParameterBeta(tBeta);
    tOutput = tProjection.apply(tDensity);
    tGoldValue = 0.587790467;
    TEST_FLOATING_EQUALITY(tOutput, tGoldValue, tTolerance);
    
    // ******************** TEST SET PARAMETERS FUNCTION ********************
    tEta = 0.15;
    tBeta = 5;
    tProjection.setProjectionParameters(tBeta, tEta);
    tOutput = tProjection.apply(tDensity);
    tGoldValue = 0.964387283;
    TEST_FLOATING_EQUALITY(tOutput, tGoldValue, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, HeavisideProjection)
{
    Plato::HeavisideProjection tProjection;
    
    // ******************** TEST APPLY FUNCTION ********************
    const Plato::Scalar tDensity = 0.5;
    Plato::Scalar tOutput = tProjection.apply(tDensity);
    Plato::Scalar tGoldValue = 0.993284752;
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tOutput, tGoldValue, tTolerance);

    // ******************** TEST SET CURVATURE PARAM FUNCTION ********************
    Plato::Scalar tBeta = 1;
    tProjection.setCurvatureParameterBeta(tBeta);
    tOutput = tProjection.apply(tDensity);
    tGoldValue = 0.5774090608;
    TEST_FLOATING_EQUALITY(tOutput, tGoldValue, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ApplyPenaltyHeaviside)
{
    // ALLOCATE PROJECTION AND PENALTY MODELS   
    Teuchos::ParameterList tParamList;
    tParamList.set<double>("Exponent", 3.0);
    tParamList.set<double>("Minimum Value", 0.0);
    SIMP tPenaltyModel(tParamList);
    Plato::HeavisideProjection tProjection;
    
    // ALLOCATE PROJECTION AND PENALTY FUNCTORS 
    Plato::OrdinalType tNumCellNodes = 4;
    Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ApplyPenalty<SIMP> tApplyPenalty(tPenaltyModel);
    Plato::ApplyProjection<Plato::HeavisideProjection> tApplyProjection(tProjection);

    // ALLOCATE AND INITIALIZE STRESS VIEW
    Plato::OrdinalType tNumCells = 2;
    Plato::ScalarMultiVectorT<Plato::Scalar> tStress("Stress",tNumCells,tNumVoigtTerms);
    auto tHostStress = Kokkos::create_mirror(tStress);
    tHostStress(0,0) = 1; tHostStress(1,0) = 2;
    tHostStress(0,1) = 2; tHostStress(1,1) = 3;
    tHostStress(0,2) = 3; tHostStress(1,2) = 4;
    tHostStress(0,3) = 4; tHostStress(1,3) = 5;
    tHostStress(0,4) = 5; tHostStress(1,4) = 6;
    tHostStress(0,5) = 6; tHostStress(1,5) = 7;
    Kokkos::deep_copy(tStress, tHostStress);

    // ALLOCATE AND INITIALIZE CONTROL VIEW
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("Control",tNumCells,tNumCellNodes);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.1; tHostControl(1,0) = 0.2;
    tHostControl(0,1) = 0.2; tHostControl(1,1) = 0.3;
    tHostControl(0,2) = 0.3; tHostControl(1,2) = 0.4;
    tHostControl(0,3) = 0.4; tHostControl(1,3) = 0.5;
    Kokkos::deep_copy(tControl, tHostControl);
    
    // RUN KERNEL
    auto & tDevicePenalty = tApplyPenalty;
    auto & tDeviceProjection = tApplyProjection;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        Plato::Scalar tCellDensity = tDeviceProjection(aCellOrdinal, tControl);
        tDevicePenalty(aCellOrdinal, tCellDensity, tStress);
    }, "UnitTest::ApplyPenaltyFunctor");
    
    // TEST OUTPUT
    tHostStress = Kokkos::create_mirror(tStress);
    Kokkos::deep_copy(tHostStress, tStress);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar>> tGoldValues =
        { {0.998443704781892, 1.996887409563784, 2.995331114345676, 3.993774819127569, 4.992218523909461, 5.990662228691353}, 
          {1.991169452968485, 2.986754179452727, 3.982338905936969, 4.977923632421212, 5.973508358905454, 6.969093085389696} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < tNumVoigtTerms; tVoigtIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostStress(tCellIndex, tVoigtIndex), tGoldValues[tCellIndex][tVoigtIndex], tTolerance);
       }
    } 
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ApplyPenaltyHyperbolicTangent_Array2D)
{
    // ALLOCATE PROJECTION AND PENALTY MODELS   
    Teuchos::ParameterList tParamList;
    tParamList.set<double>("Exponent", 3.0);
    tParamList.set<double>("Minimum Value", 0.0);
    SIMP tPenaltyModel(tParamList);
    Plato::HyperbolicTangentProjection tProjection;
    
    // ALLOCATE PROJECTION AND PENALTY FUNCTORS 
    Plato::OrdinalType tNumCellNodes = 4;
    Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ApplyPenalty<SIMP> tApplyPenalty(tPenaltyModel);
    Plato::ApplyProjection<Plato::HyperbolicTangentProjection> tApplyProjection(tProjection);

    // ALLOCATE AND INITIALIZE STRESS VIEW
    Plato::OrdinalType tNumCells = 2;
    Plato::ScalarMultiVectorT<Plato::Scalar> tStress("Stress",tNumCells,tNumVoigtTerms);
    auto tHostStress = Kokkos::create_mirror(tStress);
    tHostStress(0,0) = 1; tHostStress(1,0) = 2;
    tHostStress(0,1) = 2; tHostStress(1,1) = 3;
    tHostStress(0,2) = 3; tHostStress(1,2) = 4;
    tHostStress(0,3) = 4; tHostStress(1,3) = 5;
    tHostStress(0,4) = 5; tHostStress(1,4) = 6;
    tHostStress(0,5) = 6; tHostStress(1,5) = 7;
    Kokkos::deep_copy(tStress, tHostStress);

    // ALLOCATE AND INITIALIZE CONTROL VIEW
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("Control",tNumCells,tNumCellNodes);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.3; tHostControl(1,0) = 0.7;
    tHostControl(0,1) = 0.4; tHostControl(1,1) = 0.8;
    tHostControl(0,2) = 0.5; tHostControl(1,2) = 0.9;
    tHostControl(0,3) = 0.6; tHostControl(1,3) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);
    
    // RUN KERNEL
    auto & tDevicePenalty = tApplyPenalty;
    auto & tDeviceProjection = tApplyProjection;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        Plato::Scalar tCellDensity = tDeviceProjection(aCellOrdinal, tControl);
        tDevicePenalty(aCellOrdinal, tCellDensity, tStress);
    }, "UnitTest::ApplyPenaltyFunctor");
    
    // TEST OUTPUT
    tHostStress = Kokkos::create_mirror(tStress);
    Kokkos::deep_copy(tHostStress, tStress);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar>> tGoldValues =
        { {0.019447843055967, 0.038895686111933, 0.058343529167900, 0.077791372223866, 0.097239215279833, 0.116687058335799}, 
          {1.994810104070131, 2.992215156105197, 3.989620208140262, 4.987025260175328, 5.984430312210393, 6.981835364245459} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < tNumVoigtTerms; tVoigtIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostStress(tCellIndex, tVoigtIndex), tGoldValues[tCellIndex][tVoigtIndex], tTolerance);
       }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, StateValuesFunctor)
{
    // ALLOCATE AND INITIALIZE NODAL STATES VIEW
    const Plato::OrdinalType tNumDofs = 12;
    const Plato::OrdinalType tNumCells = 2;
    Plato::ScalarMultiVectorT<Plato::Scalar> tStates("States",tNumCells,tNumDofs);
    auto tHostStates = Kokkos::create_mirror(tStates);
    tHostStates(0,0) = 0.3; tHostStates(1,0) = 2.7;
    tHostStates(0,1) = 0.4; tHostStates(1,1) = 2.8;
    tHostStates(0,2) = 0.5; tHostStates(1,2) = 2.9;
    tHostStates(0,3) = 0.6; tHostStates(1,3) = 3.0;
    tHostStates(0,4) = 0.7; tHostStates(1,4) = 3.1;
    tHostStates(0,5) = 0.8; tHostStates(1,5) = 3.2;
    tHostStates(0,6) = 0.9; tHostStates(1,6) = 3.3;
    tHostStates(0,7) = 1.0; tHostStates(1,7) = 3.4;
    tHostStates(0,8) = 1.1; tHostStates(1,8) = 3.5;
    tHostStates(0,9) = 1.2; tHostStates(1,9) = 3.6;
    tHostStates(0,10) = 1.3; tHostStates(1,10) = 3.7;
    tHostStates(0,11) = 1.4; tHostStates(1,11) = 3.8;
    Kokkos::deep_copy(tStates, tHostStates);

    // ALLOCATE AND INITIALIZE CUB-POINTS STATES VIEW
    const Plato::OrdinalType tNumDofsPerNode = 4;
    Plato::ScalarMultiVectorT<Plato::Scalar> tValues("Values",tNumCells, tNumDofsPerNode);

    // RUN KERNEL
    Plato::StateValues tStateValues;
    auto & tDeviceStateValues = tStateValues;
    const Plato::OrdinalType tSpaceDim = 2;
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::OrdinalType tGold = 3;
    TEST_EQUALITY(tGold, tBasisFunctions.size());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tDeviceStateValues(aCellOrdinal, tBasisFunctions, tStates, tValues);
    }, "UnitTest::StateValues");

    // TEST OUTPUT
    auto tHostValues = Kokkos::create_mirror(tValues);
    Kokkos::deep_copy(tHostValues, tValues);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar>> tGoldValues = { {0.7, 0.8, 0.9, 1}, {3.1, 3.2, 3.3, 3.4} };
    for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
       for(Plato::OrdinalType tDof = 0; tDof < tNumDofsPerNode; tDof++)
       {
           TEST_FLOATING_EQUALITY(tHostValues(tCell, tDof), tGoldValues[tCell][tDof], tTolerance);
       }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, InertialForcesFunctor)
{
    // ALLOCATE AND INITIALIZE CUB-POINTS STATES VIEW
    const Plato::OrdinalType tNumCells = 2;
    const Plato::OrdinalType tSpaceDim = 3;
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateValues("StateValues", tNumCells, tSpaceDim);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    tHostStateValues(0,0) = 0.75; tHostStateValues(1,0) = 3.15;
    tHostStateValues(0,1) = 0.85; tHostStateValues(1,1) = 3.25;
    tHostStateValues(0,2) = 0.95; tHostStateValues(1,2) = 3.35;
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE AND INITIALIZE CELL VOLUMES VIEW
    Plato::ScalarVectorT<Plato::Scalar> tCellVolumes("CellVolumes", tNumCells);
    Plato::fill(static_cast<Plato::Scalar>(1), tCellVolumes);

    // ALLOCATE AND INITIALIZE INERTIAL FORCES VIEW
    const Plato::OrdinalType tNumDofs = 12;
    Plato::ScalarMultiVectorT<Plato::Scalar> tInertialForces("InertialForces", tNumCells, tNumDofs);

    // ALLOCATE CUBATURE RULE INSTANCE
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubRule;

    //ALLOCATE INERTIAL FORCE FUNCTOR (TO BE UNIT TESTED)
    const Plato::Scalar tDensity = 6.0;
    Plato::InertialForces tInertialForcesFunctor(tDensity);

    // RUN KERNEL
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    auto & tDeviceInertialForcesFunctor = tInertialForcesFunctor;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tDeviceInertialForcesFunctor(aCellOrdinal, tCellVolumes, tBasisFunctions, tStateValues, tInertialForces);
    }, "UnitTest::InertialForcesFunctor");

    // TEST OUTPUT
    auto tHostInertialForces = Kokkos::create_mirror(tInertialForces);
    Kokkos::deep_copy(tHostInertialForces, tInertialForces);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar>> tGoldValues = { {1.125, 1.275, 1.425, 1.125, 1.275, 1.425, 1.125, 1.275, 1.425, 1.125, 1.275, 1.425},
                                                            {4.725, 4.875, 5.025, 4.725, 4.875, 5.025, 4.725, 4.875, 5.025, 4.725, 4.875, 5.025} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostInertialForces(tCellIndex, tDofIndex), tGoldValues[tCellIndex][tDofIndex], tTolerance);
       }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComplexInertialEnergy)
{
    const Plato::OrdinalType tNumCells = 2;
    const Plato::OrdinalType tNumDofsPerCell = 6;
    Plato::ScalarMultiVector tStateValues("StateValues", tNumCells, tNumDofsPerCell);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    tHostStateValues(0,0) = 0.15; tHostStateValues(1,0) = 0.1;
    tHostStateValues(0,1) = 0.25; tHostStateValues(1,1) = 0.2;
    tHostStateValues(0,2) = 0.35; tHostStateValues(1,2) = 0.3;
    tHostStateValues(0,3) = 0.45; tHostStateValues(1,3) = 0.4;
    tHostStateValues(0,4) = 0.55; tHostStateValues(1,4) = 0.5;
    tHostStateValues(0,5) = 0.65; tHostStateValues(1,5) = 0.6;
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    Plato::ScalarVectorT<Plato::Scalar> tCellVolumes("CellVolumes", tNumCells);
    Plato::fill(static_cast<Plato::Scalar>(1), tCellVolumes);

    const Plato::Scalar tOmega = 1;
    const Plato::Scalar tDensity = 2;
    const Plato::OrdinalType tSpaceDim = 3;
    Plato::ScalarVector tOutput("InertialEnergy", tNumCells);
    Plato::ComplexInertialEnergy<tSpaceDim> tComputeInertialEnergy(tOmega, tDensity);

    auto & tDeviceComputeInertialEnergy = tComputeInertialEnergy;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tDeviceComputeInertialEnergy(aCellOrdinal, tCellVolumes, tStateValues, tOutput);
    }, "UnitTest::ComplexInertialEnergy");

    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldValues = { -2.27, -1.82 };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGoldValues[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComplexElasticEnergy)
{
    const Plato::OrdinalType tNumCells = 2;
    const Plato::OrdinalType tComplexDim = 2;
    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarArray3D tStress("Stress", tNumCells, tComplexDim, tNumVoigtTerms);
    auto tHostStress = Kokkos::create_mirror(tStress);
    tHostStress(0,0,0) = 1; tHostStress(1,0,0) = 2; tHostStress(0,1,0) = 3; tHostStress(1,1,0) = 4;
    tHostStress(0,0,1) = 1; tHostStress(1,0,1) = 2; tHostStress(0,1,1) = 3; tHostStress(1,1,1) = 4;
    tHostStress(0,0,2) = 1; tHostStress(1,0,2) = 2; tHostStress(0,1,2) = 3; tHostStress(1,1,2) = 4;
    tHostStress(0,0,3) = 1; tHostStress(1,0,3) = 2; tHostStress(0,1,3) = 3; tHostStress(1,1,3) = 4;
    tHostStress(0,0,4) = 1; tHostStress(1,0,4) = 2; tHostStress(0,1,4) = 3; tHostStress(1,1,4) = 4;
    tHostStress(0,0,5) = 1; tHostStress(1,0,5) = 2; tHostStress(0,1,5) = 3; tHostStress(1,1,5) = 4;
    Kokkos::deep_copy(tStress, tHostStress);

    Plato::ScalarArray3D tStrain("Strain", tNumCells, tComplexDim, tNumVoigtTerms);
    auto tHostStrain = Kokkos::create_mirror(tStrain);
    tHostStrain(0,0,0) = 0.1; tHostStrain(1,0,0) = 0.2; tHostStrain(0,1,0) = 0.3; tHostStrain(1,1,0) = 0.4;
    tHostStrain(0,0,1) = 0.1; tHostStrain(1,0,1) = 0.2; tHostStrain(0,1,1) = 0.3; tHostStrain(1,1,1) = 0.4;
    tHostStrain(0,0,2) = 0.1; tHostStrain(1,0,2) = 0.2; tHostStrain(0,1,2) = 0.3; tHostStrain(1,1,2) = 0.4;
    tHostStrain(0,0,3) = 0.1; tHostStrain(1,0,3) = 0.2; tHostStrain(0,1,3) = 0.3; tHostStrain(1,1,3) = 0.4;
    tHostStrain(0,0,4) = 0.1; tHostStrain(1,0,4) = 0.2; tHostStrain(0,1,4) = 0.3; tHostStrain(1,1,4) = 0.4;
    tHostStrain(0,0,5) = 0.1; tHostStrain(1,0,5) = 0.2; tHostStrain(0,1,5) = 0.3; tHostStrain(1,1,5) = 0.4;
    Kokkos::deep_copy(tStrain, tHostStrain);

    Plato::ScalarVector tOutput("ElasticEnergy", tNumCells);
    Plato::ComplexElasticEnergy<tNumVoigtTerms> tComputeElasticEnergy;

    auto & tDeviceComputeElasticEnergy = tComputeElasticEnergy;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tDeviceComputeElasticEnergy(aCellOrdinal, tStrain, tStress, tOutput);
    }, "UnitTest::ComplexElasticEnergy");

    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldValues = { 6.0, 12.0 };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGoldValues[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, DynamicCompliance)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::DynamicCompliance<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateValue(tResidual);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);
    
    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tValue = tScalarFunction.value(tStateValues, tControlValues, tAngularFrequency);

    Plato::Scalar tGoldValues = -0.0194453;
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValues, tValue, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, DynamicComplianceGradZ)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianZ>> tJacobianControl;
    tJacobianControl = std::make_shared<Plato::DynamicCompliance<JacobianZ, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientZ(tJacobianControl);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tGrad = tScalarFunction.gradient_z(tStateValues, tControlValues, tAngularFrequency);

    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);

    const Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGoldValues =
            { 3.72932e-08, -3.09175e-07, -5.77288e-09, -1.10274e-06, -2.38076e-07,
             -1.23347e-07, -3.87987e-07, -2.98636e-07, -1.01517e-06, -2.65061e-06,
             -1.27689e-06, -4.66673e-07, -9.42993e-07, -5.83097e-06, -2.40639e-06,
             -1.53974e-06, -2.69517e-06, -2.57357e-06, -2.52789e-06, -4.5315e-06 ,
             -6.11205e-06, -5.25015e-06, -1.40662e-06, -1.47591e-06, -3.15872e-06,
             -3.2731e-06 , -1.40645e-06};
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, DynamicComplianceGradX)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianX>> tJacobianConfig;
    tJacobianConfig = std::make_shared<Plato::DynamicCompliance<JacobianX, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientX(tJacobianConfig);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tGrad = tScalarFunction.gradient_x(tStateValues, tControlValues, tAngularFrequency);

    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);

    const Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGoldValues =
        { 0.00236153, -0.000849408, -0.000147446, 0.0028519, -0.00066485, 0.0002445, 0.000982165, -0.00031721,
          3.70458e-05, 0.0033533, 0.00104842, -0.000439875, 0.00123173, 0.000197775, -0.000541206, 0.000902254,
          -5.60646e-05, -0.00019819, 0.00151367, -0.000761575, -0.0002661, 0.000631465, -0.000893254, 0.00039139,
          0.000594981, 3.62625e-05, 0.000809181 };
    const Plato::OrdinalType tNumConfig = tMesh->nverts();
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumConfig; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, DynamicComplianceGradU)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::DynamicCompliance<JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientU(tJacobianState);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tGrad = tScalarFunction.gradient_u(tStateValues, tControlValues, tAngularFrequency);

    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldValues =
        { -0.0057, -0.00179375, -0.0023875, -0.00579375, -0.0018875, -0.00248125, -0.00509167, -0.00108333, -0.00495,
                -0.00521667, -0.00120833, -0.005075, -0.000254167, -0.000264583, -0.0032, -0.000285417, -0.000295833,
                -0.00323125, -0.00307969, -0.00719219, -0.00567969, -0.00326719, -0.00737969, -0.00586719, 0.00102604,
                -0.00191979, -0.00464062, 0.000963542, -0.00198229, -0.00470312, 0.000602083, -0.00210833, -0.00211875,
                0.000570833, -0.00213958, -0.00215, 0.000205729, -0.0043151, -0.00163594, 0.000143229, -0.0043776,
                -0.00169844, -0.000260417, -0.00207083, -0.00028125, -0.000291667, -0.00210208, -0.0003125,
                -0.000457292, -0.00241146, 0.000809375, -0.000582292, -0.00253646, 0.000684375, -0.00966094,
                -0.00657344, -0.00596094, -0.00984844, -0.00676094, -0.00614844, -0.00443333, -0.00265417, -0.0038,
                -0.00449583, -0.00271667, -0.0038625, -0.00181042, -0.00182083, -3.125e-05, -0.00184167, -0.00185208,
                -6.25e-05, -0.00140052, -0.00322135, 0.000357812, -0.00146302, -0.00328385, 0.000295312, -0.0135156,
                -0.0106031, -0.0112906, -0.0138906, -0.0109781, -0.0116656, -0.004125, -0.0050875, -0.008075,
                -0.0043125, -0.005275, -0.0082625, -0.00101979, -0.00398646, -0.00447812, -0.00114479, -0.00411146,
                -0.00460312, -0.00137187, -0.00537187, -0.00194687, -0.00155937, -0.00555937, -0.00213437, -0.00462292,
                -0.00488958, -0.00268125, -0.00474792, -0.00501458, -0.00280625, -0.0026125, -0.00354375, -0.003575,
                -0.00270625, -0.0036375, -0.00366875, -0.00372604, -0.00489271, -0.00470938, -0.00385104, -0.00501771,
                -0.00483438, -0.00398438, -0.00427188, -0.00365938, -0.00417188, -0.00445937, -0.00384688, -0.00799062,
                4.6875e-05, -0.00316563, -0.00817812, -0.000140625, -0.00335313, -0.00143646, 0.00146771, -0.00102812,
                -0.00149896, 0.00140521, -0.00109062, -0.000904167, -0.00136458, -0.000925, -0.000935417, -0.00139583,
                -0.00095625, -0.00137708, -0.00297292, -0.00186875, -0.00143958, -0.00303542, -0.00193125, -0.00398854,
                0.00204479, -0.000921875, -0.00411354, 0.00191979, -0.00104688, 0.000114583, -0.00102083, -0.00103125,
                8.33333e-05, -0.00105208, -0.0010625 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ExpVolumeValue)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ExpVolume<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateValue(tResidual);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tValue = tScalarFunction.value(tStateValues, tControlValues, tAngularFrequency);

    Plato::Scalar tGoldValues = 1.0;
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValues, tValue, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ExpVolumeGradZ)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianZ>> tJacobianControl;
    tJacobianControl = std::make_shared<Plato::ExpVolume<JacobianZ, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientZ(tJacobianControl);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tGrad = tScalarFunction.gradient_z(tStateValues, tControlValues, tAngularFrequency);

    const Plato::Scalar tTolerance = 1e-4;
    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);
    std::vector<Plato::Scalar> tGoldValues =
        { 8.51249e-05, 0.0001135, 2.8375e-05, 0.00017025, 5.67499e-05, 2.8375e-05, 5.67499e-05, 2.8375e-05, 0.0001135,
                0.00017025, 5.67499e-05, 2.8375e-05, 5.67499e-05, 0.000340499, 0.00017025, 0.0001135, 0.00017025,
                0.0001135, 8.51249e-05, 0.0001135, 0.00017025, 0.00017025, 5.67499e-05, 2.8375e-05, 5.67499e-05,
                0.0001135, 2.8375e-05 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ExpVolumeGradU)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ExpVolume<JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientU(tJacobianState);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tGrad = tScalarFunction.gradient_u(tStateValues, tControlValues, tAngularFrequency);

    const Plato::Scalar tTolerance = 1e-5;
    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);
    std::vector<Plato::Scalar> tGoldValues(tNumStates, 0.0);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ExpVolumeGradX)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 3;
    using JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;

    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE INVERSE COMPLIANCE TRANSFER FUNCTION CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianX>> tJacobianConfig;
    tJacobianConfig = std::make_shared<Plato::ExpVolume<JacobianX, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientX(tJacobianConfig);

    // ALLOCATE STATES
    const Plato::OrdinalType tNumDofsPerNode = 6;
    const Plato::OrdinalType tNumStates = tMesh->nverts() * tNumDofsPerNode;
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    Plato::OrdinalType tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tGrad = tScalarFunction.gradient_x(tStateValues, tControlValues, tAngularFrequency);

    const Plato::Scalar tTolerance = 1e-5;
    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);
    auto tNumConfig = tMesh->nverts() * tSpaceDim;
    TEST_EQUALITY(tNumConfig, tGrad.size());
    std::vector<Plato::Scalar> tGoldValues =
        { -0.0833333, -0.0833333, -0.0833333, -0.125, -0.125, 0, -0.0416667, -0.0416667, 0.0833333, -0.25, 0,
                -1.38778e-17, -0.125, 0, 0.125, -0.0833333, 0.0416667, 0.0416667, -0.125, 0.125, 0, -0.0416667,
                0.0833333, -0.0416667, -0.125, 0, -0.125, 1.38778e-17, -1.38778e-17, -0.25, 0.125, 0, -0.125, 0.0416667,
                0.0416667, -0.0833333, 0, 0.125, -0.125, 1.38778e-17, -1.38778e-17, -1.38778e-17, 0, 0, 0.25, 0, 0.125,
                0.125, 0, 0.25, 0, 0.125, 0.125, 0, 0.0833333, 0.0833333, 0.0833333, 0.125, 0, 0.125, 0.25,
                -1.38778e-17, 0, 1.38778e-17, -0.25, -1.38778e-17, 0, -0.125, 0.125, 0.0416667, -0.0833333, 0.0416667,
                0.125, -0.125, 0, 0, -0.125, -0.125, 0.0833333, -0.0416667, -0.0416667 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumConfig; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComplexRayleighDamping)
{
    // ALLOCATE INERTIAL FORCES
    const Plato::OrdinalType tNumCells = 2;
    const Plato::OrdinalType tNumDofsPerNode = 6;
    Plato::ScalarMultiVector tInertialForces("InertialForces", tNumCells, tNumDofsPerNode);
    auto tHostInertialForces = Kokkos::create_mirror(tInertialForces);
    tHostInertialForces(0,0) = 1; tHostInertialForces(1,0) = 2;
    tHostInertialForces(0,1) = 1; tHostInertialForces(1,1) = 2;
    tHostInertialForces(0,2) = 1; tHostInertialForces(1,2) = 2;
    tHostInertialForces(0,3) = 2; tHostInertialForces(1,3) = 4;
    tHostInertialForces(0,4) = 2; tHostInertialForces(1,4) = 4;
    tHostInertialForces(0,5) = 2; tHostInertialForces(1,5) = 4;
    Kokkos::deep_copy(tInertialForces, tHostInertialForces);

    // ALLOCATE ELASTIC FORCES
    Plato::ScalarMultiVector tElasticForces("ElasticForces", tNumCells, tNumDofsPerNode);
    auto tHostElasticForces = Kokkos::create_mirror(tElasticForces);
    tHostElasticForces(0,0) = 2; tHostElasticForces(1,0) = 4;
    tHostElasticForces(0,1) = 2; tHostElasticForces(1,1) = 4;
    tHostElasticForces(0,2) = 2; tHostElasticForces(1,2) = 4;
    tHostElasticForces(0,3) = 4; tHostElasticForces(1,3) = 8;
    tHostElasticForces(0,4) = 4; tHostElasticForces(1,4) = 8;
    tHostElasticForces(0,5) = 4; tHostElasticForces(1,5) = 8;
    Kokkos::deep_copy(tElasticForces, tHostElasticForces);

    // COMPUTE DAMPING FORCES
    const Plato::OrdinalType tSpaceDim = 3;
    Plato::ScalarMultiVector tDampingForces("DampingForces", tNumCells, tNumDofsPerNode);
    Plato::ComplexRayleighDamping<tSpaceDim> tComputeDamping(0.5, 0.5);

    // RUN KERNEL
    auto & tDeviceComputeDamping = tComputeDamping;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tDeviceComputeDamping(aCellOrdinal, tElasticForces, tInertialForces, tDampingForces);
    }, "UnitTest::ComplexRayleighDamping");

    // TEST OUTPUT VALUES
    auto tHostDampingForces = Kokkos::create_mirror(tDampingForces);
    Kokkos::deep_copy(tHostDampingForces, tDampingForces);
    std::vector<std::vector<Plato::Scalar>> tGoldValues =
        { {3.0, 3.0, 3.0, -1.5, -1.5, -1.5}, {6.0, 6.0, 6.0, -3.0, -3.0, -3.0} };

    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostDampingForces(tCellIndex, tDofIndex), tGoldValues[tCellIndex][tDofIndex], tTolerance);
       }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointComplexRayleighDamping)
{
    // ALLOCATE INERTIAL FORCES
    const Plato::OrdinalType tNumCells = 2;
    const Plato::OrdinalType tNumDofsPerNode = 6;
    Plato::ScalarMultiVector tInertialForces("InertialForces", tNumCells, tNumDofsPerNode);
    auto tHostInertialForces = Kokkos::create_mirror(tInertialForces);
    tHostInertialForces(0,0) = 1; tHostInertialForces(1,0) = 2;
    tHostInertialForces(0,1) = 1; tHostInertialForces(1,1) = 2;
    tHostInertialForces(0,2) = 1; tHostInertialForces(1,2) = 2;
    tHostInertialForces(0,3) = 2; tHostInertialForces(1,3) = 4;
    tHostInertialForces(0,4) = 2; tHostInertialForces(1,4) = 4;
    tHostInertialForces(0,5) = 2; tHostInertialForces(1,5) = 4;
    Kokkos::deep_copy(tInertialForces, tHostInertialForces);

    // ALLOCATE ELASTIC FORCES
    Plato::ScalarMultiVector tElasticForces("ElasticForces", tNumCells, tNumDofsPerNode);
    auto tHostElasticForces = Kokkos::create_mirror(tElasticForces);
    tHostElasticForces(0,0) = 2; tHostElasticForces(1,0) = 4;
    tHostElasticForces(0,1) = 2; tHostElasticForces(1,1) = 4;
    tHostElasticForces(0,2) = 2; tHostElasticForces(1,2) = 4;
    tHostElasticForces(0,3) = 4; tHostElasticForces(1,3) = 8;
    tHostElasticForces(0,4) = 4; tHostElasticForces(1,4) = 8;
    tHostElasticForces(0,5) = 4; tHostElasticForces(1,5) = 8;
    Kokkos::deep_copy(tElasticForces, tHostElasticForces);

    // COMPUTE DAMPING FORCES
    const Plato::OrdinalType tSpaceDim = 3;
    Plato::ScalarMultiVector tDampingForces("DampingForces", tNumCells, tNumDofsPerNode);
    Plato::AdjointComplexRayleighDamping<tSpaceDim> tComputeDamping(0.5, 0.5);

    // RUN KERNEL
    auto & tDeviceComputeDamping = tComputeDamping;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tDeviceComputeDamping(aCellOrdinal, tElasticForces, tInertialForces, tDampingForces);
    }, "UnitTest::AdjointComplexRayleighDamping");

    // TEST OUTPUT VALUES
    auto tHostDampingForces = Kokkos::create_mirror(tDampingForces);
    Kokkos::deep_copy(tHostDampingForces, tDampingForces);
    std::vector<std::vector<Plato::Scalar>> tGoldValues =
        { {-3.0, -3.0, -3.0, 1.5, 1.5, 1.5}, {-6.0, -6.0, -6.0, 3.0, 3.0, 3.0} };

    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostDampingForces(tCellIndex, tDofIndex), tGoldValues[tCellIndex][tDofIndex], tTolerance);
       }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, StructuralDynamicsResidual)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tVectorFunction(*tMesh, tDataMap);

    // ALLOCATE ELASTODYNAMICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    using ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::StructuralDynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::StructuralDynamicsResidual<JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tVectorFunction.allocateResidual(tResidual, tJacobianState);

    // TEST SIZE FUNCTION
    Plato::OrdinalType tSizeGold = 162;
    TEST_EQUALITY(tSizeGold, tVectorFunction.size());

    // ALLOCATE STATES
    const Plato::OrdinalType tNumStates = tVectorFunction.size();
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(0.1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);
    
    // ALLOCATE CONTROLS
    tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    auto tHostControlValues = Kokkos::create_mirror(tControlValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        tHostControlValues(tIndex) = static_cast<Plato::Scalar>(1.0);
    }    
    Kokkos::deep_copy(tControlValues, tHostControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tResidualVec = tVectorFunction.value(tStateValues, tControlValues, tAngularFrequency);

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldValues =
        { -0.003125, -0.003125, -0.003125, -0.003125, -0.003125, -0.003125, -0.00416667, -0.00416667, -0.00416667,
                -0.00416667, -0.00416667, -0.00416667, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167,
                -0.00104167, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00208333, -0.00208333,
                -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00104167, -0.00104167, -0.00104167, -0.00104167,
                -0.00104167, -0.00104167, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00208333,
                -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00416667, -0.00416667,
                -0.00416667, -0.00416667, -0.00416667, -0.00416667, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625,
                -0.00625, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00104167,
                -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00208333, -0.00208333, -0.00208333,
                -0.00208333, -0.00208333, -0.00208333, -0.0125, -0.0125, -0.0125, -0.0125, -0.0125, -0.0125, -0.00625,
                -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00416667, -0.00416667, -0.00416667, -0.00416667,
                -0.00416667, -0.00416667, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00416667,
                -0.00416667, -0.00416667, -0.00416667, -0.00416667, -0.00416667, -0.003125, -0.003125, -0.003125,
                -0.003125, -0.003125, -0.003125, -0.00416667, -0.00416667, -0.00416667, -0.00416667, -0.00416667,
                -0.00416667, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625, -0.00625,
                -0.00625, -0.00625, -0.00625, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00208333,
                -0.00208333, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00208333,
                -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00208333, -0.00416667, -0.00416667, -0.00416667,
                -0.00416667, -0.00416667, -0.00416667, -0.00104167, -0.00104167, -0.00104167, -0.00104167, -0.00104167,
                -0.00104167 };
    auto tHostResidualVec = Kokkos::create_mirror(tResidualVec);
    Kokkos::deep_copy(tHostResidualVec, tResidualVec);
    TEST_EQUALITY(tNumStates, tResidualVec.size());
    for(Plato::OrdinalType tIndex = 0; tIndex < tResidualVec.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostResidualVec(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointStructuralDynamicsResidual)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tVectorFunction(*tMesh, tDataMap);

    // ALLOCATE ADJOINT ELASTODYNAMICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    using ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::AdjointStructuralDynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::AdjointStructuralDynamicsResidual<JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tVectorFunction.allocateResidual(tResidual, tJacobianState);

    // TEST SIZE FUNCTION
    Plato::OrdinalType tSizeGold = 162;
    TEST_EQUALITY(tSizeGold, tVectorFunction.size());

    // ALLOCATE STATES
    const Plato::OrdinalType tNumStates = tVectorFunction.size();
    Plato::ScalarVector tStateValues("StateValues", tNumStates);
    auto tHostStateValues = Kokkos::create_mirror(tStateValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumStates; tIndex++)
    {
        tHostStateValues(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex + 1);
    }
    Kokkos::deep_copy(tStateValues, tHostStateValues);

    // ALLOCATE CONTROLS
    tSizeGold = 27;
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    TEST_EQUALITY(tSizeGold, tNumControls);
    Plato::ScalarVector tControlValues("ControlValues", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1.0), tControlValues);

    // TEST VALUE FUNCTION
    Plato::Scalar tAngularFrequency = 1.0;
    auto tOutput = tVectorFunction.value(tStateValues, tControlValues, tAngularFrequency);

    std::vector<Plato::Scalar> tGoldValues =
        { -0.0057, -0.00179375, -0.0023875, -0.00579375, -0.0018875, -0.00248125, -0.00509167, -0.00108333, -0.00495,
                -0.00521667, -0.00120833, -0.005075, -0.000254167, -0.000264583, -0.0032, -0.000285417, -0.000295833,
                -0.00323125, -0.00307969, -0.00719219, -0.00567969, -0.00326719, -0.00737969, -0.00586719, 0.00102604,
                -0.00191979, -0.00464062, 0.000963542, -0.00198229, -0.00470312, 0.000602083, -0.00210833, -0.00211875,
                0.000570833, -0.00213958, -0.00215, 0.000205729, -0.0043151, -0.00163594, 0.000143229, -0.0043776,
                -0.00169844, -0.000260417, -0.00207083, -0.00028125, -0.000291667, -0.00210208, -0.0003125, -0.000457292,
                -0.00241146, 0.000809375, -0.000582292, -0.00253646, 0.000684375, -0.00966094, -0.00657344, -0.00596094,
                -0.00984844, -0.00676094, -0.00614844, -0.00443333, -0.00265417, -0.0038, -0.00449583, -0.00271667,
                -0.0038625, -0.00181042, -0.00182083, -3.125e-05, -0.00184167, -0.00185208, -6.25e-05, -0.00140052,
                -0.00322135, 0.000357812, -0.00146302, -0.00328385, 0.000295312, -0.0135156, -0.0106031, -0.0112906,
                -0.0138906, -0.0109781, -0.0116656, -0.004125, -0.0050875, -0.008075, -0.0043125, -0.005275, -0.0082625,
                -0.00101979, -0.00398646, -0.00447812, -0.00114479, -0.00411146, -0.00460312, -0.00137187, -0.00537187,
                -0.00194687, -0.00155937, -0.00555937, -0.00213437, -0.00462292, -0.00488958, -0.00268125, -0.00474792,
                -0.00501458, -0.00280625, -0.0026125, -0.00354375, -0.003575, -0.00270625, -0.0036375, -0.00366875,
                -0.00372604, -0.00489271, -0.00470938, -0.00385104, -0.00501771, -0.00483438, -0.00398438, -0.00427188,
                -0.00365938, -0.00417188, -0.00445937, -0.00384688, -0.00799062, 4.6875e-05, -0.00316563, -0.00817812,
                -0.000140625, -0.00335313, -0.00143646, 0.00146771, -0.00102812, -0.00149896, 0.00140521, -0.00109062,
                -0.000904167, -0.00136458, -0.000925, -0.000935417, -0.00139583, -0.00095625, -0.00137708, -0.00297292,
                -0.00186875, -0.00143958, -0.00303542, -0.00193125, -0.00398854, 0.00204479, -0.000921875, -0.00411354,
                0.00191979, -0.00104688, 0.000114583, -0.00102083, -0.00103125, 8.33333e-05, -0.00105208, -0.0010625 };
    auto tHostResidual = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostResidual, tOutput);

    const Plato::Scalar tTolerance = 1e-5;
    TEST_EQUALITY(tNumStates, tOutput.size());
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostResidual(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeFrequencyResponseMisfit)
{
    //***** INITIALIZE TRIAL STATES *****
    const Plato::OrdinalType tNumDofs = 12;
    const Plato::OrdinalType tNumCells = 2;
    Plato::ScalarMultiVectorT<Plato::Scalar> tTrialStates("TrialStates", tNumCells, tNumDofs);
    auto tHostTrialStates = Kokkos::create_mirror(tTrialStates);
    tHostTrialStates(0,0) = 0.3; tHostTrialStates(1,0) = 2.7;
    tHostTrialStates(0,1) = 0.4; tHostTrialStates(1,1) = 2.8;
    tHostTrialStates(0,2) = 0.5; tHostTrialStates(1,2) = 2.9;
    tHostTrialStates(0,3) = 0.6; tHostTrialStates(1,3) = 3.0;
    tHostTrialStates(0,4) = 0.7; tHostTrialStates(1,4) = 3.1;
    tHostTrialStates(0,5) = 0.8; tHostTrialStates(1,5) = 3.2;
    tHostTrialStates(0,6) = 0.9; tHostTrialStates(1,6) = 3.3;
    tHostTrialStates(0,7) = 1.0; tHostTrialStates(1,7) = 3.4;
    tHostTrialStates(0,8) = 1.1; tHostTrialStates(1,8) = 3.5;
    tHostTrialStates(0,9) = 1.2; tHostTrialStates(1,9) = 3.6;
    tHostTrialStates(0,10) = 1.3; tHostTrialStates(1,10) = 3.7;
    tHostTrialStates(0,11) = 1.4; tHostTrialStates(1,11) = 3.8;
    Kokkos::deep_copy(tTrialStates, tHostTrialStates);

    //***** INITIALIZE EXPERIMENTAL STATES *****
    Plato::ScalarMultiVectorT<Plato::Scalar> tExpStates("ExpStates", tNumCells, tNumDofs);
    auto tHostExpStates = Kokkos::create_mirror(tExpStates);
    tHostExpStates(0,0) = 0.35; tHostExpStates(1,0) = 2.72;
    tHostExpStates(0,1) = 0.45; tHostExpStates(1,1) = 2.82;
    tHostExpStates(0,2) = 0.55; tHostExpStates(1,2) = 2.92;
    tHostExpStates(0,3) = 0.65; tHostExpStates(1,3) = 3.02;
    tHostExpStates(0,4) = 0.75; tHostExpStates(1,4) = 3.12;
    tHostExpStates(0,5) = 0.85; tHostExpStates(1,5) = 3.22;
    tHostExpStates(0,6) = 0.95; tHostExpStates(1,6) = 3.32;
    tHostExpStates(0,7) = 1.05; tHostExpStates(1,7) = 3.42;
    tHostExpStates(0,8) = 1.15; tHostExpStates(1,8) = 3.52;
    tHostExpStates(0,9) = 1.25; tHostExpStates(1,9) = 3.62;
    tHostExpStates(0,10) = 1.35; tHostExpStates(1,10) = 3.72;
    tHostExpStates(0,11) = 1.45; tHostExpStates(1,11) = 3.82;
    Kokkos::deep_copy(tExpStates, tHostExpStates);

    //***** COMPUTE FRF MISFIT *****
    const Plato::OrdinalType tSpaceDim = 2;
    Plato::ComputeFrequencyResponseMisfit<tSpaceDim> tComputeMisfit;
    Plato::ScalarVectorT<Plato::Scalar> tOutput("Output", tNumCells);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeMisfit(aCellOrdinal, tExpStates, tTrialStates, tOutput);
    }, "UnitTest::ComputeMisfitFRF");

    // ***** UNIT TEST FUNCTION *****
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldValues = {0.015, 0.0024};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tIndex), tGoldValues[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FrequencyResponseMisfitValue)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // SET PROBLEM-RELATED DIMENSIONS
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(2));
    const Plato::OrdinalType tNumVertices = tMesh->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(4));
    const Plato::OrdinalType tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(16));

    // ALLOCATE STATES FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tStates("States", tTotalNumDofs);
    auto tHostStates = Kokkos::create_mirror(tStates);
    const Plato::OrdinalType tNumFreq = 1;
    Plato::ScalarMultiVector tExpStates("ExpStates", tNumFreq, tTotalNumDofs);
    auto tMyExpStates = Kokkos::subview(tExpStates, static_cast<Plato::OrdinalType>(tNumFreq - 1), Kokkos::ALL());
    auto tHostMyExpStates = Kokkos::create_mirror(tMyExpStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostStates(tIndex) = static_cast<Plato::Scalar>(1e-2) * static_cast<Plato::Scalar>(tIndex);
        tHostMyExpStates(tIndex) = static_cast<Plato::Scalar>(2.5e-2) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tStates, tHostStates);
    Kokkos::deep_copy(tMyExpStates, tHostMyExpStates);

    // ALLOCATE FREQUENCY RESPONSE MISFIT CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::vector<Plato::Scalar> tFreqArray = {15.0};
    using ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    std::shared_ptr<AbstractScalarFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::FrequencyResponseMisfit<ResidualT>>(*tMesh, tMeshSets, tDataMap, tFreqArray, tExpStates);
    
    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateValue(tResidual);

    // ALLOCATE CONTROLS FOR ELASTOSTATICS EXAMPLE
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControls);

    // TEST VALUE FUNCTION
    const Plato::Scalar tTolerance = 1e-4;
    Plato::Scalar tGoldValues = 0.18225;
    auto tOutput = tScalarFunction.value(tStates, tControls, tFreqArray[0]);
    TEST_FLOATING_EQUALITY(tOutput, tGoldValues, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FrequencyResponseMisfit_GradZ)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // SET EVALUATION TYPES FOR UNIT TEST

    // SET PROBLEM-RELATED DIMENSIONS
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(2));
    const Plato::OrdinalType tNumVertices = tMesh->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(4));
    const Plato::OrdinalType tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(16));

    // ALLOCATE STATES FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tStates("States", tTotalNumDofs);
    auto tHostStates = Kokkos::create_mirror(tStates);
    const Plato::OrdinalType tNumFreq = 1;
    Plato::ScalarMultiVector tExpStates("ExpStates", tNumFreq, tTotalNumDofs);
    auto tMyExpStates = Kokkos::subview(tExpStates, static_cast<Plato::OrdinalType>(tNumFreq - 1), Kokkos::ALL());
    auto tHostMyExpStates = Kokkos::create_mirror(tMyExpStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostStates(tIndex) = static_cast<Plato::Scalar>(1e-2) * static_cast<Plato::Scalar>(tIndex);
        tHostMyExpStates(tIndex) = static_cast<Plato::Scalar>(2.5e-2) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tStates, tHostStates);
    Kokkos::deep_copy(tMyExpStates, tHostMyExpStates);

    // ALLOCATE FREQUENCY RESPONSE MISFIT CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::vector<Plato::Scalar> tFreqArray = {15.0};
    using JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    std::shared_ptr<AbstractScalarFunction<JacobianZ>> tGradControl;
    tGradControl = std::make_shared<Plato::FrequencyResponseMisfit<JacobianZ>>(*tMesh, tMeshSets, tDataMap, tFreqArray, tExpStates);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientZ(tGradControl);

    // ALLOCATE CONTROLS FOR ELASTOSTATICS EXAMPLE
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControls);

    // TEST GRADIENT WRT CONTROLS
    auto tGrad = tScalarFunction.gradient_z(tStates, tControls, tFreqArray[0]);
    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);

    const Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold(tNumControls, 0.);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FrequencyResponseMisfit_GradX)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // SET EVALUATION TYPES FOR UNIT TEST
    using JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;

    // SET PROBLEM-RELATED DIMENSIONS
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(2));
    const Plato::OrdinalType tNumVertices = tMesh->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(4));
    const Plato::OrdinalType tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(16));

    // ALLOCATE STATES FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tStates("States", tTotalNumDofs);
    auto tHostStates = Kokkos::create_mirror(tStates);
    std::vector<Plato::Scalar> tFreqArray = {15.0};
    const Plato::OrdinalType tNumFreq = tFreqArray.size();
    Plato::ScalarMultiVector tExpStates("ExpStates", tNumFreq, tTotalNumDofs);
    auto tMyExpStates = Kokkos::subview(tExpStates, static_cast<Plato::OrdinalType>(tNumFreq - 1), Kokkos::ALL());
    auto tHostMyExpStates = Kokkos::create_mirror(tMyExpStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostStates(tIndex) = static_cast<Plato::Scalar>(1e-2) * static_cast<Plato::Scalar>(tIndex);
        tHostMyExpStates(tIndex) = static_cast<Plato::Scalar>(2.5e-2) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tStates, tHostStates);
    Kokkos::deep_copy(tMyExpStates, tHostMyExpStates);

    // ALLOCATE FREQUENCY RESPONSE MISFIT CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianX>> tJacobianConfig;
    tJacobianConfig = std::make_shared<Plato::FrequencyResponseMisfit<JacobianX>>(*tMesh, tMeshSets, tDataMap, tFreqArray, tExpStates);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientX(tJacobianConfig);

    // ALLOCATE CONTROLS FOR ELASTOSTATICS EXAMPLE
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControls);

    // TEST GRADIENT WRT CONFIGURATION
    auto tGrad = tScalarFunction.gradient_x(tStates, tControls, tFreqArray[0]);
    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);

    const Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold(tNumControls, 0.);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumControls; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, FrequencyResponseMisfit_GradU)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // SET EVALUATION TYPES FOR UNIT TEST
    using JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;

    // SET PROBLEM-RELATED DIMENSIONS
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(2));
    const Plato::OrdinalType tNumVertices = tMesh->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(4));
    const Plato::OrdinalType tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(16));

    // ALLOCATE STATES FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tStates("States", tTotalNumDofs);
    auto tHostStates = Kokkos::create_mirror(tStates);
    std::vector<Plato::Scalar> tFreqArray = {15.0};
    const Plato::OrdinalType tNumFreq = tFreqArray.size();
    Plato::ScalarMultiVector tExpStates("ExpStates", tNumFreq, tTotalNumDofs);
    auto tMyExpStates = Kokkos::subview(tExpStates, static_cast<Plato::OrdinalType>(tNumFreq - 1), Kokkos::ALL());
    auto tHostMyExpStates = Kokkos::create_mirror(tMyExpStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostStates(tIndex) = static_cast<Plato::Scalar>(1e-2) * static_cast<Plato::Scalar>(tIndex);
        tHostMyExpStates(tIndex) = static_cast<Plato::Scalar>(2.5e-2) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tStates, tHostStates);
    Kokkos::deep_copy(tMyExpStates, tHostMyExpStates);

    // ALLOCATE FREQUENCY RESPONSE MISFIT CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractScalarFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::FrequencyResponseMisfit<JacobianU>>(*tMesh, tMeshSets, tDataMap, tFreqArray, tExpStates);

    // ALLOCATE SCALAR FUNCTION
    ScalarFunction<Plato::StructuralDynamics<tSpaceDim>> tScalarFunction(*tMesh, tDataMap);
    tScalarFunction.allocateGradientU(tJacobianState);

    // ALLOCATE CONTROLS FOR ELASTOSTATICS EXAMPLE
    const Plato::OrdinalType tNumControls = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumControls);
    Plato::fill(static_cast<Plato::Scalar>(1), tControls);

    // TEST GRADIENT WRT STATES
    auto tGrad = tScalarFunction.gradient_u(tStates, tControls, tFreqArray[0]);
    TEST_EQUALITY(tGrad.size(), tTotalNumDofs);

    auto tHostGrad = Kokkos::create_mirror(tGrad);
    Kokkos::deep_copy(tHostGrad, tGrad);

    const Plato::Scalar tTolerance = 1e-5;
    std::vector<Plato::Scalar> tGold =
        { 0.0, -0.03, -0.06, -0.09, -0.06, -0.075, -0.09, -0.105, -0.24, -0.27, -0.3, -0.33, -0.18, -0.195, -0.21, -0.225 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, StructuralDynamicsSolve)
{
    // CREATE 2D-MESH
    Omega_h::LO aNx = 4;
    Omega_h::LO aNy = 4;
    Omega_h::Real aX = 1;
    Omega_h::Real aY = 1;
    std::shared_ptr<Omega_h::Mesh> tMesh = PlatoUtestHelpers::build_2d_box_mesh(aX, aY, aNx, aNy);

    // PROBLEM INPUTS
    const Plato::Scalar tDensity = 1000;
    const Plato::Scalar tPoissonRatio = 0.3;
    const Plato::Scalar tYoungsModulus = 1e9;
    const Plato::Scalar tMassPropDamping = 0.000025;
    const Plato::Scalar tStiffPropDamping = 0.000023;

    // ALLOCATE STRUCTURAL DYNAMICS RESIDUAL
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    const Plato::OrdinalType tSpaceDim = 2;
    using ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    std::shared_ptr<Plato::StructuralDynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>> tResidual;
    tResidual = std::make_shared<Plato::StructuralDynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tResidual->setMaterialDensity(tDensity);
    tResidual->setMassPropDamping(tMassPropDamping);
    tResidual->setStiffPropDamping(tStiffPropDamping);
    tResidual->setIsotropicLinearElasticMaterial(tYoungsModulus, tPoissonRatio);
    
    std::shared_ptr<Plato::StructuralDynamicsResidual<JacobianU, SIMP, Plato::HyperbolicTangentProjection>> tJacobianState;
    tJacobianState = std::make_shared<Plato::StructuralDynamicsResidual<JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tResidual->setMaterialDensity(tDensity);
    tJacobianState->setMassPropDamping(tMassPropDamping);
    tJacobianState->setStiffPropDamping(tStiffPropDamping);
    tJacobianState->setIsotropicLinearElasticMaterial(tYoungsModulus, tPoissonRatio);
   
    // ALLOCATE VECTOR FUNCTION
    std::shared_ptr<VectorFunction<Plato::StructuralDynamics<tSpaceDim>>> tVectorFunction =
        std::make_shared<VectorFunction<Plato::StructuralDynamics<tSpaceDim>>>(*tMesh, tDataMap);
    tVectorFunction->allocateResidual(tResidual, tJacobianState);

    // ALLOCATE STRUCTURAL DYNAMICS PROBLEM
    Plato::StructuralDynamicsProblem<Plato::StructuralDynamics<tSpaceDim>> tProblem(*tMesh, tVectorFunction);

    // SET DIRICHLET BOUNDARY CONDITIONS
    Plato::Scalar tValue = 0;
    auto tNumDofsPerNode = 2*tSpaceDim;
    Omega_h::LOs tCoordsX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    auto tNumDirichletDofs = tNumDofsPerNode*tCoordsX0.size();
    Plato::ScalarVector tDirichletValues("DirichletValues", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("DirichletDofs", tNumDirichletDofs);
    PlatoUtestHelpers::set_dirichlet_boundary_conditions(tNumDofsPerNode, tValue, tCoordsX0, tDirichletDofs, tDirichletValues);
    tProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // SET FREQUENCY    
    std::vector<Plato::Scalar> tFreq = { 5 };
    tProblem.setFrequencyArray(tFreq);

    // SET EXTERNAL FORCE
    auto tNumDofs = tVectorFunction->size();
    auto tNumDofsGold = tNumDofsPerNode * tMesh->nverts();
    TEST_EQUALITY(tNumDofsGold, tNumDofs);   
    Plato::ScalarVector tPointLoad("PointLoad", tNumDofs);
    
    Plato::ScalarMultiVector tValues("Values", 2, tSpaceDim);
    auto tHostValues = Kokkos::create_mirror(tValues);
    tHostValues(0,0) = 0;    tHostValues(1,0) = 0;
    tHostValues(0,1) = -1e5; tHostValues(1,1) = -1e5;
    Kokkos::deep_copy(tValues, tHostValues);
    
    auto tTopOrdinalIndex = 0;
    auto tNodeOrdinalsX1 = PlatoUtestHelpers::get_2D_boundary_nodes_x1(*tMesh);
    PlatoUtestHelpers::set_point_load(tTopOrdinalIndex, tNodeOrdinalsX1, tValues, tPointLoad);
    tProblem.setExternalForce(tPointLoad); 

    //SOLVE ELASTODYNAMICS PROBLEM
    auto tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Kokkos::deep_copy(tControl, static_cast<Plato::Scalar>(1));
    tProblem.setMaxNumIterationsAmgX(500);
    auto tSolution = tProblem.solution(tControl);

    // OUTPUT DATA
    //Plato::StructuralDynamicsOutput<tSpaceDim> tOutput(*tMesh);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointStructuralDynamicsSolve)
{
    // CREATE 2D-MESH
    Omega_h::LO aNx = 4;
    Omega_h::LO aNy = 4;
    Omega_h::Real aX = 1;
    Omega_h::Real aY = 1;
    std::shared_ptr<Omega_h::Mesh> tMesh = PlatoUtestHelpers::build_2d_box_mesh(aX, aY, aNx, aNy);

    // PROBLEM INPUTS
    const Plato::Scalar tDensity = 1000;
    const Plato::Scalar tPoissonRatio = 0.3;
    const Plato::Scalar tYoungsModulus = 1e9;
    const Plato::Scalar tMassPropDamping = 0.000025;
    const Plato::Scalar tStiffPropDamping = 0.000023;

    // ALLOCATE STRUCTURAL DYNAMICS RESIDUAL
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    const Plato::OrdinalType tSpaceDim = 2;
    using ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    std::shared_ptr<Plato::AdjointStructuralDynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>> tResidual;
    tResidual = std::make_shared<Plato::AdjointStructuralDynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tResidual->setMaterialDensity(tDensity);
    tResidual->setMassPropDamping(tMassPropDamping);
    tResidual->setStiffPropDamping(tStiffPropDamping);
    tResidual->setIsotropicLinearElasticMaterial(tYoungsModulus, tPoissonRatio);

    std::shared_ptr<Plato::AdjointStructuralDynamicsResidual<JacobianU, SIMP, Plato::HyperbolicTangentProjection>> tJacobianState;
    tJacobianState = std::make_shared<Plato::AdjointStructuralDynamicsResidual<JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tResidual->setMaterialDensity(tDensity);
    tJacobianState->setMassPropDamping(tMassPropDamping);
    tJacobianState->setStiffPropDamping(tStiffPropDamping);
    tJacobianState->setIsotropicLinearElasticMaterial(tYoungsModulus, tPoissonRatio);

    // ALLOCATE VECTOR FUNCTION
    std::shared_ptr<VectorFunction<Plato::StructuralDynamics<tSpaceDim>>> tVectorFunction =
        std::make_shared<VectorFunction<Plato::StructuralDynamics<tSpaceDim>>>(*tMesh, tDataMap);
    tVectorFunction->allocateResidual(tResidual, tJacobianState);

    // ALLOCATE ADJOINT STRUCTURAL DYNAMICS PROBLEM
    Plato::StructuralDynamicsProblem<Plato::StructuralDynamics<tSpaceDim>> tProblem(*tMesh, tVectorFunction);

    // SET DIRICHLET BOUNDARY CONDITIONS
    Plato::Scalar tValue = 0;
    auto tNumDofsPerNode = 2*tSpaceDim;
    Omega_h::LOs tCoordsX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    auto tNumDirichletDofs = tNumDofsPerNode*tCoordsX0.size();
    Plato::ScalarVector tDirichletValues("DirichletValues", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("DirichletDofs", tNumDirichletDofs);
    PlatoUtestHelpers::set_dirichlet_boundary_conditions(tNumDofsPerNode, tValue, tCoordsX0, tDirichletDofs, tDirichletValues);
    tProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // SET FREQUENCY    
    std::vector<Plato::Scalar> tFreq = { 5 };
    tProblem.setFrequencyArray(tFreq);

    // SET EXTERNAL FORCE
    auto tNumDofs = tVectorFunction->size();
    auto tNumDofsGold = tNumDofsPerNode * tMesh->nverts();
    TEST_EQUALITY(tNumDofsGold, tNumDofs);
    Plato::ScalarVector tPointLoad("PointLoad", tNumDofs);

    Plato::ScalarMultiVector tValues("Values", 2, tSpaceDim);
    auto tHostValues = Kokkos::create_mirror(tValues);
    tHostValues(0,0) = 0;    tHostValues(1,0) = 0;
    tHostValues(0,1) = -1e5; tHostValues(1,1) = -1e5;
    Kokkos::deep_copy(tValues, tHostValues);

    auto tTopOrdinalIndex = 0;
    auto tNodeOrdinalsX1 = PlatoUtestHelpers::get_2D_boundary_nodes_x1(*tMesh);
    PlatoUtestHelpers::set_point_load(tTopOrdinalIndex, tNodeOrdinalsX1, tValues, tPointLoad);
    tProblem.setExternalForce(tPointLoad);

    //SOLVE ELASTODYNAMICS ADJOINT PROBLEM
    auto tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Kokkos::deep_copy(tControl, static_cast<Plato::Scalar>(1));
    tProblem.setMaxNumIterationsAmgX(500);
    auto tSolution = tProblem.solution(tControl);
}

} //namespace PlatoUnitTests
