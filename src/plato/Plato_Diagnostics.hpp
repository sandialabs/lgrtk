/*
 * Plato_Diagnostics.hpp
 *
 *  Created on: Feb 11, 2019
 */

#pragma once

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/ScalarFunctionBase.hpp"
#include "plato/LocalVectorFunctionInc.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Test partial derivative with respect to the control variables
 * @param [in] aMesh mesh database
 * @param [in] aCriterion scalar function (i.e. scalar criterion) interface
 * @param [in] aSuperscriptLowerBound lower bound on the superscript used to compute the step (e.g. \f$10^{lb}/$f
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_control(Omega_h::Mesh & aMesh,
                                 Plato::AbstractScalarFunction<EvaluationType> & aCriterion,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = SimplexPhysics::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = SimplexPhysics::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<SimplexPhysics> tWorksetBase(aMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // FINITE DIFFERENCE TEST
    aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
    constexpr Plato::OrdinalType tNumControlFields = 1;
    Plato::ScalarVector tPartialZ("objective partial control", tNumVerts);
    Plato::VectorEntryOrdinal<tSpaceDim, tNumControlFields> tControlEntryOrdinal(&aMesh);
    Plato::assemble_scalar_gradient<tNodesPerCell>(tNumCells, tControlEntryOrdinal, tResultWS, tPartialZ);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for(Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_control

/******************************************************************************//**
 * @brief Test partial derivative with respect to the state variables
 * @param [in] aMesh mesh database
 * @param [in] aCriterion scalar function (i.e. scalar criterion) interface
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_state(Omega_h::Mesh & aMesh, Plato::AbstractScalarFunction<EvaluationType> & aCriterion)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = SimplexPhysics::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = SimplexPhysics::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(aMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // finite difference
    aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
    Plato::ScalarVector tPartialU("objective partial state", tTotalNumDofs);
    Plato::VectorEntryOrdinal<tSpaceDim, tDofsPerNode> tStateEntryOrdinal(&aMesh);
    Plato::assemble_vector_gradient<tNodesPerCell, tDofsPerNode>(tNumCells, tStateEntryOrdinal, tResultWS, tPartialU);

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    Plato::ScalarVector tTrialState("trial state", tTotalNumDofs);
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_state


template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_control(Omega_h::Mesh & aMesh,
                                 Plato::ScalarFunctionBase & aScalarFuncBase,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);

    // FINITE DIFFERENCE TEST
    Plato::ScalarVector tPartialZ = aScalarFuncBase.gradient_z(tState, tControl, 0.0);


    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for(Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(tState, tTrialControl, 0.0);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(tState, tTrialControl, 0.0);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(tState, tTrialControl, 0.0);

        Plato::update(1.0, tControl, 0.0, tTrialControl);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(tState, tTrialControl, 0.0);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_control

/******************************************************************************//**
 * @brief Test partial derivative with respect to the state variables
 * @param [in] aMesh mesh database
 * @param [in] aCriterion scalar function (i.e. scalar criterion) interface
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_state(Omega_h::Mesh & aMesh, Plato::ScalarFunctionBase & aScalarFuncBase)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);

    Plato::ScalarVector tPartialU = aScalarFuncBase.gradient_u(tState, tControl, 0.0);

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    Plato::ScalarVector tTrialState("trial state", tTotalNumDofs);
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(tTrialState, tControl, 0.0);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(tTrialState, tControl, 0.0);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(tTrialState, tControl, 0.0);

        Plato::update(1.0, tState, 0.0, tTrialState);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(tTrialState, tControl, 0.0);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_state

template<typename MatrixScalarType>
inline Plato::ScalarVector worksetMatrixVectorMultiply(
                                        const Plato::ScalarMultiVectorT<MatrixScalarType> & aMatrix,
                                        const Plato::ScalarVector & aVector)
{
    const Plato::OrdinalType tNumCells            = aMatrix.extent(0);
    const Plato::OrdinalType tNumLocalDofsPerCell = aMatrix.extent(1);
    const Plato::OrdinalType tVectorSize          = aVector.extent(0);

    Plato::ScalarVector tResult("result", tVectorSize);

    printf("%d %d %d\n", tNumCells, tNumLocalDofsPerCell, tVectorSize);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), 
                         LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::OrdinalType tStartingRowIndex = aCellOrdinal * tNumLocalDofsPerCell;
        for (Plato::OrdinalType tRow = 0; tRow < tNumLocalDofsPerCell; ++tRow)
        {
            tResult(tStartingRowIndex + tRow) = 0.0;
            for (Plato::OrdinalType tColumn = 0; tColumn < tNumLocalDofsPerCell; ++tColumn)
            {
                Plato::Scalar tValue = 
                             aMatrix(aCellOrdinal, tColumn).dx(tRow) * aVector(tStartingRowIndex + tRow);
                tResult(tStartingRowIndex + tRow) += tValue;
            }
        }
    }, "matrix vector multiply");
    return tResult;
}

/******************************************************************************//**
 * @brief Test partial derivative with respect to the local state variables
 * @param [in] aMesh mesh database
 * @param [in] aCriterion scalar function (i.e. scalar criterion) interface
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_local_state(Omega_h::Mesh & aMesh, 
                                     Plato::LocalVectorFunctionInc<SimplexPhysics> & aLocalVectorFuncInc)
{
    using StateT   = typename EvaluationType::StateScalarType;
    using LocalStateT   = typename EvaluationType::LocalStateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create global state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    auto tHostGlobalState = Kokkos::create_mirror(tGlobalState);
    Plato::random(1, 5, tHostGlobalState);
    Kokkos::deep_copy(tGlobalState, tHostGlobalState);

    // Create previous global state workset
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    auto tHostPrevGlobalState = Kokkos::create_mirror(tPrevGlobalState);
    Plato::random(1, 5, tHostPrevGlobalState);
    Kokkos::deep_copy(tPrevGlobalState, tHostPrevGlobalState);

    // Create local state workset
    const Plato::OrdinalType tTotalNumLocalDofs = tNumCells * tLocalDofsPerCell;
    Plato::ScalarVector tLocalState("Local State", tTotalNumLocalDofs);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Plato::random(1, 5, tHostLocalState);
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    // Create previous local state workset
    Plato::ScalarVector tPrevLocalState("Previous Local State", tTotalNumLocalDofs);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    Plato::random(1, 5, tHostPrevLocalState);
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarMultiVectorT<LocalStateT> tPartialC = 
                aLocalVectorFuncInc.gradient_c(tGlobalState, tPrevGlobalState,
                                                tLocalState, tPrevLocalState,
                                                   tControl, 0.0);

    Plato::ScalarVector tStep("local state step", tTotalNumLocalDofs); 
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    // Plato::ScalarVector tGradientDotStep = worksetMatrixVectorMultiply(tPartialC, tStep);

    std::cout << std::right << std::setw(14) << "\nStep Size" 
               << std::setw(19) << "abs(Error)" << std::endl;
printf("YAY\n\n\n");
    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    Plato::ScalarVector tTrialLocalState("trial local state", tTotalNumLocalDofs);

    Plato::ScalarVector tErrorVector("error vector", tTotalNumLocalDofs);

    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::update(tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueOne = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tTrialLocalState, tPrevLocalState,
                                                                        tControl, 0.0);

        Plato::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::update(-tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueTwo = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tTrialLocalState, tPrevLocalState,
                                                                        tControl, 0.0);

        Plato::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::update(2.0 * tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueThree = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                          tTrialLocalState, tPrevLocalState,
                                                                          tControl, 0.0);

        Plato::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::update(-2.0 * tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueFour = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                         tTrialLocalState, tPrevLocalState,
                                                                         tControl, 0.0);

        // Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumLocalDofs), 
        //                          LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        // {
        //     Plato::Scalar tValuePlus1Eps  = tVectorValueOne(aDofOrdinal);
        //     Plato::Scalar tValueMinus1Eps = tVectorValueTwo(aDofOrdinal);
        //     Plato::Scalar tValuePlus2Eps  = tVectorValueThree(aDofOrdinal);
        //     Plato::Scalar tValueMinus2Eps = tVectorValueFour(aDofOrdinal);

        //     Plato::Scalar tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
        //                                - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
        //     Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        //     Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        //     Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep(aDofOrdinal));

        //     tErrorVector(aDofOrdinal) = tAppxError;
        // }, "compute error");

        // Plato::Scalar tL1Error = 0.0;
        // Plato::local_sum(tErrorVector, tL1Error);

        // std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) 
        //           << tEpsilon << std::setw(19)
        //           << tL1Error << std::endl;
    }
}
// function test_partial_local_state

}
