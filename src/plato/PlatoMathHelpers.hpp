/*
 * PlatoMathHelpers.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOMATHHELPERS_HPP_
#define PLATOMATHHELPERS_HPP_

#include <cassert>

#include <Kokkos_Macros.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Device only function used to compare two values (conditional values)
 * between themselves and return the decision (consequent value). The conditional
 * expression evaluated in this function is defined as if(X > Y) A = B.
 * @param [in] aConditionalValOne conditional value given by X
 * @param [in] aConditionalValTwo conditional value given by Y
 * @param [in] aConsequentValOne consequent value given by A
 * @param [in] aConsequentValTwo consequent value given by B
 * @return result/decision
**********************************************************************************/
DEVICE_TYPE inline Plato::Scalar
conditional_expression(const Plato::Scalar & aX,
                       const Plato::Scalar & aY,
                       const Plato::Scalar & aA,
                       const Plato::Scalar & aB)
{
    auto tConditionalExpression = aX - aY - static_cast<Plato::Scalar>(1.0);
    tConditionalExpression = exp(tConditionalExpression);
    Plato::OrdinalType tCoeff = fmin(static_cast<Plato::Scalar>(1.0), tConditionalExpression);
    Plato::Scalar tScalarCoeff = tCoeff;
    auto tOutput = tScalarCoeff * aB + (static_cast<Plato::Scalar>(1.0) - tScalarCoeff) * aA;
    return (tOutput);
}
// function conditional_expression

/******************************************************************************//**
 * @brief Fill host 1D container with random numbers
 * @param [in] aLowerBound lower bounds on random numbers
 * @param [in] aUpperBound upper bounds on random numbers
 * @param [in] aOutput output 1D container
**********************************************************************************/
template<typename VecType>
inline void random(const Plato::Scalar & aLowerBound, const Plato::Scalar & aUpperBound, VecType & aOutput)
{
    unsigned int tRANDOM_SEED = 1;
    std::srand(tRANDOM_SEED);
    const Plato::OrdinalType tSize = aOutput.size();
    for(Plato::OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
    {
        const Plato::Scalar tRandNum = static_cast<Plato::Scalar>(std::rand()) / static_cast<Plato::Scalar>(RAND_MAX);
        aOutput(tIndex) = aLowerBound + ( (aUpperBound - aLowerBound) * tRandNum);
    }
}
// function random

/******************************************************************************//**
 * @brief Compute inner product
 * @param [in] aVec1 1D container
 * @param [in] aVec2 1D container
 * @return inner product
**********************************************************************************/
template<typename VecOneT, typename VecTwoT>
inline Plato::Scalar dot(const VecOneT & aVec1, const VecTwoT & aVec2)
{
    assert(aVec2.size() == aVec1.size());
    Plato::Scalar tOutput = 0.;
    const Plato::OrdinalType tSize = aVec1.size();
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tSize),
                            LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex, Plato::Scalar & aSum)
    {
        aSum += aVec1(aIndex) * aVec2(aIndex);
    }, tOutput);
    return (tOutput);
}
// function dot

/******************************************************************************//**
 * @brief Compute the norm/length of a vector
 * @param [in] aVector 1D container
 * @return norm/length
**********************************************************************************/
template<typename VecOneT>
inline Plato::Scalar norm(const VecOneT & aVector)
{
    const Plato::Scalar tDot = Plato::dot(aVector, aVector);
    const Plato::Scalar tOutput = std::sqrt(tDot);
    return (tOutput);
}
// function norm

/******************************************************************************//**
 * @brief Set all the elements to a scalar value
 * @param [in] aInput scalar value
 * @param [out] aVector 1D container
**********************************************************************************/
template<typename VectorT>
inline void fill(const Plato::Scalar & aInput, const VectorT & aVector)
{
    Plato::OrdinalType tNumLocalVals = aVector.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aVector(aOrdinal) = aInput;
    }, "fill vector");
}
// function fill

/******************************************************************************//**
 * @brief Copy input 1D container into output 1D container
 * @param [in] aInput 1D container
 * @param [out] aOutput 1D container
**********************************************************************************/
template<typename VecOneT, typename VecTwoT>
inline void copy(const VecOneT & aInput, const VecTwoT & aOutput)
{
    assert(aInput.size() == aOutput.size());
    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) = aInput(aOrdinal);
    }, "copy vector");
}
// function copy

/******************************************************************************//**
 * @brief Scale all the elements by input scalar value
 * @param [in] aInput scalar value
 * @param [out] aOutput 1D container
**********************************************************************************/
template<typename VecT>
inline void scale(const Plato::Scalar & aInput, const VecT & aVector)
{
    int tNumLocalVals = aVector.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aVector(aOrdinal) *= aInput;
    }, "scale vector");
}
// function scale

/******************************************************************************//**
 * @brief Update elements of B with scaled values of A, /f$ B = B + alpha*A /f$
 * @param [in] aAlpha multiplier of 1D container A
 * @param [in] aInput input 1D container
 * @param [out] aOutput output 1D container
**********************************************************************************/
template<typename VecT>
inline void axpy(const Plato::Scalar & aAlpha, const VecT & aInput, const VecT & aOutput)
{
    assert(aInput.size() == aOutput.size());
    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) += aAlpha * aInput(aOrdinal);
    }, "Plato::axpy");
}
// function axpy

/******************************************************************************//**
 * @brief Update elements of B with scaled values of A, /f$ B = beta*B + alpha*A /f$
 * @param [in] aAlpha multiplier of 1D container A
 * @param [in] aInput input 1D container
 * @param [in] aBeta multiplier of 1D container B
 * @param [out] aOutput output 1D container
**********************************************************************************/
template<typename VecT>
void update(const Plato::Scalar & aAlpha, const VecT & aInput, const Plato::Scalar & aBeta, const VecT & aOutput)
{
    assert(aInput.size() == aOutput.size());
    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) = aAlpha * aInput(aOrdinal) + aBeta * aOutput(aOrdinal);
    }, "update vector");
}
// function update

/******************************************************************************//**
 * @brief Reduced operation: sum all the elements in input array and return local sum
 * @param [in] aInput 1D container
 * @param [out] aOutput local sum
**********************************************************************************/
template<typename VecT, typename ScalarT>
void local_sum(const VecT & aInput, ScalarT & aOutput)
{
    ScalarT tOutput = 0.0;
    const Plato::OrdinalType tNumLocalElems = aInput.size();
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumLocalElems), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal, ScalarT & aLocalSum)
    {
      aLocalSum += aInput(aCellOrdinal);
    }, tOutput);
    aOutput = tOutput;
}
// function local_sum

/******************************************************************************//**
 * @brief Matrix times vector plus vector
 * @param [in] aMatrix multiplier of 1D container A
 * @param [in] aInput input 1D container
 * @param [out] aOutput output 1D container
**********************************************************************************/
template<typename ScalarT>
void MatrixTimesVectorPlusVector(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                                 const Plato::ScalarVectorT<ScalarT> & aInput,
                                 const Plato::ScalarVectorT<ScalarT> & aOutput)
{
    if(aMatrix->isBlockMatrix())
    {
        auto tNodeRowMap = aMatrix->rowMap();
        auto tNodeColIndices = aMatrix->columnIndices();
        auto tBlockRowSize = aMatrix->blockSizeRow();
        auto tBlockColSize = aMatrix->blockSizeCol();
        auto tEntries = aMatrix->entries();
        auto tNumNodeRows = tNodeRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodeRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeRowOrdinal)
        {
            auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
            auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
            for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
            {
                auto tNodeColumnIndex = tNodeColIndices(tCrsIndex);

                auto tFromDofColIndex = tBlockRowSize*tNodeColumnIndex;
                auto tToDofColIndex = tFromDofColIndex + tBlockRowSize;

                auto tFromDofRowIndex = tBlockColSize*aNodeRowOrdinal;
                auto tToDofRowIndex = tFromDofRowIndex + tBlockColSize;

                auto tMatrixEntryIndex = tBlockColSize*tBlockRowSize*tCrsIndex;
                for ( auto tDofRowIndex = tFromDofRowIndex; tDofRowIndex < tToDofRowIndex; tDofRowIndex++ )
                {
                    ScalarT tSum = 0.0;
                    for ( auto tDofColIndex = tFromDofColIndex; tDofColIndex < tToDofColIndex; tDofColIndex++ )
                    {
                        tSum += tEntries(tMatrixEntryIndex) * aInput(tDofColIndex);
                        tMatrixEntryIndex += 1;
                    }
                    aOutput(tDofRowIndex) += tSum;
                }
            }
        }, "BlockMatrix * Vector_a + Vector_b");
    }
    else
    {
        auto tRowMap = aMatrix->rowMap();
        auto tColIndices = aMatrix->columnIndices();
        auto tEntries = aMatrix->entries();
        auto tNumRows = tRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRowOrdinal)
        {
            auto tRowStart = tRowMap(aRowOrdinal);
            auto tRowEnd = tRowMap(aRowOrdinal + 1);
            ScalarT tSum = 0.0;
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto tColumnIndex = tColIndices(tEntryIndex);
                tSum += tEntries(tEntryIndex) * aInput(tColumnIndex);
            }
            aOutput(aRowOrdinal) += tSum;
        },"Matrix * Vector_a + Vector_b");
    }
}
// function MatrixTimesVectorPlusVector

/******************************************************************************//**
 * @brief Compute the global maximum element in range.
 * @param [in] aInput array of elements
 * @param [out] aOutput maximum element
**********************************************************************************/
template<typename VecT, typename ScalarT>
void max(const VecT & aInput, ScalarT & aOutput)
{
    assert(aInput.size() > static_cast<OrdinalType>(0));
    const OrdinalType tSize = aInput.size();
    //const ScalarT* tInputData = aInput.data();
    aOutput = 0.0;

    Kokkos::Max<ScalarT> tMaxReducer(aOutput);
    Kokkos::parallel_reduce("KokkosReductionOperations::max",
                            Kokkos::RangePolicy<>(0, tSize),
                            KOKKOS_LAMBDA(const OrdinalType & aIndex, ScalarT & aValue){
        tMaxReducer.join(aValue, aInput[aIndex]);
    }, tMaxReducer);
}

/******************************************************************************//**
 * @brief Compute the global minimum element in range.
 * @param [in] aInput array of elements
 * @param [out] aOutput minimum element
**********************************************************************************/
template<typename VecT, typename ScalarT>
void min(const VecT & aInput, ScalarT & aOutput)
{
    assert(aInput.size() > static_cast<OrdinalType>(0));
    const OrdinalType tSize = aInput.size();
    //const ScalarT* tInputData = aInput.data();
    aOutput = 0.0;

    Kokkos::Min<ScalarT> tMinReducer(aOutput);
    Kokkos::parallel_reduce("KokkosReductionOperations::min",
                            Kokkos::RangePolicy<>(0, tSize),
                            KOKKOS_LAMBDA(const OrdinalType & aIndex, ScalarT & aValue){
        tMinReducer.join(aValue, aInput[aIndex]);
    }, tMinReducer);
}

/******************************************************************************//**
 * @brief Extract a sub array
 * @param [in] aFromVector
 * @param [out] aToVector
 *
 * aToVector(i) = aFromVector(i*NumStride+NumOffset)
 *
**********************************************************************************/
template<int NumStride, int NumOffset>
inline void extract(const Plato::ScalarVector& aFromVector, Plato::ScalarVector& aToVector)
{
    auto tNumRows = aToVector.extent(0);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & ordinal)
    {
        aToVector(ordinal) = aFromVector(ordinal*NumStride + NumOffset);
    }, "extract");
}
// function extract


} // namespace Plato

#endif /* PLATOMATHHELPERS_HPP_ */
