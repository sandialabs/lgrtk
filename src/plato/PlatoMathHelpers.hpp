/*
 * PlatoMathHelpers.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOMATHHELPERS_HPP_
#define PLATOMATHHELPERS_HPP_

#include <cassert>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename ScalarT, typename VectorT>
void fill(const ScalarT & aInput, const VectorT & aVector)
/******************************************************************************/
{
    Plato::OrdinalType tNumLocalVals = aVector.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aVector(aOrdinal) = aInput;
    }, "fill vector");
} // function fill

/******************************************************************************/
template<typename ScalarT, typename VectorT>
void copy(const Plato::ScalarVectorT<ScalarT> & aInput, const VectorT & aOutput)
/******************************************************************************/
{
    assert(aInput.size() == aOutput.size());
    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) = aInput(aOrdinal);
    }, "copy vector");
} // function copy

/******************************************************************************/
template<typename ScalarT, typename VectorT>
void scale(const ScalarT & aInput, const VectorT & aVector)
/******************************************************************************/
{
    int tNumLocalVals = aVector.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aVector(aOrdinal) *= aInput;
    }, "scale vector");
} // function scale

/******************************************************************************/
template<typename ScalarT, typename VectorT>
void axpy(const ScalarT & aAlpha, const VectorT & aInput, const VectorT & aOutput)
/******************************************************************************/
{
    assert(aInput.size() == aOutput.size());
    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) += aAlpha * aInput(aOrdinal);
    }, "update vector");
} // function update

/******************************************************************************/
template<typename ScalarT, typename VectorT>
void update(const ScalarT & aAlpha, const VectorT & aInput, const ScalarT & aBeta, const VectorT & aOutput)
/******************************************************************************/
{
    assert(aInput.size() == aOutput.size());
    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) = aAlpha * aInput(aOrdinal) + aBeta * aOutput(aOrdinal);
    }, "update vector");
} // function update

/******************************************************************************/
template<typename ScalarT>
void MatrixTimesVectorPlusVector(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                                 const Plato::ScalarVectorT<ScalarT> & aVector_a,
                                 const Plato::ScalarVectorT<ScalarT> & aVector_b)
/******************************************************************************/
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
                        tSum += tEntries(tMatrixEntryIndex) * aVector_a(tDofColIndex);
                        tMatrixEntryIndex += 1;
                    }
                    aVector_b(tDofRowIndex) += tSum;
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
                tSum += tEntries(tEntryIndex) * aVector_a(tColumnIndex);
            }
            aVector_b(aRowOrdinal) += tSum;
        },"Matrix * Vector_a + Vector_b");
    }
} // function MatrixTimesVectorPlusVector

} // namespace Plato

#endif /* PLATOMATHHELPERS_HPP_ */
