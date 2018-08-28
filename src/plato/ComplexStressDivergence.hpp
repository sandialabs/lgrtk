/*
 * ComplexStressDivergence.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef COMPLEXSTRESSDIVERGENCE_HPP_
#define COMPLEXSTRESSDIVERGENCE_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! ComplexStressDivergence Functor.
*
*   Computes stress divergence for linear structural dynamics problems.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode>
class ComplexStressDivergence
{
private:
    Plato::OrdinalType mVoigtIndices[SpaceDim][SpaceDim];

public:

    /******************************************************************************/
    ComplexStressDivergence()
    /******************************************************************************/
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            mVoigtIndices[tDimIndex][tDimIndex] = tVoigtTerm++;
        }

        for (Plato::OrdinalType tIndexJ = SpaceDim - 1; tIndexJ >= 1; tIndexJ--)
        {
            for (Plato::OrdinalType tIndexI = tIndexJ - 1; tIndexI >= 0; tIndexI--)
            {
                mVoigtIndices[tIndexI][tIndexJ] = tVoigtTerm;
                mVoigtIndices[tIndexJ][tIndexI] = tVoigtTerm++;
            }
        }
    }

    /******************************************************************************/
    ~ComplexStressDivergence(){}
    /******************************************************************************/

    /******************************************************************************/
    template<typename ForcingScalarType, typename StressScalarType, typename GradientScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
               const Plato::ScalarArray3DT<GradientScalarType> & aGradient,
               const Plato::ScalarArray3DT<StressScalarType> & aStress,
               const Plato::ScalarMultiVectorT<ForcingScalarType> & aForce) const
    /******************************************************************************/
    {
        const Plato::OrdinalType tComplexSpaceDim = aStress.extent(1);
        const Plato::OrdinalType tNumNodesPerCell = aGradient.extent(1);
        for(Plato::OrdinalType tComplexIndex = 0; tComplexIndex < tComplexSpaceDim; tComplexIndex++)
        {
            for(Plato::OrdinalType tSpaceDimI = 0; tSpaceDimI < SpaceDim; tSpaceDimI++)
            {
                for( Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tCellDof = (tNodeIndex * NumDofsPerNode) + ((tComplexIndex * SpaceDim) + tSpaceDimI);
                    aForce(aCellOrdinal, tCellDof) = 0.0;
                    for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < SpaceDim; tDimIndexJ++)
                    {
                        const Plato::OrdinalType tVoigtIndex = mVoigtIndices[tSpaceDimI][tDimIndexJ];
                        aForce(aCellOrdinal, tCellDof) += aCellVolume(aCellOrdinal) * aStress(aCellOrdinal, tComplexIndex, tVoigtIndex)
                                * aGradient(aCellOrdinal, tNodeIndex, tDimIndexJ);
                    }
                }
            }
        }
    }
};
// class ComplexStressDivergence

} // namespace Plato

#endif /* COMPLEXSTRESSDIVERGENCE_HPP_ */
