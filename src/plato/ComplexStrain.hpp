/*
 * ComplexStrain.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef COMPLEXSTRAIN_HPP_
#define COMPLEXSTRAIN_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! ComplexStrain Functor.
*
*   Computes linear strain tensor for structural dynamics applications.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode>
class ComplexStrain
{
public:
    /******************************************************************************/
    ComplexStrain()
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    ~ComplexStrain()
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    template<typename StrainScalarType, typename DispScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<DispScalarType> & aState,
                                       const Plato::ScalarArray3DT<GradientScalarType> & aGradient,
                                       const Plato::ScalarArray3DT<StrainScalarType> & aStrain) const
    /******************************************************************************/
    {
        const Plato::OrdinalType tComplexSpaceDim = aStrain.extent(1);
        const Plato::OrdinalType tNumNodesPerCell = aGradient.extent(1);

        for(Plato::OrdinalType tComplexIndex = 0; tComplexIndex < tComplexSpaceDim; tComplexIndex++)
        {
            Plato::OrdinalType tVoigtTerm=0;
            for(Plato::OrdinalType tDimIndex=0; tDimIndex<SpaceDim; tDimIndex++)
            {
                aStrain(aCellOrdinal, tComplexIndex, tVoigtTerm)=0.0;
                for( Plato::OrdinalType tNodeIndex=0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tCellDof = (tNodeIndex * NumDofsPerNode) + ((SpaceDim * tComplexIndex) + tDimIndex);
                    aStrain(aCellOrdinal, tComplexIndex, tVoigtTerm) += aState(aCellOrdinal, tCellDof)
                            * aGradient(aCellOrdinal, tNodeIndex, tDimIndex);
                }
                tVoigtTerm++;
            }

            for(Plato::OrdinalType tIndexJ=SpaceDim-1; tIndexJ>=1; tIndexJ--)
            {
                for (Plato::OrdinalType tIndexI=tIndexJ-1; tIndexI>=0; tIndexI--)
                {
                    for( Plato::OrdinalType tNodeIndex=0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
                    {
                        Plato::OrdinalType tCellDofI = (tNodeIndex * NumDofsPerNode) + ((SpaceDim * tComplexIndex) + tIndexI);
                        Plato::OrdinalType tCellDofJ = (tNodeIndex * NumDofsPerNode) + ((SpaceDim * tComplexIndex) + tIndexJ);
                        aStrain(aCellOrdinal, tComplexIndex, tVoigtTerm) +=
                                (aState(aCellOrdinal,tCellDofJ) * aGradient(aCellOrdinal,tNodeIndex,tIndexI)
                                + aState(aCellOrdinal,tCellDofI) * aGradient(aCellOrdinal,tNodeIndex,tIndexJ));
                    }
                    tVoigtTerm++;
                }
            }
        }
    }
};
// class ComplexStrain

} // namespace Plato

#endif /* COMPLEXSTRAIN_HPP_ */
