/*
 * ComplexLinearStress.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef COMPLEXLINEARSTRESS_HPP_
#define COMPLEXLINEARSTRESS_HPP_

#include <cassert>

#include <Omega_h_matrix.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! ComplexLinearStress Functor.
*
*   Computes linear stress tensor for structural dynamics applications.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumVoigtTerms>
class ComplexLinearStress
{
private:
    const Omega_h::Matrix<NumVoigtTerms,NumVoigtTerms> mCellStiffness;

public:
    /******************************************************************************/
    ComplexLinearStress(const Omega_h::Matrix<NumVoigtTerms,NumVoigtTerms> & aCellStiffness) :
        mCellStiffness(aCellStiffness)
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    ~ComplexLinearStress(){}
    /******************************************************************************/

    /******************************************************************************/
    template<typename StressScalarType, typename StrainScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarArray3DT<StrainScalarType> & aStrain,
               const Plato::ScalarArray3DT<StressScalarType> & aStress) const
    /******************************************************************************/
    {
        assert(aStress.extent(1) == aStrain.extent(1));
        assert(aStress.extent(2) == aStrain.extent(2));
        assert(static_cast<Plato::OrdinalType>(aStrain.extent(2)) == NumVoigtTerms);

        const Plato::OrdinalType tComplexDim = aStrain.extent(1);
        for(Plato::OrdinalType tComplexIndex = 0; tComplexIndex < tComplexDim; tComplexIndex++)
        {
            for(Plato::OrdinalType tVoigtIndexI = 0; tVoigtIndexI < NumVoigtTerms; tVoigtIndexI++)
            {
                aStress(aCellOrdinal, tComplexIndex, tVoigtIndexI) = 0.0;
                for(Plato::OrdinalType tVoigtIndexJ = 0; tVoigtIndexJ < NumVoigtTerms; tVoigtIndexJ++)
                {
                    aStress(aCellOrdinal, tComplexIndex, tVoigtIndexI) += aStrain(aCellOrdinal, tComplexIndex, tVoigtIndexJ)
                            * mCellStiffness(tVoigtIndexI, tVoigtIndexJ);
                }
            }
        }
    }
};
// class ComplexLinearStress

} // namespace Plato

#endif /* COMPLEXLINEARSTRESS_HPP_ */
