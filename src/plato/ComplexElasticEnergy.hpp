/*
 * ComplexElasticEnergy.hpp
 *
 *  Created on: Apr 25, 2018
 */

#ifndef COMPLEXELASTICENERGY_HPP_
#define COMPLEXELASTICENERGY_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Compute elastic energy, i.e. \epsilon_e : \sigma_e, where, \epsilon_e is the
 *  strain tensor for the e-th cell and \sigma_e stress tensor for the e-th cell.
 *
 *  Function Description: Given the stress and strain Voigt tensors, compute the
 *  elastic energy for structural dynamics applications. Assumes single point
 *  integration.
*/
/******************************************************************************/
template<Plato::OrdinalType NumVoigtTerms>
class ComplexElasticEnergy
{
public:
    ComplexElasticEnergy(){}
    ~ComplexElasticEnergy(){}

    template<typename OutputScalarType, typename Tensor1ScalarType, typename Tensor2ScalarType>
    DEVICE_TYPE inline void
    operator()( const Plato::OrdinalType & aCellOrdinal,
                const Plato::ScalarArray3DT<Tensor1ScalarType> & aStress,
                const Plato::ScalarArray3DT<Tensor2ScalarType> & aStrain,
                const Plato::ScalarVectorT<OutputScalarType> & aOutput) const
    {
        assert(aStress.size() == aStrain.size());

        aOutput(aCellOrdinal) = 0.0;
        const Plato::OrdinalType tComplexSpaceDim = aStress.extent(1);
        for(Plato::OrdinalType tComplexDim = 0; tComplexDim < tComplexSpaceDim; tComplexDim++)
        {
            for(Plato::OrdinalType tIndex = 0; tIndex < NumVoigtTerms; tIndex++)
            {
                aOutput(aCellOrdinal) += aStress(aCellOrdinal, tComplexDim, tIndex)
                        * aStrain(aCellOrdinal, tComplexDim, tIndex);
            }
        }
    }
};
// class ComplexDotProduct

} // namespace Plato

#endif /* COMPLEXELASTICENERGY_HPP_ */
