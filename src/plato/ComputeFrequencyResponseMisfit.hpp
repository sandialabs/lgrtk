/*
 * ComputeFrequencyResponseMisfit.hpp
 *
 *  Created on: May 24, 2018
 */

#ifndef SRC_PLATO_COMPUTEFREQUENCYRESPONSEMISFIT_HPP_
#define SRC_PLATO_COMPUTEFREQUENCYRESPONSEMISFIT_HPP_

#include "plato/SimplexStructuralDynamics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Compute frequency response function misfit, i.e. \delta = \frac{1}{2}
 *  (u_i^s - u_i^e)^2, where, u_i^s is the trial state value at the i-th degree
 *  of freedom and u_i^e is the experimental state value at the i-th degree
 *  of freedom for a given cell.
 *
 *  Function Description: Given the trial and experimental state values,
 *  compute the misfit. Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class ComputeFrequencyResponseMisfit : public Plato::SimplexStructuralDynamics<SpaceDim, NumControls>
{
private:
    using Plato::SimplexStructuralDynamics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexStructuralDynamics<SpaceDim>::m_numNodesPerCell;

public:
    /*************************************************************************/
    ComputeFrequencyResponseMisfit()
    /*************************************************************************/
    {
    }

    /*************************************************************************/
    ~ComputeFrequencyResponseMisfit()
    /*************************************************************************/
    {
    }

    /*************************************************************************/
    template<typename StateScalarType, typename OutputScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVector & aExpStates,
               const Plato::ScalarMultiVectorT<StateScalarType> & aTrialStates,
               const Plato::ScalarVectorT<OutputScalarType> & aOutput) const
    /*************************************************************************/
    {
        assert(aExpStates.size() == aTrialStates.size());
        assert(aExpStates.extent(0) == aOutput.size());

        aOutput(aCellOrdinal) = 0.0;
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < m_numNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
            {
                Plato::OrdinalType tRealDofIndex = (m_numDofsPerNode * tNodeIndex) + tDimIndex;
                OutputScalarType tRealMisfit = aTrialStates(aCellOrdinal, tRealDofIndex) - aExpStates(aCellOrdinal, tRealDofIndex);
                Plato::OrdinalType tImagDofIndex = (m_numDofsPerNode * tNodeIndex) + SpaceDim + tDimIndex;
                OutputScalarType tImagMisfit = aTrialStates(aCellOrdinal, tImagDofIndex) - aExpStates(aCellOrdinal, tImagDofIndex);
                aOutput(aCellOrdinal) += (tRealMisfit * tRealMisfit) + (tImagMisfit * tImagMisfit);
            }
        }
        aOutput(aCellOrdinal) = static_cast<Plato::Scalar>(0.5) * aOutput(aCellOrdinal);
    }
};
// class ComputeFrequencyResponseMisfit

} // namespace Plato

#endif /* SRC_PLATO_COMPUTEFREQUENCYRESPONSEMISFIT_HPP_ */
