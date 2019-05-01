/*
 * InterpolateFromNodal.hpp
 *
 *  Created on: Feb 18, 2019
 */

#ifndef INTERPOLATE_FROM_NODAL_HPP_
#define INTERPOLATE_FROM_NODAL_HPP_

#include "plato/PlatoStaticsTypes.hpp"
#include "plato/Simplex.hpp"

namespace Plato
{

/***********************************************************************************
* 
* \brief InterpolateFromNodal Functor.
*
* Evaluate cell's nodal states at cubature points.
*
***********************************************************************************/
template<int SpaceDim, int NumDofsPerNode=SpaceDim, int DofOffset=0>
class InterpolateFromNodal : public Plato::Simplex<SpaceDim>
{
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;

public:
    /*******************************************************************************
    * 
    * \brief Constructor
    *
    *******************************************************************************/
    InterpolateFromNodal()
    {
    }
    
    /*******************************************************************************
    * 
    * \brief Destructor 
    *
    *******************************************************************************/
    ~InterpolateFromNodal()
    {
    }

    /*******************************************************************************
    * 
    * \brief Compute state values at cubature points 
    *
    * The state values are computed as follows: \hat{s} = \sum_{i=1}^{I}
    * \sum_\phi_{i} s_i, where \hat{s} is the state value, 
    * \phi_{i} is the array of basis functions.
    *
    * The input arguments are defined as:
    *
    *   \param aCellOrdinal      cell (i.e. element) ordinal 
    *   \param aBasisFunctions   cell interpolation functions
    *   \param aNodalCellStates  cell nodal states
    *   \param aStateValues      cell interpolated state at the cubature points 
    *
    *******************************************************************************/
    template<typename StateType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVector & aBasisFunctions,
                                       const Plato::ScalarMultiVectorT<StateType> & aNodalCellStates,
                                       const Plato::ScalarVectorT<StateType> & aStateValues) const
    {

        aStateValues(aCellOrdinal) = 0.0;

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < m_numNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tCellDofIndex = (NumDofsPerNode * tNodeIndex) + DofOffset;
            aStateValues(aCellOrdinal) += aBasisFunctions(tNodeIndex) * aNodalCellStates(aCellOrdinal, tCellDofIndex);
        }
    }
};
// class InterpolateFromNodal

} // namespace Plato

#endif
