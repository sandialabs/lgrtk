/*
 * StateValues.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef STATEVALUES_HPP_
#define STATEVALUES_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/*******************************************************************************//**
* 
* \brief StateValues Functor.
*
* Evaluate cell's nodal states at cubature points.
*
***********************************************************************************/
class StateValues
{
public:
    /***************************************************************************//**
    * 
    * \brief Constructor
    *
    *******************************************************************************/
    StateValues()
    {
    }
    
    /***************************************************************************//**
    * 
    * \brief Destructor 
    *
    *******************************************************************************/
    ~StateValues()
    {
    }

    /***************************************************************************//**
    * 
    * \brief Compute displacements values at cubature points 
    *
    * The displacements values are computed as follows: \hat{u}_{d} = \sum_{i=1}^{I}
    * \sum_\phi_{di}u_i, where \hat{u}_d is the d-th dimension displacement value, 
    * \phi_{di} is the matrix of basis functions matrix, J denotes the number of 
    * degrees of freedom and d is the spatial dimension free index.
    *
    * The input arguments are defined as:
    *
    *   \param aCellOrdinal      cell (i.e. element) ordinal 
    *   \param aBasisFunctions   cell interpolation functions
    *   \param aNodalCellStates  cell nodal states (e.g. displacement)  
    *   \param aStateValues      cell interpolated states at the cubature points 
    *
    *******************************************************************************/
    template<typename StateType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVector & aBasisFunctions,
                                       const Plato::ScalarMultiVectorT<StateType> & aNodalCellStates,
                                       const Plato::ScalarMultiVectorT<StateType> & aStateValues) const
    {
        assert(aStateValues.size() > static_cast<Plato::OrdinalType>(0));
        assert(aNodalCellStates.size() > static_cast<Plato::OrdinalType>(0));

        const Plato::OrdinalType tNumDofsPerNode = aStateValues.extent(1);
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
        {
            aStateValues(aCellOrdinal, tDofIndex) = 0.0;
        }

        const Plato::OrdinalType tNumNodesPerCell = aBasisFunctions.size();
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tCellDofIndex = (tNumDofsPerNode * tNodeIndex) + tDofIndex;
                aStateValues(aCellOrdinal, tDofIndex) += aBasisFunctions(tNodeIndex) * aNodalCellStates(aCellOrdinal, tCellDofIndex);
            }
        }
    }
};
// class StateValues

} // namespace Plato

#endif /* STATEVALUES_HPP_ */
