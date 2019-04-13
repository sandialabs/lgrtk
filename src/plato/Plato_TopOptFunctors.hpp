/*
 * Plato_TopOptFunctors.hpp
 *
 *  Created on: Feb 12, 2019
 */

#include "plato/PlatoStaticsTypes.hpp"

#pragma once

namespace Plato
{

/******************************************************************************//**
 * @brief Compute cell/element mass, /f$ \sum_{i=1}^{N} \[M\] \{z\} /f$, where
 * /f$ \[M\] /f$ is the mass matrix, /f$ \{z\} /f$ is the control vector and
 * /f$ N /f$ is the number of nodes.
 * @param [in] aCellOrdinal cell/element index
 * @param [in] aBasisFunc 1D container of cell basis functions
 * @param [in] aCellControls 2D container of cell controls
 * @return cell/element penalized mass
 **********************************************************************************/
template<Plato::OrdinalType CellNumNodes, typename ControlType>
DEVICE_TYPE inline ControlType
cell_mass(const Plato::OrdinalType & aCellOrdinal,
          const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunc,
          const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
{
    ControlType tCellMass = 0.0;
    for(Plato::OrdinalType tIndex_I = 0; tIndex_I < CellNumNodes; tIndex_I++)
    {
        ControlType tNodalMass = 0.0;
        for(Plato::OrdinalType tIndex_J = 0; tIndex_J < CellNumNodes; tIndex_J++)
        {
            auto tValue = aBasisFunc(tIndex_I) * aBasisFunc(tIndex_J) * aCellControls(aCellOrdinal, tIndex_J);
            tNodalMass += tValue;
        }
        tCellMass += tNodalMass;
    }
    return (tCellMass);
}

/******************************************************************************//**
 * @brief Compute average cell density
 * @param [in] aCellOrdinal cell/element index
 * @param [in] aNumControls number of controls
 * @param [in] aCellControls 2D container of cell controls
 * @return average density for this cell/element
 **********************************************************************************/
template<Plato::OrdinalType NumControls, typename ControlType>
DEVICE_TYPE inline ControlType
cell_density(const Plato::OrdinalType & aCellOrdinal,
             const Plato::ScalarMultiVectorT<ControlType> & aCellControls)
{
    ControlType tCellDensity = 0.0;
    for(Plato::OrdinalType tIndex = 0; tIndex < NumControls; tIndex++)
    {
        tCellDensity += aCellControls(aCellOrdinal, tIndex);
    }
    tCellDensity /= NumControls;
    return (tCellDensity);
}
// function cell_density

} // namespace Plato
