/*
 * InertialForces.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef INERTIALFORCES_HPP_
#define INERTIALFORCES_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! InertialForces Functor.
*
*   Evaluates cell inertial forces.
*/
/******************************************************************************/
class InertialForces
{
public:
    /******************************************************************************/
    explicit InertialForces(const Plato::Scalar & aDensity) :
            mDensity(aDensity)
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    ~InertialForces()
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    template<typename DispScalarType, typename ForceScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & tBasisFunctions,
                                       const Plato::ScalarMultiVectorT<DispScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ForceScalarType> & aInertialForces) const
    /******************************************************************************/
    {
        const Plato::OrdinalType tNumNodesPerCell = tBasisFunctions.size();
        const Plato::OrdinalType tNumDofsPerNode = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (tNumDofsPerNode * tNodeIndex) + tDofIndex;
                aInertialForces(aCellOrdinal, tMyDofIndex) = tBasisFunctions(tNodeIndex)
                        * aStateValues(aCellOrdinal, tDofIndex) * aCellVolume(aCellOrdinal) * mDensity;
            }
        }
    }

private:
    Plato::Scalar mDensity; /* Material Density */
};
// class InertialForces

} // namespace Plato

#endif /* INERTIALFORCES_HPP_ */
