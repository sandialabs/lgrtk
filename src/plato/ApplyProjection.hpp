/*
 * ApplyProjection.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef APPLYPROJECTION_HPP_
#define APPLYPROJECTION_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Projection Functor.

 Given a set of cell control values, apply Heaviside projection operator.
 Assumes single point integration.
 */
/******************************************************************************/
template<class ProjectionFunction>
class ApplyProjection
{
public:
    /******************************************************************************/
    ApplyProjection() :
            mProjectionFunction()
    {
    }
    
    /******************************************************************************/
    explicit ApplyProjection(const ProjectionFunction & aProjectionFunction) :
            mProjectionFunction(aProjectionFunction)
    {
    }
    
    /******************************************************************************/
    ~ApplyProjection()
    {
    }

    /**************************************************************************/
    template<typename WeightScalarType>
    DEVICE_TYPE inline WeightScalarType
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<WeightScalarType> & aControl) const
    /**************************************************************************/
    {
        WeightScalarType tCellDensity = 0.0;
        const Plato::OrdinalType tRangePolicy = aControl.extent(1);
        for(Plato::OrdinalType tIndex = 0; tIndex < tRangePolicy; tIndex++)
        {
            tCellDensity += aControl(aCellOrdinal, tIndex);
        }
        tCellDensity = (tCellDensity / tRangePolicy);
        tCellDensity = mProjectionFunction.apply(tCellDensity);

        return (tCellDensity);
    }

private:
    ProjectionFunction mProjectionFunction;
};
// class ApplyProjection

} // namespace Plato

#endif /* APPLYPROJECTION_HPP_ */
