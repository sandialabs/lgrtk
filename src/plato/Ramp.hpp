#ifndef RAMP_HPP
#define RAMP_HPP

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
class RAMP
/******************************************************************************/
{
    double aPenaltyParam, aMinValue;

public:
    RAMP() :
        aPenaltyParam(3),
        aMinValue(0)
    {
    }
    RAMP(Teuchos::ParameterList & aParamList)
    {
        aPenaltyParam = aParamList.get<Plato::Scalar>("Exponent");
        aMinValue = aParamList.get<Plato::Scalar>("Minimum Value", 0.0);
    }
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()( ScalarType aInput ) const
    {
        ScalarType tOutput = aMinValue + (static_cast<ScalarType>(1.0) - aMinValue)*aInput
                /(static_cast<ScalarType>(1.0) + aPenaltyParam*(static_cast<ScalarType>(1.0) - aInput));
        return (tOutput);
    }
};

#endif
