#ifndef RAMP_HPP
#define RAMP_HPP

#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
class RAMP
/******************************************************************************/
{
    Plato::Scalar mPenaltyParam;
    Plato::Scalar mMinValue;

public:
    RAMP() :
            mPenaltyParam(3),
            mMinValue(0)
    {
    }

    RAMP(Teuchos::ParameterList & aParamList)
    {
        mPenaltyParam = aParamList.get < Plato::Scalar > ("Exponent");
        mMinValue = aParamList.get < Plato::Scalar > ("Minimum Value", 0.0);
    }

    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()(ScalarType aInput) const
    {
        ScalarType tOutput = mMinValue
                + (static_cast<ScalarType>(1.0) - mMinValue) * aInput / (static_cast<ScalarType>(1.0)
                        + mPenaltyParam * (static_cast<ScalarType>(1.0) - aInput));
        return (tOutput);
    }
};

}

#endif
