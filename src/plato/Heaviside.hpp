#ifndef HEAVISIDE_HPP
#define HEAVISIDE_HPP

#ifndef T_PI
#define T_PI 3.1415926535897932385
#endif

#include <stdio.h>

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
class Heaviside
/******************************************************************************/
{
    Plato::Scalar mPenaltyParam;
    Plato::Scalar mRegLength;
    Plato::Scalar mMinValue;

public:
    /**************************************************************************/
    Heaviside(Plato::Scalar aPenalty = 1.0, Plato::Scalar aRegLength = 1.0, Plato::Scalar aMinValue = 0.0) :
            mPenaltyParam(aPenalty),
            mRegLength(aRegLength),
            mMinValue(aMinValue)
    /**************************************************************************/
    {
    }

    /**************************************************************************/
    Heaviside(Teuchos::ParameterList & paramList)
    /**************************************************************************/
    {
        mPenaltyParam = paramList.get<Plato::Scalar>("Exponent", 1.0);
        mRegLength = paramList.get<Plato::Scalar>("Regularization Length", 1.0);
        mMinValue = paramList.get<Plato::Scalar>("Minimum Value", 0.0);
    }

    /**************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()( ScalarType aInput ) const
    /**************************************************************************/
    {
        if (aInput <= -mRegLength)
        {
            return mMinValue;
        }
        else
        if (aInput >=  mRegLength)
        {
            return 1.0;
        }
        else
        {
            return mMinValue + (1.0 - mMinValue) * pow(1.0/2.0*(1.0 + sin(T_PI*aInput/(2.0*mRegLength))),mPenaltyParam);
        }
    }
};

#endif
