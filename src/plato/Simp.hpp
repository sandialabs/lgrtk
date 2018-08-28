#ifndef SIMP_HPP
#define SIMP_HPP

#include <stdio.h>

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
class SIMP
/******************************************************************************/
{
    double mPenaltyParam, mMinValue;

public:
    /**************************************************************************/
    SIMP(Plato::Scalar aPenalty = 3.0, Plato::Scalar aMinValue = 0.0) :
            mPenaltyParam(aPenalty),
            mMinValue(aMinValue)
    /**************************************************************************/
    {
    }

    /**************************************************************************/
    SIMP(Teuchos::ParameterList & paramList)
    /**************************************************************************/
    {
        mPenaltyParam = paramList.get<Plato::Scalar>("Exponent", 3.0);
        mMinValue = paramList.get<Plato::Scalar>("Minimum Value", 0.0);
    }

    /**************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()( ScalarType aInput ) const
    /**************************************************************************/
    {
        if (aInput != static_cast<ScalarType>(0.0))
        {
            return mMinValue + (static_cast<ScalarType>(1.0) - mMinValue) * pow(aInput, mPenaltyParam);
        }
        else
        {
            return mMinValue;
        }
    }
};

#endif
