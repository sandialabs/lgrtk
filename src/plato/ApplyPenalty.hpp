/*
 * ApplyPenalty.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef APPLYPENALTY_HPP_
#define APPLYPENALTY_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * 
 * \brief Apply weighting (i.e. penalty) to a quantity of interest.
 *
**********************************************************************************/
template<class PenaltyFunction>
class ApplyPenalty
{
public:
    /******************************************************************************//**
     * 
     * \brief Default constructor
     *
    **********************************************************************************/
    ApplyPenalty() :
            mPenaltyFunction()
    {
    }
    
    /******************************************************************************//**
     * 
     * \brief Constructor
     *
     * Input arguments
     *
     * @param [in] aPenaltyFunction penalty function used to penalize a quantity of interest
     *
    **********************************************************************************/
    explicit ApplyPenalty(const PenaltyFunction & aPenaltyFunction) :
            mPenaltyFunction(aPenaltyFunction)
    {
    }
    
    /******************************************************************************//**
     * 
     * \brief Destructor
     *
    **********************************************************************************/
    ~ApplyPenalty()
    {
    }

    /******************************************************************************//**
     * 
     * \brief Apply penalty to a quantity of interest.
     * 
     * Input and output arguments
     *
     * @param [in] aCellOrdinal cell (i.e. element) ordinal
     * @param [in] aCellDensity cell density scalar value
     * @param [in,out] aOutput 3D scalar array
     *
    **********************************************************************************/
    template<typename OutputScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const WeightScalarType & aCellDensity,
               const Plato::ScalarArray3DT<OutputScalarType> & aOutput) const
    {
        const Plato::OrdinalType tRangePolicyOne = aOutput.extent(1);
        const Plato::OrdinalType tRangePolicyTwo = aOutput.extent(2);
        for(Plato::OrdinalType tIndexOne = 0; tIndexOne < tRangePolicyOne; tIndexOne++)
        {
            for(Plato::OrdinalType tIndexTwo = 0; tIndexTwo < tRangePolicyTwo; tIndexTwo++)
            {
                aOutput(aCellOrdinal, tIndexOne, tIndexTwo) *= mPenaltyFunction(aCellDensity);
            }
        }
    }

    /******************************************************************************//**
     * 
     * \brief Apply penalty to a quantity of interest.
     * 
     * Input and output arguments
     *
     * @param [in] aCellOrdinal cell (i.e. element) ordinal
     * @param [in] aCellDensity cell density scalar value
     * @param [in,out] aOutput 2D scalar array
     *
    **********************************************************************************/
    template<typename OutputScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const WeightScalarType & aCellDensity,
               const Plato::ScalarMultiVectorT<OutputScalarType> & aOutput) const
    {
        const Plato::OrdinalType tRangePolicy = aOutput.extent(1);
        for(Plato::OrdinalType tIndex = 0; tIndex < tRangePolicy; tIndex++)
        {
            aOutput(aCellOrdinal, tIndex) *= mPenaltyFunction(aCellDensity);
        }
    }

private:
    PenaltyFunction mPenaltyFunction;
};
// Class ApplyPenalty

} // namespace Plato

#endif /* APPLYPENALTY_HPP_ */
