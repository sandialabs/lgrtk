/*
 * ComplexInertialEnergy.hpp
 *
 *  Created on: Apr 25, 2018
 */

#ifndef COMPLEXINERTIALENERGY_HPP_
#define COMPLEXINERTIALENERGY_HPP_

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Compute inertial energy, i.e. \rho\omega^2\langle u^h_e, u^h_e\rangle_{\Omega_e}.
 * Here, \rho is the material density, \omega is the nagular frequency and u^h_e is
 * the state values at the element vertices.
 *
 *  Function Description: Given two 2D-Views, compute the dot product of two complex
 *  quantities. Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComplexInertialEnergy
{
public:
    ComplexInertialEnergy(const Plato::Scalar & aOmega, const Plato::Scalar & aDensity) :
        mOmegaTimesOmegaTimesDensity(1)
    {
        mOmegaTimesOmegaTimesDensity = aOmega * aOmega * aDensity;
    }
    ~ComplexInertialEnergy()
    {
    }

    template<typename OutputScalarType, typename StateScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarMultiVectorT<StateScalarType> & aStateValues,
                                       const Plato::ScalarVectorT<OutputScalarType> & aInertialEnergy) const
    {
        aInertialEnergy(aCellOrdinal) = 0.0;
        for(Plato::OrdinalType tIndex = 0; tIndex < SpaceDim; tIndex++)
        {
            aInertialEnergy(aCellOrdinal) -= mOmegaTimesOmegaTimesDensity * aCellVolume(aCellOrdinal) 
                    * ( (aStateValues(aCellOrdinal, tIndex) * aStateValues(aCellOrdinal, tIndex))
                    + (aStateValues(aCellOrdinal, SpaceDim + tIndex) * aStateValues(aCellOrdinal, SpaceDim + tIndex)) );
        }
    }

private:
    /* Omega = Angular Frequency,  Density = Material Density*/
    Plato::Scalar mOmegaTimesOmegaTimesDensity;
};
// class ComplexInertialEnergy

} // namespace Plato

#endif /* COMPLEXINERTIALENERGY_HPP_ */
