/*
 * ComplexRayleighDamping.hpp
 *
 *  Created on: Apr 29, 2018
 */

#ifndef COMPLEXRAYLEIGHDAMPING_HPP_
#define COMPLEXRAYLEIGHDAMPING_HPP_

#include <cassert>

#include <Teuchos_ParameterList.hpp>

#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/****************************************************************************//**/
//! ComplexRayleighDamping Functor.
/*!
*   Evaluates classical Rayleigh damping at a given cell for structural dynamics
*   applications. Rayleigh damping is viscous damping which is proportional to
*   a linear combination of mass and stiffness. The damping matrix \f$C\f$ is
*   given by \f$C=\mu M+\lambda K\f$, where \f$M\f$ and \f$K\f$ are the mass and
*   stiffness matrices respectively and \f$\mu\f$ and \f$\lambda\f$ are constants
*   of proportionality.
*
*   Rayleigh damping does afford certain mathematical conveniences and is widely
*   used to model internal structural damping. One of the less attractive features
*   of Rayleigh damping however is that the achieved damping ratio \f$\xi\f$ varies
*   with response frequency. The stiffness proportional term contributes damping
*   that is linearly proportional to response frequency and the mass proportional
*   term contributes damping that is inversely proportional to response frequency.
*
*********************************************************************************/
template<const Plato::OrdinalType SpaceDim>
class ComplexRayleighDamping
{
private:
    Plato::Scalar mMassConstant;
    Plato::Scalar mStiffConstant;

public:
    /******************************************************************************/
    explicit ComplexRayleighDamping(Teuchos::ParameterList & aProblemParams) :
            mMassConstant(aProblemParams.get<double>("Rayleigh Mass Damping", 0.0)),
            mStiffConstant(aProblemParams.get<double>("Rayleigh Stiffness Damping", 0.0))
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    ComplexRayleighDamping(Plato::Scalar aMassConstant = 0.025, Plato::Scalar aStiffConstant = 0.023) :
            mMassConstant(aMassConstant),
            mStiffConstant(aStiffConstant)
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    ~ComplexRayleighDamping(){}
    /******************************************************************************/

    /******************************************************************************/
    template<typename ElasticForceScalarType, typename InertialForceScalarType, typename DampingForceScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ElasticForceScalarType> & aElasticForces,
                                       const Plato::ScalarMultiVectorT<InertialForceScalarType> & aInertialForces,
                                       const Plato::ScalarMultiVectorT<DampingForceScalarType> & aDampingForces) const
    /******************************************************************************/
    {
        assert(aElasticForces.extent(0) == aInertialForces.extent(0));
        assert(aElasticForces.extent(0) == aInertialForces.extent(0));
        assert(aElasticForces.extent(1) == aInertialForces.extent(1));
        assert(aDampingForces.extent(1) == aInertialForces.extent(1));

        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aDampingForces(aCellOrdinal, tDimIndex) = 
                    ( (mMassConstant * aInertialForces(aCellOrdinal, SpaceDim + tDimIndex)) +
                            (mStiffConstant * aElasticForces(aCellOrdinal, SpaceDim + tDimIndex)) );

            aDampingForces(aCellOrdinal, SpaceDim + tDimIndex) = static_cast<Plato::Scalar>(-1) *
                    ( (mMassConstant * aInertialForces(aCellOrdinal, tDimIndex)) +
                            (mStiffConstant * aElasticForces(aCellOrdinal, tDimIndex)) );
        }
    }
};
// class ComplexRayleighDamping

} // namespace Plato

#endif /* COMPLEXRAYLEIGHDAMPING_HPP_ */
