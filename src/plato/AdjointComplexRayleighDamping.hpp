/*
 * AdjointComplexRayleighDamping.hpp
 *
 *  Created on: Apr 29, 2018
 */

#ifndef ADJOINTCOMPLEXRAYLEIGHDAMPING_HPP_
#define ADJOINTCOMPLEXRAYLEIGHDAMPING_HPP_

#include <cassert>

#include <Teuchos_ParameterList.hpp>

#include "plato/SimplexFadTypes.hpp"

namespace Plato
{

/****************************************************************************//**/
/*! AdjointComplexRayleighDamping Functor.
*
*   Evaluates the adjoint version of the classical Rayleigh damping model
*   at a given cell for structural dynamics applications. This model is used
*   during the solution of the adjoint problem. Rayleigh damping is viscous
*   damping, which is proportional to a linear combination of mass and stiffness.
*   The damping matrix \f$\mathbf{K}\f$ is given by \f$\mathbf{C}=\mu\mathbf{M}
*   + \lambda\mathbf{K}\f$, where \mathbf{M} and \mathbf{K} are the mass and
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
*   Finally, \f$\alpha\mathbf{M}\f$ lower modes most heavily and \f$\alpha
*   \mathbf{K}\f$ damps higher modes most heavily.
*
*********************************************************************************/
template<const Plato::OrdinalType SpaceDim>
class AdjointComplexRayleighDamping
{
private:
    Plato::Scalar mMassConstant;
    Plato::Scalar mStiffConstant;

public:
    /******************************************************************************/
    explicit AdjointComplexRayleighDamping(Teuchos::ParameterList & aProblemParams) :
            mMassConstant(aProblemParams.get<double>("Rayleigh Mass Damping", 0.0)),
            mStiffConstant(aProblemParams.get<double>("Rayleigh Stiffness Damping", 0.0))
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    AdjointComplexRayleighDamping(Plato::Scalar aMassConstant = 0.025, Plato::Scalar aStiffConstant = 0.023) :
            mMassConstant(aMassConstant),
            mStiffConstant(aStiffConstant)
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    ~AdjointComplexRayleighDamping(){}
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
            aDampingForces(aCellOrdinal, tDimIndex) = static_cast<Plato::Scalar>(-1) *
                    ( (mMassConstant * aInertialForces(aCellOrdinal, SpaceDim + tDimIndex)) +
                            (mStiffConstant * aElasticForces(aCellOrdinal, SpaceDim + tDimIndex)) );

            aDampingForces(aCellOrdinal, SpaceDim + tDimIndex) =
                    ( (mMassConstant * aInertialForces(aCellOrdinal, tDimIndex)) +
                            (mStiffConstant * aElasticForces(aCellOrdinal, tDimIndex)) );
        }
    }
};
// class AdjointComplexRayleighDamping

} // namespace Plato

#endif /* ADJOINTCOMPLEXRAYLEIGHDAMPING_HPP_ */
