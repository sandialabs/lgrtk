/*
 * RayleighDamping.hpp
 *
 *  Created on: Jun 26, 2018
 */

#ifndef SRC_PLATO_RAYLEIGHDAMPING_HPP_
#define SRC_PLATO_RAYLEIGHDAMPING_HPP_

#include <cassert>

#include <Teuchos_ParameterList.hpp>

#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/****************************************************************************//**
 * \brief Rayleigh damping functor for elastodynamics applications.
 *
 * Evaluates classical Rayleigh damping at a given cell for elastodynamics
 * applications. Rayleigh damping is viscous damping which is proportional
 * to a linear combination of mass and stiffness. The damping matrix \f$C\f$
 * is given by \f$\mathbf{C}=\mu\mathbf{M}+\lambda\mathbf{K}\f$, where
 * \f$\mathbf{M}\f$ and \f$\mathbf{K}\f$ are the mass and stiffness matrices
 * respectively and \f$\mu\f$ and \f$\lambda\f$ are constants of proportionality.
 *
 * Rayleigh damping does afford certain mathematical conveniences and is widely
 * used to model internal structural damping. One of the less attractive features
 * of Rayleigh damping however is that the achieved damping ratio \f$\xi\f$ varies
 * with response frequency. The stiffness proportional term contributes damping
 * that is linearly proportional to response frequency and the mass proportional
 * term contributes damping that is inversely proportional to response frequency.
 *
 * Finally, \f$\alpha\mathbf{M}\f$ lower modes most heavily and \f$\alpha
 * \mathbf{K}\f$ damps higher modes most heavily.
 *
 *******************************************************************************/
template<const Plato::OrdinalType NumDofsPerCell>
class RayleighDamping
{
private:
    Plato::Scalar mMassConstant; /*!< mass proportional damping coefficients */
    Plato::Scalar mStiffConstant; /*!< stiffness proportional damping coefficients */

public:
    /************************************************************************//**
     *
     * \brief Construct an instance of class Plato::RayleighDamping.
     *        \param aProblemParams Teuchos parameter list used to access the mass
     *                 and stiffness proportional damping coefficients
     *
     ***************************************************************************/
    explicit RayleighDamping(Teuchos::ParameterList & aProblemParams) :
            mMassConstant(aProblemParams.get<Plato::Scalar>("Rayleigh Mass Damping", 0.0)),
            mStiffConstant(aProblemParams.get<Plato::Scalar>("Rayleigh Stiffness Damping", 0.0))
    {
    }

    /************************************************************************//**
     *
     * \brief Construct an instance of class Plato::RayleighDamping.
     *        \param aMassConstant mass proportional damping coefficient,
     *          which is set to 0.025 by default
     *        \param aStiffConstant stiffness proportional damping coefficient,
     *          which is set to 0.023 by default
     *
     ***************************************************************************/
    RayleighDamping(Plato::Scalar aMassConstant = 0.025, Plato::Scalar aStiffConstant = 0.023) :
            mMassConstant(aMassConstant),
            mStiffConstant(aStiffConstant)
    {
    }

    /**************************************************************************//**
     *
     * \brief Destructor
     *        Destructor for class Plato::RayleighDamping.
     *
     *****************************************************************************/
    ~RayleighDamping(){}

    /**************************************************************************//**
     *
     * \brief Compute element (i.e. cell) damping forces.
     *        \param aCellOrdinal      element index
     *        \param aStiffPropDamping 2D stiffness proportional damping forces array
     *        \param aMassPropDamping  2D stiffness proportional damping forces array
     *        \param aViscousForces    2D viscous force array
     *
     *****************************************************************************/
    template<typename ElasticForceScalarType, typename InertialForceScalarType, typename DampingForceScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ElasticForceScalarType> & aStiffPropDamping,
                                       const Plato::ScalarMultiVectorT<InertialForceScalarType> & aMassPropDamping,
                                       const Plato::ScalarMultiVectorT<DampingForceScalarType> & aViscousForces) const
    {
        assert(aStiffPropDamping.extent(0) == aMassPropDamping.extent(0));
        assert(aStiffPropDamping.extent(0) == aMassPropDamping.extent(0));
        assert(aStiffPropDamping.extent(1) == aMassPropDamping.extent(1));
        assert(aViscousForces.extent(1) == aMassPropDamping.extent(1));

        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerCell; tDofIndex++)
        {
            aViscousForces(aCellOrdinal, tDofIndex) =
                    ( (mMassConstant * aMassPropDamping(aCellOrdinal, tDofIndex)) +
                            (mStiffConstant * aStiffPropDamping(aCellOrdinal, tDofIndex)) );
        }
    }
};
// class RayleighDamping

} // namespace Plato

#endif /* SRC_PLATO_RAYLEIGHDAMPING_HPP_ */
