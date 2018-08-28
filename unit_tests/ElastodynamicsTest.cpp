/*
 * ElastodynamicsTest.cpp
 *
 *  Created on: Jun 24, 2018
 */

#include <vector>
#include <cstdint>

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/RayleighDamping.hpp"

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "ApplyConstraints.hpp"
#include "LinearElasticMaterial.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Strain.hpp"
#include "plato/ExpVolume.hpp"
#include "plato/StateValues.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/LinearStress.hpp"
#include "plato/ApplyPenalty.hpp"
#include "plato/PlatoUtilities.hpp"
#include "plato/ScalarFunction.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/InertialForces.hpp"
#include "plato/ApplyProjection.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoAbstractProblem.hpp"
#include "plato/HeavisideProjection.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/HyperbolicTangentProjection.hpp"

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

namespace Plato
{

namespace Experimental
{

template <typename SimplexPhysicsT>
struct DynamicResidualTypes : Plato::EvaluationTypes<SimplexPhysicsT>
{
    using StateUScalarType  = Plato::Scalar;
    using StateVScalarType  = Plato::Scalar;
    using StateAScalarType  = Plato::Scalar;
    using ControlScalarType = Plato::Scalar;
    using ConfigScalarType  = Plato::Scalar;
    using ResultScalarType  = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientStateUTypes : Plato::EvaluationTypes<SimplexPhysicsT>
{
    using SFadType = typename Plato::SimplexFadTypes<SimplexPhysicsT>::StateFad;

    using StateUScalarType  = SFadType;
    using StateVScalarType  = Plato::Scalar;
    using StateAScalarType  = Plato::Scalar;
    using ControlScalarType = Plato::Scalar;
    using ConfigScalarType  = Plato::Scalar;
    using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientStateVTypes : Plato::EvaluationTypes<SimplexPhysicsT>
{
    using SFadType = typename Plato::SimplexFadTypes<SimplexPhysicsT>::StateFad;

    using StateUScalarType  = Plato::Scalar;
    using StateVScalarType  = SFadType;
    using StateAScalarType  = Plato::Scalar;
    using ControlScalarType = Plato::Scalar;
    using ConfigScalarType  = Plato::Scalar;
    using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientStateATypes : Plato::EvaluationTypes<SimplexPhysicsT>
{
    using SFadType = typename Plato::SimplexFadTypes<SimplexPhysicsT>::StateFad;

    using StateUScalarType  = Plato::Scalar;
    using StateVScalarType  = Plato::Scalar;
    using StateAScalarType  = SFadType;
    using ControlScalarType = Plato::Scalar;
    using ConfigScalarType  = Plato::Scalar;
    using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : Plato::EvaluationTypes<SimplexPhysicsT>
{
    using SFadType = typename Plato::SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

    using StateUScalarType  = Plato::Scalar;
    using StateVScalarType  = Plato::Scalar;
    using StateAScalarType  = Plato::Scalar;
    using ControlScalarType = Plato::Scalar;
    using ConfigScalarType  = SFadType;
    using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : Plato::EvaluationTypes<SimplexPhysicsT>
{
    using SFadType = typename Plato::SimplexFadTypes<SimplexPhysicsT>::ControlFad;

    using StateUScalarType  = Plato::Scalar;
    using StateVScalarType  = Plato::Scalar;
    using StateAScalarType  = Plato::Scalar;
    using ControlScalarType = SFadType;
    using ConfigScalarType  = Plato::Scalar;
    using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct DynamicsEvaluation
{
    using Residual  = DynamicResidualTypes<SimplexPhysicsT>;
    using GradientU = GradientStateUTypes<SimplexPhysicsT>;
    using GradientV = GradientStateVTypes<SimplexPhysicsT>;
    using GradientA = GradientStateATypes<SimplexPhysicsT>;
    using GradientZ = GradientZTypes<SimplexPhysicsT>;
    using GradientX = GradientXTypes<SimplexPhysicsT>;
};


/******************************************************************************/
inline void copy(const std::vector<Plato::Scalar> & aData, Plato::ScalarVector & aDeviceVector)
/******************************************************************************/
{
    const Plato::OrdinalType tNumDofs = aDeviceVector.extent(0);
    auto tHostVector = Kokkos::create_mirror(aDeviceVector);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    {
        tHostVector(tIndex) = aData[tIndex];
    }
    Kokkos::deep_copy(aDeviceVector, tHostVector);
}

/****************************************************************************//**
 *
 * \brief Update multi-vector values.
 *
 * Update multi-vector values, VecOut = ScalarOut*VecOut + ScalarIn*VecIn.
 *
 * Input and Output arguments:
 *
 * @param [in] aCellOrdinal cell (i.e. element) ordinal
 * @param [in] aScalarIn scaling for input multi-vector
 * @param [in] aInput input multi-vector
 * @param [in,out] aOutput output multi-vector
 *
**********************************************************************************/
template<class InType, class OutType>
KOKKOS_INLINE_FUNCTION void
axpy(const Plato::OrdinalType & aCellOrdinal,
     const Plato::Scalar & aScalarIn,
     const InType & aInput,
     OutType & aOutput)
{
    assert(aInput.size() == aOutput.size());
    assert(aInput.extent(1) == aOutput.extent(1));

    const Plato::OrdinalType tLength = aOutput.extent(1);
    for(Plato::OrdinalType tIndex = 0; tIndex < tLength; tIndex++)
    {
        aOutput(aCellOrdinal, tIndex) += aScalarIn * aInput(aCellOrdinal, tIndex);
    }
}

/******************************************************************************//**
 *
 * \brief Check if generalized Newmark \f$\alpha\f$ parameter is within the range.
 *
 * Check if generalized Newmark \f$\alpha\f$ parameter is within the following
 * range \f$\frac{-1}{3}\leq\alpha\leq{0}\f$ to maintain unconditional stability.
 *
 * Input and output parameters
 *
 * @param [in,out] aAlpha algorithmic damping parameter
 *
 *********************************************************************************/
inline void check_newmark_alpha_param(Plato::Scalar & aAlpha)
{
    aAlpha = aAlpha < static_cast<Plato::Scalar>(0) ? aAlpha : static_cast<Plato::Scalar>(-0.1);
    aAlpha = aAlpha < static_cast<Plato::Scalar>(-1/3) ? static_cast<Plato::Scalar>(-0.1) : aAlpha;
}

/******************************************************************************//**
 *
 * \brief Computes the optimal Newmark damping parameters, i.e. \f$\beta\f$ and \f$\gamma\f$.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, the \f$\beta\f$ and \f$\gamma\f$ equations shown in Table 11.13-1,
 * page 420.
 *
 * The following equations for \f$\beta\f$ and \f$\gamma\f$ parameters introduce
 * algorithmic damping and retain unconditional stability,
 *
 * \f$\beta = \frac{1}{4}\left(1 - \alpha\right)^{2}\f$
 * \f$\gamma = \frac{1}{2}\left(1 - 2\alpha\right)\f$,
 *
 * where \f$\alpha\f$ introduces algorithmic damping to damp lower modes and
 * \f$\gamma\f$ and \f$\beta\f$ introduces algorithmic damping to damp higher
 * modes.
 *
 * Input and output parameters
 *
 * @param [in,out] aAlpha algorithmic damping parameter
 * @param [in,out] aBeta algorithmic damping parameter
 * @param [in,out] aGamma algorithmic damping parameter
 *
 *********************************************************************************/
inline void compute_newmark_damping_coeff(Plato::Scalar & aAlpha, Plato::Scalar & aBeta, Plato::Scalar & aGamma)
{
    Plato::Experimental::check_newmark_alpha_param(aAlpha);
    aBeta = static_cast<Plato::Scalar>(0.25)
            * ( (static_cast<Plato::Scalar>(1.0) - aAlpha) * (static_cast<Plato::Scalar>(1.0) - aAlpha) );
    aGamma = static_cast<Plato::Scalar>(0.5)
            * ( static_cast<Plato::Scalar>(1.0) - (static_cast<Plato::Scalar>(2.0) * aAlpha) );
}

/*************************************************************************//**
 *
 * \brief Checks if the objective function is allocated.
 *
 * Checks if the objective function is allocated. A runtime error is thrown
 * if the objective function is not allocated.
 *
 * Input arguments
 *
 * @param [in] aInput objective function shared pointer
 *
 ******************************************************************************/
template<typename ScalarFunctionType>
inline void is_objective_func_defined(const std::shared_ptr<ScalarFunctionType> & aInput)
{
    try
    {
        if(aInput == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::invalid_argument(tErrorMessage.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::runtime_error(tErrorMsg.what());
    }
}

/*************************************************************************//**
 *
 * \brief Checks if the equality constraint is allocated.
 *
 * Checks if the equality constraint is allocated. A runtime error is thrown
 * if the equality constraint is not allocated.
 *
 * Input arguments
 *
 * @param [in] aInput equality constraint shared pointer
 *
 ******************************************************************************/
template<typename VectorFunctionType>
inline void is_equality_constraint_defined(const std::shared_ptr<VectorFunctionType> & aInput)
{
    try
    {
        if(aInput == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << "\nMESSAGE: EQUALITY (I.E. PDE) CONSTRAINT PTR IS NOT DEFINED."
                    << " USER SHOULD MAKE SURE THAT THE EQUALITY CONSTRAINT WAS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::invalid_argument(tErrorMessage.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::runtime_error(tErrorMsg.what());
    }
}

/*************************************************************************//**
 *
 * \brief Checks if the inequality constraint is allocated.
 *
 * Checks if the inequality constraint is allocated. A runtime error is thrown
 * if the inequality constraint is not allocated.
 *
 * Input arguments
 *
 * @param [in] aInput inequality constraint shared pointer
 *
******************************************************************************/
template<typename ScalarFunctionType>
inline void is_inequality_constraint_defined(const std::shared_ptr<ScalarFunctionType> & aInput)
{
    try
    {
        if(aInput == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << "\nMESSAGE: INEQUALITY CONSTRAINT PTR IS NOT DEFINED."
                    << " USER SHOULD MAKE SURE THAT THE INEQUALITY CONSTRAINT WAS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::invalid_argument(tErrorMessage.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::runtime_error(tErrorMsg.what());
    }
}

/*************************************************************************//**
 *
 * \brief Checks if the adjoint vector function is allocated.
 *
 * Checks if the adjoint vector function is allocated. A runtime error is thrown
 * if the adjoint vector function is not allocated. The adjoint vector function
 * is used to compute the Lagrange multipliers.
 *
 * Input arguments
 *
 * @param [in] aInput adjoint residual shared pointer
 *
******************************************************************************/
template<typename VectorFunctionType>
inline void is_adjoint_residual_defined(const std::shared_ptr<VectorFunctionType> & aInput)
{
    try
    {
        if(aInput == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << "\nMESSAGE: ADJOINT RESIDUAL PTR IS NOT DEFINED. MAJOR ERROR!!!"
                    << " THE PTR ASSOCIATED WITH THE ADJOINT RESIDUAL WAS UNEXPECTEDLY DELETED OR WAS NOT ALLOCATED. **************\n\n";
            throw std::invalid_argument(tErrorMessage.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::runtime_error(tErrorMsg.what());
    }
}

/*************************************************************************//**
 *
 * \brief Checks if sublist is defined.
 *
 * Checks if input sublist is defined in the input file. A runtime error is
 * thrown if the sublist is not defined.
 *
 * Input arguments
 *
 * @param [in] aName sublist name
 * @param [in] aParamList Teuchos parameter list
 *
******************************************************************************/
inline void is_sublist_defined(const std::string aName, Teuchos::ParameterList & aParamList)
{
    try
    {
        if(aParamList.isSublist(aName) == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << "\nMESSAGE: REQUIRED SUBLIST IS NOT DEFINED."
                    << " USER SHOULD DEFINE SUBLIST = '" << aName.c_str() << "' IN THE INPUT FILE. **************\n\n";
            throw std::invalid_argument(tErrorMessage.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::runtime_error(tErrorMsg.what());
    }
}

/*************************************************************************//**
 *
 * \brief Checks if sublist is defined.
 *
 * Checks if input parameter is defined in the input file. A runtime error is
 * thrown if the parameter is not defined.
 *
 * Input arguments
 *
 * @param [in] aName parameter name
 * @param [in] aParamList Teuchos parameter list
 *
******************************************************************************/
inline void is_parameter_defined(const std::string aName, Teuchos::ParameterList & aParamList)
{
    try
    {
        if(aParamList.isParameter(aName) == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << "\nMESSAGE: PARAMETER IS NOT DEFINED."
                    << " USER SHOULD DEFINE PARAMETER = '" << aName.c_str() << "' IN THE INPUT FILE. **************\n\n";
            throw std::invalid_argument(tErrorMessage.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::runtime_error(tErrorMsg.what());
    }
}

/******************************************************************************//**
 *
 * \brief Compute interpolated displacement vector.
 *
 * Computes the interpolated cell displacements vector needed to compute the elastodynamics
 * residual when the generalized Newmark \f$\alpha\f$-method is used. The interpolated
 * displacements are given by
 *
 * \f$\hat{\mathbf{u}}_{n} = \mathbf{u}_{n-1} + \Delta{t}_n\dot{\mathbf{u}}_{n-1}
 * + \Delta{t}_n^2\left(\frac{1}{2} - \beta\right)\ddot{\mathbf{u}}_{n-1}\f$,
 *
 * where \f$\Delta{t}_n\f$ is the current time step, \f$\beta\f$ is an algorithmic
 * damping parameter. Finally, \f$\ddot{\mathbf{u}}_{n-1}\f$, \f$\dot{\mathbf{u}}_{n-1}\f$
 * and \f$\mathbf{u}_{n-1}\f$ are the current acceleration, velocity and displacement
 * vectors,
 *
 * Input and output arguments:
 *
 * @param [in] aCellOrdinal cell ordinal
 * @param [in] aTimeStep current time step
 * @param [in] aBeta algorithmic damping parameter
 * @param [in] aDisp displacement vector
 * @param [in] aVel velocity vector
 * @param [in] aAcc acceleration vector
 * @param [in,out] aOutput interpolated displacements
 *
*******************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, class DispType, class VelType, class AccType, class OutType>
KOKKOS_INLINE_FUNCTION void
compute_interpolated_disp(const Plato::OrdinalType & aCellOrdinal,
                          const Plato::Scalar & aTimeStep,
                          const Plato::Scalar & aBeta,
                          const DispType & aDisp,
                          const VelType & aVel,
                          const AccType & aAcc,
                          OutType & aOutput)
{
    assert(aVel.extent(1) == NumDofsPerCell);
    assert(aAcc.extent(1) == NumDofsPerCell);
    assert(aDisp.extent(1) == NumDofsPerCell);
    assert(aOutput.extent(1) == NumDofsPerCell);
    auto tConstant = aTimeStep * aTimeStep * (static_cast<Plato::Scalar>(0.5) - aBeta);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerCell; tDofIndex++)
    {
        aOutput(aCellOrdinal, tDofIndex) = aDisp(aCellOrdinal, tDofIndex) + aTimeStep * aVel(aCellOrdinal, tDofIndex)
            + tConstant * aAcc(aCellOrdinal, tDofIndex);
    }
}

/******************************************************************************//**
 *
 * \brief Compute interpolated velocity vector.
 *
 * Computes the interpolated cell velocity vector needed to compute the elastodynamics
 * residual when the generalized Newmark \f$\alpha\f$-method is used. The interpolated
 * velocity vector is given by
 *
 * \f$\hat{\dot{\mathbf{u}}}_{n} =
 *   \dot{\mathbf{u}}_{n-1} + \Delta{t}_n\left(1-\gamma\right)\ddot{\mathbf{u}}_{n-1}\f$,
 *
 * where \f$\Delta{t}_n\f$ is the current time step, \f$\gamma\f$ is an algorithmic
 * damping parameter. Finally, \f$\ddot{\mathbf{u}}_{n-1}\f$ and \f$\dot{\mathbf{u}}_{n-1}\f$
 * are the current acceleration and velocity vectors,
 *
 * Input and output arguments:
 *
 * @param [in] aCellOrdinal cell ordinal
 * @param [in] aTimeStep current time step
 * @param [in] aGamma algorithmic damping parameter
 * @param [in] aVel velocity vector
 * @param [in] aAcc acceleration vector
 * @param [in,out] aOutput interpolated displacements
 *
*******************************************************************************/
template<Plato::OrdinalType NumDofsPerCell, class VelType, class AccType, class OutType>
KOKKOS_INLINE_FUNCTION void
compute_interpolated_vel(const Plato::OrdinalType & aCellOrdinal,
                         const Plato::Scalar & aTimeStep,
                         const Plato::Scalar & aGamma,
                         const VelType & aVel,
                         const AccType & aAcc,
                         OutType & aOutput)
{
    assert(aVel.extent(1) == NumDofsPerCell);
    assert(aAcc.extent(1) == NumDofsPerCell);
    assert(aOutput.extent(1) == NumDofsPerCell);
    auto tConstant = aTimeStep * (static_cast<Plato::Scalar>(1.0) - aGamma);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerCell; tDofIndex++)
    {
        aOutput(aCellOrdinal, tDofIndex) = aVel(aCellOrdinal, tDofIndex) + tConstant * aAcc(aCellOrdinal, tDofIndex);
    }
}

/******************************************************************************//**
 *
 * \brief Generalized Newmark-\f$\alpha\f$ time integration method.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.13-4a) and (11.13-4b) on page 418.
 *
 * \f$\dot{\mathbf{u}}_{n} = \dot{\mathbf{u}}_{n-1} + \Delta{t}_n\left( \gamma
 * \ddot{\mathbf{u}}_{n} + \left(1-\gamma\right)\ddot{\mathbf{u}}_{n-1} \right)\f$
 *
 * and
 *
 * \f$\mathbf{u}_{n} = \mathbf{u}_{n-1} + \Delta{t}_n\dot{\mathbf{u}}_{n-1} +
 * \Delta{t}_n^2\left[ \beta\ddot{\mathbf{u}}_{n} + \left(\frac{1}{2}-\beta\right)
 * \ddot{\mathbf{u}}_{n-1} \right]\f$,
 *
 * where \f$\gamma\f$ and \f$\beta\f$ are numerical factors used in the Newmark
 * method, \f$\Delta{t}_n$\f is the current time step, \f$\mathbf{u}\f$,
 * \f$\dot{\mathbf{u}}\f$ and \f$\ddot{\mathbf{u}}\f$ are the displacement,
 * velocity and acceleration vectors.
 *
 * Input and Output arguments:
 *
 * @param [in] aTimeStep current time step
 * @param [in] aBeta algorithmic damping parameter
 * @param [in] aGamma algorithmic damping parameter
 * @param [in] aOldAcc accelerations at the previous time step
 * @param [in] aOldVel velocities at the previous time step
 * @param [in] aOldDisp displacements at the previous time step
 * @param [in] aNewAcc accelerations at the current time step
 * @param [in,out] aNewVel velocities at the current time step
 * @param [in,out] aNewDisp displacements at the current time step
 *
 *********************************************************************************/
inline void newmark_update(Plato::Scalar aTimeStep,
                           Plato::Scalar aBeta,
                           Plato::Scalar aGamma,
                           Plato::ScalarVector aOldAcc,
                           Plato::ScalarVector aOldVel,
                           Plato::ScalarVector aOldDisp,
                           Plato::ScalarVector aNewAcc,
                           Plato::ScalarVector aNewVel,
                           Plato::ScalarVector aNewDisp)
{
    assert(aOldDisp.size() == aOldAcc.size());
    assert(aOldDisp.size() == aOldVel.size());
    assert(aOldDisp.size() == aNewAcc.size());
    assert(aOldDisp.size() == aNewVel.size());
    assert(aOldDisp.size() == aNewDisp.size());

    auto tNumDofs = aNewDisp.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        // Update velocities
        aNewVel(aIndex) = aOldVel(aIndex) + ( aTimeStep * (aGamma * aNewAcc(aIndex) + (static_cast<Plato::Scalar>(1) - aGamma) * aOldAcc(aIndex) ) );
        // Update displacements
        aNewDisp(aIndex) = aOldDisp(aIndex) + ( aTimeStep * aOldVel(aIndex) ) + ( aTimeStep * aTimeStep * ( (aBeta * aNewAcc(aIndex) )
                + ( (static_cast<Plato::Scalar>(0.5) - aBeta) * aOldAcc(aIndex) ) ) );
    }, "NewmarkUpdate");
}

/******************************************************************************//**
 *
 * \brief Adjoint displacement transition equation
 *
 * The new adjoint displacement vector \f$\Phi_n\f$ for \f$n = 0,\dots,N-1\f$, where 4
 * \f$N\f$ is the total number of time steps, is given by
 *
 * \f$\Phi_n = \left(\frac{\partial{f}_{N-n}}{\partial\mathbf{u}_{N-n}}\right)^{T}
 * + H(n)\bigg[ \mathbf{K}\Lambda_{n-1} + \Phi_{n-1} \bigg]\f$.
 *
 * \f$f_{N-n}\equiv{f}(\mathbf{u}_{N-n},\mathbf{v}_{N-n},\mathbf{a}_{N-n},\mathbf{z})\f$
 * is a criterion of interest, \f$H(n)\f$ is the Heaviside step function, \f$\mathbf{K}\f$
 * is the stiffness matrix, \f$\mathbf{u}\f$ is the displacement vector. Finally, \f$Phi\f$
 * and \f$\Lambda\f$ are the adjoint displacement and acceleration vectors.
 *
 * Input and Output arguments:
 *
 * @param [in] aOldDisp old adjoint displacement vector
 * @param [in] aOldForce old elastic force vector
 * @param [in] aDfDu partial derivative of a criterion with respect to the displacement vector
 * @param [in,out] aNewDisp new adjoint displacement vector
 *
 *********************************************************************************/
inline void adjoint_displacement_update(const Plato::ScalarVector& aOldDisp,
                                        const Plato::ScalarVector& aOldForce,
                                        const Plato::ScalarVector& aDfDu,
                                        Plato::ScalarVector& aNewDisp)
{
    assert(aOldDisp.size() == aDfDu.size());
    assert(aOldDisp.size() == aNewDisp.size());
    assert(aOldDisp.size() == aOldForce.size());

    auto tNumDofs = aNewDisp.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        aNewDisp(aIndex) = aDfDu(aIndex) + aOldForce(aIndex) + aOldDisp(aIndex);
    }, "NewmarkUpdate - Adjoint Displacements");
}

/******************************************************************************//**
 *
 * \brief Adjoint velocity transition equation
 *
 * The new adjoint velocity vector \f$\Theta_n\f$ for \f$n = 0,\dots,N-1\f$, where
 * \f$N\f$ is the total number of time steps, is given by
 *
 * \f$ \Theta_n = \left(\frac{\partial{f}_{N-n}}{\partial\mathbf{v}_{N-n}}\right)^{T}
 * + H(n)\bigg[ \Delta{t}_{N-n+1}\left(\bar{\alpha}\mathbf{K}\Lambda_{n-1}+\Phi_{n-1}
 * \right) \bigg] + \mathbf{C}\Lambda_{n-1}+\Theta_{n-1} \f$.
 *
 * \f$f_{N-n}\equiv{f}(\mathbf{u}_{N-n},\mathbf{v}_{N-n},\mathbf{a}_{N-n},\mathbf{z})\f$
 * is a criterion of interest, \f$\Delta{t}\f$ is the time step, \f$\bar{\alpha}=\left(
 * 1 + \alpha\right)\f$ is an algorithmic damping parameter, \f$H(n)\f$ is the Heaviside
 * step function, \f$\mathbf{K}\f$ and \f$\mathbf{C}\f$ are the stiffness and damping
 * matrices and \f$\mathbf{v}\f$ is the velocity vector. Finally, \f$\Phi\f$ and
 * \f$\Lambda\f$ are the adjoint displacement and acceleration vectors.
 *
 * Input and Output arguments:
 *
 * @param [in] aTimeStep time step
 * @param [in] aAlpha algorithmic damping parameter
 * @param [in] aOldDisp old adjoint displacement vector
 * @param [in] aOldVel old adjoint velocity vector
 * @param [in] aOldElasticForce old elastic force vector
 * @param [in] aOldViscousForce old viscous force vector
 * @param [in] aDfDv partial derivative of a criterion with respect to the velocity vector
 * @param [in,out] aNewVel new adjoint velocity vector
 *
 *********************************************************************************/
inline void adjoint_velocity_update(const Plato::Scalar& aTimeStep,
                                    const Plato::Scalar& aAlpha,
                                    const Plato::ScalarVector& aOldDisp,
                                    const Plato::ScalarVector& aOldVel,
                                    const Plato::ScalarVector& aOldElasticForce,
                                    const Plato::ScalarVector& aOldViscousForce,
                                    const Plato::ScalarVector& aDfDv,
                                    Plato::ScalarVector& aNewVel)
{
    assert(aOldDisp.size() == aDfDv.size());
    assert(aOldDisp.size() == aOldVel.size());
    assert(aOldDisp.size() == aNewVel.size());
    assert(aOldDisp.size() == aOldElasticForce.size());
    assert(aOldDisp.size() == aOldViscousForce.size());

    auto tNumDofs = aNewVel.size();
    auto tOnePlusAlpha = static_cast<Plato::Scalar>(1) + aAlpha;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        aNewVel(aIndex) = aDfDv(aIndex) + aTimeStep * ( tOnePlusAlpha * aOldElasticForce(aIndex) + aOldDisp(aIndex) )
                + aOldViscousForce(aIndex) + aOldVel(aIndex);
    }, "NewmarkUpdate - Adjoint Velocities");
}

/****************************************************************************//**
 *
 * \brief Abstract interface for a force function
 *
********************************************************************************/
template<typename EvaluationType>
class AbstractForceFunction
{
public:
    /****************************************************************************//**
     *
     * \brief Destructor
     *
    ********************************************************************************/
    virtual ~AbstractForceFunction(){}

    /****************************************************************************//**
     *
     * \brief Pure virtual force evaluation function
     *
    ********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateUScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> & aForce,
             Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class AbstractForceFunction

/****************************************************************************//**
 *
 * \brief Linear elastic force function
 *
 * A linear elastic force is given by \f$\mathbf{f}_e = \mathbf{K}\mathbf{u}\f$,
 * where \f$\mathbf{f}_e\f$ is the elastic force vector, \f$\mathbf{K}\f$ is the
 * stiffness matrix and \f$\mathbf{u}\f$ is the displacement vector.
 *
********************************************************************************/
template<typename EvaluationType, typename PenaltyType, typename ProjectionType>
class LinearElasticForce :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public AbstractForceFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpatialDim = EvaluationType::SpatialDim;

    using Simplex<mSpatialDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<mSpatialDim>::m_numVoigtTerms;

    using StateType = typename EvaluationType::StateUScalarType;
    using ConfigType = typename EvaluationType::ConfigScalarType;
    using ResultType = typename EvaluationType::ResultScalarType;
    using ControlType = typename EvaluationType::ControlScalarType;

    PenaltyType mPenaltyFunction;
    ProjectionType mProjectionFunction;
    Plato::ApplyPenalty<PenaltyType> mApplyPenalty;
    Plato::ApplyProjection<ProjectionType> mApplyProjection;
    Plato::LinearTetCubRuleDegreeOne<mSpatialDim> mCubatureRule;

    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> mCellStiffness;

private:
    /****************************************************************************//**
     *
     * \brief Initialize material model
     *
     * Input arguments
     *
     * @param [in] aParamList Teuchos parameter list with input data
     *
    ********************************************************************************/
    void initialize(Teuchos::ParameterList & aParamList)
    {
        // Create material model and get stiffness
        Plato::Experimental::is_sublist_defined("Material Model", aParamList);
        Plato::ElasticModelFactory<mSpatialDim> tElasticModelFactory(aParamList);
        auto tMaterialModel = tElasticModelFactory.create();
        mCellStiffness = tMaterialModel->getStiffnessMatrix();
    }

public:
    /****************************************************************************//**
     *
     * \brief Constructor
     *
     * Input arguments
     *
     * @param [in] aProblemParams Teuchos parameter list with physics' input data
     * @param [in] aPenaltyParams Teuchos parameter list with penalty model's input data
     *
    ********************************************************************************/
    LinearElasticForce(Teuchos::ParameterList& aProblemParams, Teuchos::ParameterList& aPenaltyParams) :
        mProjectionFunction(),
        mPenaltyFunction(aPenaltyParams),
        mApplyPenalty(mPenaltyFunction),
        mApplyProjection(mProjectionFunction),
        mCubatureRule(),
        mCellStiffness()
    {
        this->initialize(aProblemParams);
    }

    /****************************************************************************//**
     *
     * \brief Destructor
     *
    ********************************************************************************/
    ~LinearElasticForce()
    {
    }

    /****************************************************************************//**
     *
     * \brief Evaluate linear elastic forces
     *
     * A linear elastic force is given by \f$\mathbf{f}_e = \mathbf{K}\mathbf{u}\f$,
     * where \f$\mathbf{f}_e\f$ is the elastic force vector, \f$\mathbf{K}\f$ is the
     * stiffness matrix and \f$\mathbf{u}\f$ is the displacement vector.
     *
     * Input arguments
     *
     * @param [in] aState state vector (e.g. displacements)
     * @param [in] aControl design variable vectors
     * @param [in] aConfig configuration vector (i.e. coordinates)
     * @param [in,out] aForce force vector
     * @param [in] aTimeStep current time step
     *
    ********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateType> & aState,
                  const Plato::ScalarMultiVectorT<ControlType> & aControl,
                  const Plato::ScalarArray3DT<ConfigType> & aConfig,
                  Plato::ScalarMultiVectorT<ResultType> & aForce,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        // elastic forces functors
        Strain<mSpatialDim> tComputeStrains;
        LinearStress<mSpatialDim> tComputeStress(mCellStiffness);
        StressDivergence<mSpatialDim> tComputeStressDivergence;
        Plato::ComputeGradientWorkset<mSpatialDim> tComputeGradient;

        // allocate local containers
        auto tNumCells = aState.extent(0);
        Plato::ScalarVector tCellVolume("Cell Volume", tNumCells);
        Plato::ScalarMultiVector tCellStrain("Cell Strain", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVector tCellStress("Cell Stress", tNumCells, m_numVoigtTerms);
        Plato::ScalarArray3D tCellGradient("Cell Gradient", tNumCells, m_numNodesPerCell, mSpatialDim);

        // Copy data from host into device
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto tQuadratureWeight = mCubatureRule.getCubWeight();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tCellGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeStrains(aCellOrdinal, tCellStrain, aState, tCellGradient);
            tComputeStress(aCellOrdinal, tCellStress, tCellStrain);
            ControlType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            tApplyPenalty(aCellOrdinal, tCellDensity, tCellStress);
            tComputeStressDivergence(aCellOrdinal, aForce, tCellStress, tCellGradient, tCellVolume);
        }, "EvaluateElasticForces");
    }
};
// class LinearElasticForce

/****************************************************************************//**
 *
 * \brief Rayleigh viscous force function
 *
 * The Rayleigh damping force vecotr is given by \f$\mathbf{f}_d = \left(\alpha
 * \mathbf{M} + \beta\mathbf{K}\right)\dot{\mathbf{u}}\f$, where \f$\mathbf{f}_d\f$
 * is the damping force vector, \f$\alpha\f$ is the mass proportional damping
 * coefficient, \f$\beta\f$ is the stiffness proportional damping coefficient
 * \f$\mathbf{K}\f$ is the stiffness matrix, \f$\mathbf{M}\f$ is the mass matrix
 * and \f$\dot{\mathbf{u}}\f$ is the velocity vector.
 *
********************************************************************************/
template<typename EvaluationType, typename PenaltyType, typename ProjectionType>
class RayleighViscousForce :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public AbstractForceFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpatialDim = EvaluationType::SpatialDim;

    using Simplex<mSpatialDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<mSpatialDim>::m_numVoigtTerms;
    using Plato::SimplexMechanics<mSpatialDim>::m_numDofsPerNode;
    using Plato::SimplexMechanics<mSpatialDim>::m_numDofsPerCell;

    using StateType = typename EvaluationType::StateUScalarType;
    using ConfigType = typename EvaluationType::ConfigScalarType;
    using ResultType = typename EvaluationType::ResultScalarType;
    using ControlType = typename EvaluationType::ControlScalarType;

    Plato::Scalar mDensity;
    Plato::Scalar mMassPropDamp;
    Plato::Scalar mStiffPropDamp;

    PenaltyType mPenaltyFunction;
    ProjectionType mProjectionFunction;
    Plato::ApplyPenalty<PenaltyType> mApplyPenalty;
    Plato::ApplyProjection<ProjectionType> mApplyProjection;
    Plato::LinearTetCubRuleDegreeOne<mSpatialDim> mCubatureRule;

    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> mCellStiffness;

private:
    /****************************************************************************//**
     *
     * \brief Initialize material model
     *
     * Input arguments
     *
     * @param [in] aProblemParams Teuchos parameter list with input data
     *
    ********************************************************************************/
    void initialize(Teuchos::ParameterList & aParamList)
    {
        // Create material model and get stiffness
        Plato::Experimental::is_sublist_defined("Material Model", aParamList);
        Plato::ElasticModelFactory<mSpatialDim> tElasticModelFactory(aParamList);
        auto tMaterialModel = tElasticModelFactory.create();
        mCellStiffness = tMaterialModel->getStiffnessMatrix();

        // Parse material density
        auto tMaterialParamList = aParamList.sublist("Material Model");
        Plato::Experimental::is_parameter_defined("Density", tMaterialParamList);
        mDensity = tMaterialParamList.get<Plato::Scalar>("Density");

        // Parse Rayleigh damping coefficients
        mMassPropDamp = tMaterialParamList.get<Plato::Scalar>("Mass Proportional Damping", 0.0);
        mStiffPropDamp = tMaterialParamList.get<Plato::Scalar>("Stiffness Proportional Damping", 0.0);
    }

public:
    /****************************************************************************//**
     *
     * \brief Constructor
     *
     * Input arguments
     *
     * @param [in] aProblemParams Teuchos parameter list with physics' input data
     * @param [in] aPenaltyParams Teuchos parameter list with penalty model's input data
     *
    ********************************************************************************/
    RayleighViscousForce(Teuchos::ParameterList& aProblemParams, Teuchos::ParameterList& aPenaltyParams) :
        mDensity(1),
        mMassPropDamp(0.025),
        mStiffPropDamp(0.023),
        mProjectionFunction(),
        mPenaltyFunction(aPenaltyParams),
        mApplyPenalty(mPenaltyFunction),
        mApplyProjection(mProjectionFunction),
        mCubatureRule(),
        mCellStiffness()
    {
        this->initialize(aProblemParams);
    }

    /****************************************************************************//**
     *
     * \brief Destructor
     *
    ********************************************************************************/
    ~RayleighViscousForce()
    {
    }

    /****************************************************************************//**
     *
     * \brief Evaluate Rayleigh viscous forces
     *
     * The Rayleigh damping force vecotr is given by \f$\mathbf{f}_d = \left(\alpha
     * \mathbf{M} + \beta\mathbf{K}\right)\dot{\mathbf{u}}\f$, where \f$\mathbf{f}_d\f$
     * is the damping force vector, \f$\alpha\f$ is the mass proportional damping
     * coefficient, \f$\beta\f$ is the stiffness proportional damping coefficient
     * \f$\mathbf{K}\f$ is the stiffness matrix, \f$\mathbf{M}\f$ is the mass matrix
     * and \f$\dot{\mathbf{u}}\f$ is the velocity vector.
     *
     * Input arguments
     *
     * @param [in] aState state vector (e.g. displacements)
     * @param [in] aControl design variable vectors
     * @param [in] aConfig configuration vector (i.e. coordinates)
     * @param [in,out] aForce force vector
     * @param [in] aTimeStep current time step
     *
    ********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateType> & aState,
                  const Plato::ScalarMultiVectorT<ControlType> & aControl,
                  const Plato::ScalarArray3DT<ConfigType> & aConfig,
                  Plato::ScalarMultiVectorT<ResultType> & aForce,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        // elastic force functors
        Strain<mSpatialDim> tComputeStrains;
        LinearStress<mSpatialDim> tComputeStress(mCellStiffness);
        StressDivergence<mSpatialDim> tComputeStressDivergence;
        Plato::ComputeGradientWorkset<mSpatialDim> tComputeGradient;
        // Inertial force functors
        Plato::StateValues tComputeValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        // allocate local containers
        auto tNumCells = aState.extent(0);
        Plato::ScalarVector tCellVolume("Cell Volume", tNumCells);
        Plato::ScalarMultiVector tCellStrain("Cell Strain", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVector tCellStress("Cell Stress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVector tStateValues("Cell State Values", tNumCells, m_numDofsPerNode);
        Plato::ScalarArray3D tCellGradient("Cell Gradient", tNumCells, m_numNodesPerCell, mSpatialDim);
        Plato::ScalarMultiVector tCellElasticForce("Cell Elastic Force", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVector tCellInertialForce("Cell Inertial Force", tNumCells, m_numDofsPerCell);

        // Copy data from host into device
        auto tMassPropDamp = mMassPropDamp;
        auto tStiffPropDamp = mStiffPropDamp;
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto tQuadratureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tCellGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            // stiffness proportional damping
            tComputeStrains(aCellOrdinal, tCellStrain, aState, tCellGradient);
            tComputeStress(aCellOrdinal, tCellStress, tCellStrain);
            ControlType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            tApplyPenalty(aCellOrdinal, tCellDensity, tCellStress);
            tComputeStressDivergence(aCellOrdinal, tCellElasticForce, tCellStress, tCellGradient, tCellVolume);
            Plato::Experimental::axpy(aCellOrdinal, tStiffPropDamp, tCellElasticForce, aForce);
            // mass proportional damping
            tComputeValues(aCellOrdinal, tBasisFunctions, aState, tStateValues);
            tComputeInertialForces(aCellOrdinal, tCellVolume, tBasisFunctions, tStateValues, tCellInertialForce);
            tApplyPenalty(aCellOrdinal, tCellDensity, tCellInertialForce);
            Plato::Experimental::axpy(aCellOrdinal, tMassPropDamp, tCellInertialForce, aForce);
        }, "Evaluate-RayleighDampingForces");
    }
};
// class RayleighViscousForce

/****************************************************************************//**
 *
 * \brief Pure virtual class.
 *
 * Defines interface for an abstract dynamic scalar function (i.e. design criterion)
 *
********************************************************************************/
template<typename EvaluationType>
class AbstractDynamicsScalarFunction
{
public:
    virtual ~AbstractDynamicsScalarFunction()
    {
    }

    virtual std::string getName() const = 0;

    /****************************************************************************//**
     *
     * \brief Pure virtual function.
     *
     * Evaluate the scalar function (e.g. objective function)
     *
     * Input and output arguments
     *
     * @param [in] aTimeStep current time instance
     * @param [in] aState 2D array of cell's states
     * @param [in] aDotState 2D array of cell first-order time derivative of the states
     * @param [in] aDDotState 2D array of cell's second-order time derivative of the states
     * @param [in] aControl 2D array of cell's controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell's coordinates
     * @param [in,out] aResidual 2D array of cell's residuals
     *
    ********************************************************************************/
    virtual void
    evaluate(const Plato::Scalar & aTimeStep,
             const Plato::ScalarMultiVectorT<typename EvaluationType::StateUScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::StateVScalarType> & aDotState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::StateAScalarType> & aDDotState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> & aResult) const = 0;
};

/****************************************************************************//**
 *
 * \brief This class manages evaluations associated with a dynamic scalar function.
 *
********************************************************************************/
template<typename PhysicsT>
class DynamicScalarFunction : public WorksetBase<PhysicsT>
{
private:
    using WorksetBase<PhysicsT>::m_numNodes;
    using WorksetBase<PhysicsT>::m_numCells;
    using WorksetBase<PhysicsT>::m_numControl;
    using WorksetBase<PhysicsT>::m_numDofsPerCell;
    using WorksetBase<PhysicsT>::m_numDofsPerNode;
    using WorksetBase<PhysicsT>::m_numSpatialDims;
    using WorksetBase<PhysicsT>::m_numNodesPerCell;

    using WorksetBase<PhysicsT>::m_stateEntryOrdinal;
    using WorksetBase<PhysicsT>::m_configEntryOrdinal;
    using WorksetBase<PhysicsT>::m_controlEntryOrdinal;

    using Residual  = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::Residual;
    using GradientU = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientU;
    using GradientV = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientV;
    using GradientA = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientA;
    using GradientX = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientX;
    using GradientZ = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientZ;

    std::shared_ptr<AbstractDynamicsScalarFunction<Residual>>  mScalarFunctionValue;
    std::shared_ptr<AbstractDynamicsScalarFunction<GradientU>> mScalarFunctionGradientU;
    std::shared_ptr<AbstractDynamicsScalarFunction<GradientV>> mScalarFunctionGradientV;
    std::shared_ptr<AbstractDynamicsScalarFunction<GradientA>> mScalarFunctionGradientA;
    std::shared_ptr<AbstractDynamicsScalarFunction<GradientX>> mScalarFunctionGradientX;
    std::shared_ptr<AbstractDynamicsScalarFunction<GradientZ>> mScalarFunctionGradientZ;

    Plato::DataMap& mDataMap;

public:

  /**************************************************************************/
    DynamicScalarFunction(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap & aDataMap,
                          Teuchos::ParameterList& aParamList,
                          const std::string & aType ) :
          WorksetBase<PhysicsT>(aMesh),
          mDataMap(aDataMap)
  /**************************************************************************/
  {
        typename PhysicsT::FunctionFactory tFactory;
        mScalarFunctionValue = tFactory.template createScalarFunction<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aType);
        //mScalarFunctionGradientU = tFactory.template createScalarFunction<GradientU>(aMesh, aMeshSets, aDataMap, aParamList, aType);
        //mScalarFunctionGradientV = tFactory.template createScalarFunction<GradientV>(aMesh, aMeshSets, aDataMap, aParamList, aType);
        //mScalarFunctionGradientA = tFactory.template createScalarFunction<GradientA>(aMesh, aMeshSets, aDataMap, aParamList, aType);
        //mScalarFunctionGradientX = tFactory.template createScalarFunction<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aType);
        //mScalarFunctionGradientZ = tFactory.template createScalarFunction<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aType);
  }

  /**************************************************************************/
  Plato::Scalar value(const Plato::Scalar & aTimeStep,
                      const Plato::ScalarVector & aDisp,
                      const Plato::ScalarVector & aVel,
                      const Plato::ScalarVector & aAcc,
                      const Plato::ScalarVector & aControl) const
  /**************************************************************************/
  {
      using StateUScalar  = typename Residual::StateUScalarType;
      using StateVScalar  = typename Residual::StateVScalarType;
      using StateAScalar  = typename Residual::StateAScalarType;
      using ConfigScalar  = typename Residual::ConfigScalarType;
      using ResultScalar  = typename Residual::ResultScalarType;
      using ControlScalar = typename Residual::ControlScalarType;

      // workset state
      Plato::ScalarMultiVectorT<StateUScalar>
        tStateDispWS("state displacement workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aDisp, tStateDispWS);

      // workset state velocity
      Plato::ScalarMultiVectorT<StateVScalar>
      tStateVelWS("state velocity workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aVel, tStateVelWS);

      // workset state acceleration
      Plato::ScalarMultiVectorT<StateAScalar>
      tStateAccWS("state acceleration workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aAcc, tStateAccWS);

      // workset control
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result view
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);
      mDataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;

      // evaluate function
      mScalarFunctionValue->evaluate(aTimeStep, tStateDispWS, tStateVelWS, tStateAccWS, tControlWS, tConfigWS, tResult);

      // sum across elements
      auto tReturnVal = Plato::local_result_sum<Plato::Scalar>(m_numCells, tResult);

      return tReturnVal;
  }

  /**************************************************************************/
  Plato::ScalarVector gradient_x(const Plato::Scalar & aTimeStep,
                                 const Plato::ScalarVector & aDisp,
                                 const Plato::ScalarVector & aVel,
                                 const Plato::ScalarVector & aAcc,
                                 const Plato::ScalarVector & aControl) const
  /**************************************************************************/
  {
      using StateUScalar  = typename GradientX::StateUScalarType;
      using StateVScalar  = typename GradientX::StateVScalarType;
      using StateAScalar  = typename GradientX::StateAScalarType;
      using ConfigScalar  = typename GradientX::ConfigScalarType;
      using ResultScalar  = typename GradientX::ResultScalarType;
      using ControlScalar = typename GradientX::ControlScalarType;

      // workset state
      Plato::ScalarMultiVectorT<StateUScalar>
        tStateDispWS("state displacement workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aDisp, tStateDispWS);

      // workset state velocity
      Plato::ScalarMultiVectorT<StateVScalar>
      tStateVelWS("state velocity workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aVel, tStateVelWS);

      // workset state acceleration
      Plato::ScalarMultiVectorT<StateAScalar>
      tStateAccWS("state acceleration workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aAcc, tStateAccWS);

      // workset control
      Plato::ScalarMultiVectorT<ControlScalar>
        tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      // evaluate function
      mScalarFunctionGradientX->evaluate(aTimeStep, tStateDispWS, tStateVelWS, tStateAccWS, tControlWS, tConfigWS, tResult);

      // create and assemble to return view
      Plato::ScalarVector tObjGradientX("gradient configuration",m_numSpatialDims*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numSpatialDims>(m_numCells, m_configEntryOrdinal, tResult, tObjGradientX);
      Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      return tObjGradientX;
  }

  /**************************************************************************/
  Plato::ScalarVector gradient_u(const Plato::Scalar & aTimeStep,
                                 const Plato::ScalarVector & aDisp,
                                 const Plato::ScalarVector & aVel,
                                 const Plato::ScalarVector & aAcc,
                                 const Plato::ScalarVector & aControl) const
  /**************************************************************************/
  {
      using StateUScalar  = typename GradientU::StateUScalarType;
      using StateVScalar  = typename GradientU::StateVScalarType;
      using StateAScalar  = typename GradientU::StateAScalarType;
      using ConfigScalar  = typename GradientU::ConfigScalarType;
      using ResultScalar  = typename GradientU::ResultScalarType;
      using ControlScalar = typename GradientU::ControlScalarType;

      // workset state
      Plato::ScalarMultiVectorT<StateUScalar>
        tStateDispWS("state displacement state",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aDisp, tStateDispWS);

      // workset state velocity
      Plato::ScalarMultiVectorT<StateVScalar>
      tStateVelWS("state velocity workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aVel, tStateVelWS);

      // workset state acceleration
      Plato::ScalarMultiVectorT<StateAScalar>
      tStateAccWS("state acceleration workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aAcc, tStateAccWS);

      // workset control
      Plato::ScalarMultiVectorT<ControlScalar>
        tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      // evaluate function
      mScalarFunctionGradientU->evaluate(aTimeStep, tStateDispWS, tStateVelWS, tStateAccWS, tControlWS, tConfigWS, tResult);

      // create and assemble to return view
      Plato::ScalarVector tGradientDisp("gradient state displacement",m_numDofsPerNode*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells, m_stateEntryOrdinal, tResult, tGradientDisp);
      Plato::Scalar tFunctionValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      return tGradientDisp;
  }

  /**************************************************************************/
  Plato::ScalarVector gradient_v(const Plato::Scalar & aTimeStep,
                                 const Plato::ScalarVector & aDisp,
                                 const Plato::ScalarVector & aVel,
                                 const Plato::ScalarVector & aAcc,
                                 const Plato::ScalarVector & aControl) const
  /**************************************************************************/
  {
      using StateUScalar  = typename GradientV::StateUScalarType;
      using StateVScalar  = typename GradientV::StateVScalarType;
      using StateAScalar  = typename GradientV::StateAScalarType;
      using ConfigScalar  = typename GradientV::ConfigScalarType;
      using ResultScalar  = typename GradientV::ResultScalarType;
      using ControlScalar = typename GradientV::ControlScalarType;

      // workset state
      Plato::ScalarMultiVectorT<StateUScalar>
        tStateDispWS("state displacement workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aDisp, tStateDispWS);

      // workset state velocity
      Plato::ScalarMultiVectorT<StateVScalar>
      tStateVelWS("state velocity workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aVel, tStateVelWS);

      // workset state acceleration
      Plato::ScalarMultiVectorT<StateAScalar>
      tStateAccWS("state acceleration workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aAcc, tStateAccWS);

      // workset control
      Plato::ScalarMultiVectorT<ControlScalar>
        tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      // evaluate function
      mScalarFunctionGradientV->evaluate(aTimeStep, tStateDispWS, tStateVelWS, tStateAccWS, tControlWS, tConfigWS, tResult);

      // create and assemble to return view
      Plato::ScalarVector tGradientVel("gradient state velocity",m_numDofsPerNode*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells, m_stateEntryOrdinal, tResult, tGradientVel);
      Plato::Scalar tFunctionValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      return tGradientVel;
  }

  /**************************************************************************/
  Plato::ScalarVector gradient_a(const Plato::Scalar & aTimeStep,
                                 const Plato::ScalarVector & aDisp,
                                 const Plato::ScalarVector & aVel,
                                 const Plato::ScalarVector & aAcc,
                                 const Plato::ScalarVector & aControl) const
  /**************************************************************************/
  {
      using StateUScalar  = typename GradientA::StateUScalarType;
      using StateVScalar  = typename GradientA::StateVScalarType;
      using StateAScalar  = typename GradientA::StateAScalarType;
      using ConfigScalar  = typename GradientA::ConfigScalarType;
      using ResultScalar  = typename GradientA::ResultScalarType;
      using ControlScalar = typename GradientA::ControlScalarType;

      // workset state
      Plato::ScalarMultiVectorT<StateUScalar>
        tStateDispWS("state displacement state",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aDisp, tStateDispWS);

      // workset state velocity
      Plato::ScalarMultiVectorT<StateVScalar>
      tStateVelWS("state velocity workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aVel, tStateVelWS);

      // workset state acceleration
      Plato::ScalarMultiVectorT<StateAScalar>
      tStateAccWS("state acceleration workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aAcc, tStateAccWS);

      // workset control
      Plato::ScalarMultiVectorT<ControlScalar>
        tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      // evaluate function
      mScalarFunctionGradientA->evaluate(aTimeStep, tStateDispWS, tStateVelWS, tStateAccWS, tControlWS, tConfigWS, tResult);

      // create and assemble to return view
      Plato::ScalarVector tGradientAcc("gradient state acceleration",m_numDofsPerNode*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells, m_stateEntryOrdinal, tResult, tGradientAcc);
      Plato::Scalar tFunctionValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      return tGradientAcc;
  }

  /**************************************************************************/
  Plato::ScalarVector gradient_z(const Plato::Scalar & aTimeStep,
                                 const Plato::ScalarVector & aDisp,
                                 const Plato::ScalarVector & aVel,
                                 const Plato::ScalarVector & aAcc,
                                 const Plato::ScalarVector & aControl) const
  /**************************************************************************/
  {
      using StateUScalar  = typename GradientZ::StateUScalarType;
      using StateVScalar  = typename GradientZ::StateVScalarType;
      using StateAScalar  = typename GradientZ::StateAScalarType;
      using ConfigScalar  = typename GradientZ::ConfigScalarType;
      using ResultScalar  = typename GradientZ::ResultScalarType;
      using ControlScalar = typename GradientZ::ControlScalarType;

      // workset control
      Plato::ScalarMultiVectorT<ControlScalar>
       tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset state
      Plato::ScalarMultiVectorT<StateUScalar>
        tStateDispWS("state displacement workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aDisp, tStateDispWS);

      // workset state velocity
      Plato::ScalarMultiVectorT<StateVScalar>
      tStateVelWS("state velocity workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aVel, tStateVelWS);

      // workset state acceleration
      Plato::ScalarMultiVectorT<StateAScalar>
      tStateAccWS("state acceleration workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aAcc, tStateAccWS);

      // workset config
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result
      Plato::ScalarVectorT<ResultScalar>
        tResult("elastic energy",m_numCells);

      // evaluate function
      mScalarFunctionGradientZ->evaluate(aTimeStep, tStateDispWS, tStateVelWS, tStateAccWS, tControlWS, tConfigWS, tResult);

      // create and assemble to return view
      Plato::ScalarVector tGradientZ("gradient control",m_numNodes);
      Plato::assemble_scalar_gradient<m_numNodesPerCell>(m_numCells, m_controlEntryOrdinal, tResult, tGradientZ);
      Plato::Scalar tFunctionValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      return tGradientZ;
  }
};

/****************************************************************************//**
 *
 * \brief Pure virtual class.
 *
 * Defines interface for an abstract dynamic vector function (i.e. dynamic residual)
 *
********************************************************************************/
template<typename EvaluationType>
class AbstractDynamicsVectorFunction
{
public:
    /****************************************************************************//**
     *
     * \brief Destructor
     *
    ********************************************************************************/
    virtual ~AbstractDynamicsVectorFunction()
    {
    }

    /****************************************************************************//**
     *
     * \brief Pure virtual function.
     *
     * Returns reference to Omega_h mesh data base
     *
    ********************************************************************************/
    virtual Omega_h::Mesh& getMesh() const = 0;

    /****************************************************************************//**
     *
     * \brief Pure virtual function.
     *
     * Returns reference to Omega_h mesh side sets data base
     *
    ********************************************************************************/
    virtual Omega_h::MeshSets& getMeshSets() const = 0;

    /****************************************************************************//**
     *
     * \brief Pure virtual function.
     *
     * Evaluate time-dependent vector function
     *
     * Input and output arguments
     *
     * @param [in] aTimeStep current time instance
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aResidual 2D array of cell residuals
     *
    ********************************************************************************/
    virtual void
    evaluate(const Plato::Scalar & aTimeStep,
             const Plato::ScalarMultiVectorT<typename EvaluationType::StateUScalarType> & aDisp,
             const Plato::ScalarMultiVectorT<typename EvaluationType::StateVScalarType> & aVel,
             const Plato::ScalarMultiVectorT<typename EvaluationType::StateAScalarType> & aAcc,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> & aResult) = 0;

    /****************************************************************************//**
     *
     * @brief Pure virtual function.
     *
     * Set current state (i.e. displacements, velocities and accelerations) at \f$t=n\f$,
     * where \f$t\f$ is time and \f$n\f$ is the time index.
     *
     * @param [in] aOldDisp 2D array of previous cell displacements, \f$\ddot{\mathbf{u}}_{n-1}\f$
     * @param [in] aOldVel 2D array of previous cell velocities, \f$\dot{\mathbf{u}}_{n-1}\f$, 
     * @param [in] aOldAcc 2D array of previous cell accelerations, \f$\mathbf{u}_{n-1}\f$
     *
    ********************************************************************************/
    virtual void
    setPreviousState(const Plato::Scalar & aTimeStep,
                     const Plato::ScalarMultiVector & aDisp,
                     const Plato::ScalarMultiVector & aVel,
                     const Plato::ScalarMultiVector & aAcc) = 0;
};


/******************************************************************************//**
 *
 * \brief Evaluates the elastodynamics residual using the Newmark /f$\alpha/f$-method.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 * \f$(1-\alpha)\mathbf{f}_{n+1} - \alpha\mathbf{f}_{n} - \mathbf{M}\ddot{\mathbf{u}}_{n+1} 
 * - (1-\alpha)\mathbf{C}\dot{\mathbf{u}}_{n+1} + \alpha\mathbf{C}\dot{\mathbf{u}}_{n} 
 * - (1-\alpha)\mathbf{K}\mathbf{u}_{n+1} + \alpha\mathbf{K}\mathbf{u}_{n} = 0,
 *
 * where \f$n\f$ is the time step index, \f$\alpha\f$ introduces algorithmic damping
 * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\mathbf{f}\f$ is the external force 
 * vector, \f$\mathbf{M}$\f is the mass matrix, \f$\ddot{\mathbf{u}}\f$ is the 
 * acceleration vector, \f$\mathbf{C}\f$ is the mass matrix, \f$\dot{\mathbf{u}}\f$ 
 * is the velocity vector, \f$\mathbf{K}\f$ is the stiffness matrix and \f$\mathbf{u}\f$ 
 * is the displacement vector.
 *
 *********************************************************************************/
template<typename EvaluationType, class PenaltyFunctionType, class ProjectionType>
class ElastodynamicsResidual :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim, EvaluationType::NumControls>,
        public Plato::Experimental::AbstractDynamicsVectorFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Simplex<mSpaceDim>::m_numNodesPerCell;                 /*!< number of nodes per cell (i.e. element) */
    using Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms;   /*!< number of stress-strain components */
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerCell;  /*!< number of degrees of freedom per cell */
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerNode;  /*!< number of degrees of freedom per node */

    /*!< Automatic differentiation types */
    using StateUScalarType  = typename EvaluationType::StateUScalarType;
    using StateVScalarType  = typename EvaluationType::StateVScalarType;
    using StateAScalarType  = typename EvaluationType::StateAScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

private:
    Plato::Scalar mBeta;             /*!< Newmark beta coefficient - controls accuracy, stability and amount of algorithmic damping */
    Plato::Scalar mGamma;            /*!< Newmark gamma coefficient - controls accuracy, stability and amount of algorithmic damping */
    Plato::Scalar mAlpha;            /*!< algorithmic damping */
    Plato::Scalar mDensity;          /*!< material density */
    Plato::Scalar mMassPropDamp;     /*!< mass proportional damping coefficient */
    Plato::Scalar mStiffPropDamp;    /*!< stiffness proportional damping coefficient */
    Plato::Scalar mPrevTimeStep; /*!< previous iteration time step */

    Plato::ScalarMultiVector mPrevVel;  /*!< velocities at current time step */
    Plato::ScalarMultiVector mPrevAcc;  /*!< accelerations at current time step */
    Plato::ScalarMultiVector mPrevDisp; /*!< displacements at current time step */

    Omega_h::Mesh& mMesh;           /*!< omega_h mesh data base */
    Plato::DataMap& mDataMap;       /*!< map used to access physics data at runtime */
    Omega_h::MeshSets& mMeshSets;   /*!< omega_h mesh side sets data base */
    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> mCellStiffness; /*!< matrix of material constants (i.e. Lam\'e constants) */

    ProjectionType mProjectionFunction;                      /*!< projection operator */
    PenaltyFunctionType mPenaltyFunction;                    /*!< material penalization operator */
    Plato::ApplyPenalty<PenaltyFunctionType> mApplyPenalty;  /*!< applies material penalization operator */
    Plato::ApplyProjection<ProjectionType> mApplyProjection; /*!< applies projection operator */

    std::shared_ptr<Plato::BodyLoads<mSpaceDim, m_numDofsPerNode>> mBodyLoads; /*!< function used to compute body forces */
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule; /*!< instance with access to cubature rule */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, m_numDofsPerNode>> mBoundaryLoads; /*!< function used to compute boundary forces */

private:
    /**************************************************************************//**
     *
     * \brief Initialize material model and body forces (if active).
     *
     * Input arguments
     *
     * @param [in] aParamList Teuchos parameter list with material and external forces data.
     *
     *****************************************************************************/
    void initialize(Teuchos::ParameterList & aParamList)
    {
        if(aParamList.isSublist("Time Integration") == true)
        {
            auto tSublist = aParamList.sublist("Time Integration");
            mAlpha = tSublist.get<Plato::Scalar>("Alpha", -0.1);
        }
        Plato::Experimental::compute_newmark_damping_coeff(mAlpha, mBeta, mGamma);

        if(aParamList.isSublist("Material Model") == false)
        {
            Plato::Scalar tPoisson = 0.3;
            Plato::Scalar tModulus = 1.0;
            auto tMaterialModel = Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<mSpaceDim>(tModulus,tPoisson));
            mCellStiffness = tMaterialModel->getStiffnessMatrix();
        }
        else
        {
            // Parse inertial and viscous forces parameters
            auto tMaterialParamList = aParamList.sublist("Material Model");
            Plato::Experimental::is_parameter_defined("Density", tMaterialParamList);
            mDensity = tMaterialParamList.get<Plato::Scalar>("Density");
            Plato::Experimental::is_parameter_defined("Mass Proportional Damping", tMaterialParamList);
            mMassPropDamp = tMaterialParamList.get<Plato::Scalar>("Mass Proportional Damping", 0.);
            Plato::Experimental::is_parameter_defined("Stiffness Proportional Damping", tMaterialParamList);
            mStiffPropDamp = tMaterialParamList.get<Plato::Scalar>("Stiffness Proportional Damping", 0.);

            Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aParamList);
            auto tMaterialModel = tMaterialModelFactory.create();
            mCellStiffness = tMaterialModel->getStiffnessMatrix();
        }

        if(aParamList.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<mSpaceDim, m_numDofsPerNode>>(aParamList.sublist("Body Loads"));
        }

        if(aParamList.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim, m_numDofsPerNode>>(aParamList.sublist("Natural Boundary Conditions"));
        }
    }

public:
    /**************************************************************************//**
     *
     * \brief Constructor
     *
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh side set data base
     * @param [in] aDataMap map used to access physics data at runtime
     * @param [in] aProblemParams Teuchos parameter list that provides access 
     *             to material properties
     * @param [in] aPenaltyParams Teuchos parameter list that provides access 
     *             to parameters associated with the penalty function
     *
    ******************************************************************************/
    ElastodynamicsResidual(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap & aDataMap,
                           Teuchos::ParameterList & aProblemParams,
                           Teuchos::ParameterList & aPenaltyParams) :
            mBeta(0.3025),
            mGamma(0.6),
            mAlpha(-0.1),
            mDensity(1.0),
            mMassPropDamp(0.025),
            mStiffPropDamp(0.023),
            mPrevTimeStep(0),
            mMesh(aMesh),
            mMeshSets(aMeshSets),
            mDataMap(aDataMap),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>()),
            mBoundaryLoads(nullptr)
    {
        this->initialize(aProblemParams);
    }

    /**************************************************************************//**
     *
     * @brief Constructor
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh side set data base
     * @param [in] aDataMap map used to access physics data at runtime
     *
    ******************************************************************************/
    ElastodynamicsResidual(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap & aDataMap) :
            mBeta(0.3025),
            mGamma(0.6),
            mAlpha(-0.1),
            mDensity(1.0),
            mMassPropDamp(0.025),
            mStiffPropDamp(0.023),
            mPrevTimeStep(0),
            mMesh(aMesh),
            mMeshSets(aMeshSets),
            mDataMap(aDataMap),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>()),
            mBoundaryLoads(nullptr)
    {
    }

    /**************************************************************************//**
     *
     * \brief Destructor
     *
     *****************************************************************************/
    virtual ~ElastodynamicsResidual(){}

    /******************************************************************************//**
     *
     * @brief Set material density
     * @param [in] aInput material density
     *
    **********************************************************************************/
    void setMaterialDensity(const Plato::Scalar& aInput)
    {
        mDensity = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set mass proportional damping constant
     * @param [in] aInput mass proportional damping constant
     *
    **********************************************************************************/
    void setMassPropDamping(const Plato::Scalar& aInput)
    {
        mMassPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set stiffness proportional damping constant
     * @param [in] aInput stiffness proportional damping constant
     *
    **********************************************************************************/
    void setStiffPropDamping(const Plato::Scalar& aInput)
    {
        mStiffPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set algorithmic damping (\f$\alpha | \frac{-1}{3}\leq\alpha\leq{0}\f$.
     * @param [in] aInput algorithmic damping
     *
    **********************************************************************************/
    void setAlgorithmicDamping(const Plato::Scalar& aInput)
    {
        mAlpha = aInput;
        Plato::Experimental::compute_newmark_damping_coeff(mAlpha, mBeta, mGamma);
    }

    /******************************************************************************//**
     *
     * @brief Set material stiffness constants (i.e. Lame constants)
     * @param [in] aInput material stiffness constants
     *
    **********************************************************************************/
    void setMaterialStiffnessConstants(const Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms>& aInput)
    {
        mCellStiffness = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set isotropic linear elastic material constants (i.e. Lame constants)
     * @param [in] aYoungsModulus Young's modulus
     * @param [in] aPoissonsRatio Poisson's ratio
     *
    **********************************************************************************/
    void setIsotropicLinearElasticMaterial(const Plato::Scalar& aYoungsModulus, const Plato::Scalar& aPoissonsRatio)
    {
        Plato::IsotropicLinearElasticMaterial<mSpaceDim> tMaterialModel(aYoungsModulus, aPoissonsRatio);
        mCellStiffness = tMaterialModel.getStiffnessMatrix();
    }

    /****************************************************************************//**
     *
     * \brief Returns reference to Omega_h mesh data base
     *
    ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
     *
     * \brief Returns reference to Omega_h mesh side sets data base
     *
    ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /****************************************************************************//**
     *
     * \brief Evaluate elastodynamics residual.
     *
     * Input and output arguments
     *
     * @param [in] aTimeStep current time instance
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aResidual 2D array of cell residuals
     *
    ********************************************************************************/
    void evaluate(const Plato::Scalar & aTimeStep,
                  const Plato::ScalarMultiVectorT<StateUScalarType> & aStateU,
                  const Plato::ScalarMultiVectorT<StateVScalarType> & aStateV,
                  const Plato::ScalarMultiVectorT<StateAScalarType> & aStateA,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType> & aResult)
    {
        this->evaluateLHS(aTimeStep, aStateA, aControl, aConfig, aResult);
        this->evaluateRHS(aTimeStep, aStateA, aControl, aConfig, aResult);
    }

    /*****************************************************************************//**
     *
     * \brief Evaluate elastodynamics left-hand-side (LHS) using the generalized Newmark \f$\alpha\f$-method.
     *
     * The elastodynamics LHS formulation derived from the generalized Newmark 
     * \f$\alpha\f$-method is given by 
     *
     * \f$\mathbf{R}_{lhs} = \left(\mathbf{M} \Delta{t}_n\bar{\alpha}\gamma\mathbf{C}
     * + \Delta{t}_n^2\bar{\alpha}\beta\mathbf{K}\right)\ddot{\mathbf{u}}_{n},
     *
     * where \f$n=1,\dots,N\f$ and \f$N\f$ is the number of time steps, \f$\Delta{t}\f$
     * is the time step, \f$\gamma\f$ and \f$\beta\f$ are numerical factors used to 
     * damp higher modes in the solution \f$\alpha\f$ is a numerical factor used to
     * introduce algorithmic damping in the generalized Newmark \f$\alpha\f$-method 
     * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\mathbf{M}$\f, \f$\mathbf{C}\f$ and
     * \f$\mathbf{K}\f$ are the mass, damping and stiffness matrices. Finally, 
     * \f$\ddot{\mathbf{u}}\f$ is the acceleration vector.
     *
     * Input and output arguments:
     *
     * @param [in] aTimeStep current time step
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aOutput 2D array of cell LHS
     *
    **********************************************************************************/
    void evaluateLHS(const Plato::Scalar & aTimeStep,
                     const Plato::ScalarMultiVectorT<StateAScalarType> & aStateA,
                     const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                     const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                     Plato::ScalarMultiVectorT<ResultScalarType> & aOutput)
    {
        // Elastic force functors
        Strain<mSpaceDim> tComputeVoigtStrain;
        StressDivergence<mSpaceDim> tComputeStressDivergence;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        LinearStress<mSpaceDim> tComputeVoigtStress(mCellStiffness);
        // Inertial force functors
        Plato::StateValues tComputeValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        // Define strain-scalar AD-type (AD = automatic differentiation) //
        using StrainScalarType =
                typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateAScalarType, ConfigScalarType>;

        // Effective internal forces forces containers
        auto tNumCells = aStateA.extent(0);
        Plato::ScalarVectorT<ConfigScalarType> tVolume("CellVolume", tNumCells);
        Plato::ScalarMultiVectorT<ResultScalarType> tStress("CellStress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<StrainScalarType> tStrain("CellStrain", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<StateAScalarType> tValues("CellValues", tNumCells, m_numDofsPerNode);
        Plato::ScalarMultiVectorT<ResultScalarType> tElasticForces("CellElasticForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tInertialForces("CellInertialForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarArray3DT<ConfigScalarType> tGradient("CellGradient",tNumCells, m_numNodesPerCell, mSpaceDim);

        // Update constants used to compute effective internal forces, i.e. [\mathbf{K}^{eff}]\{\mathbf{D}\}_{n+1}
        auto tOnePlusAlpha = static_cast<Plato::Scalar>(1.0) + mAlpha;
        auto tDampingForcesConstant = aTimeStep * tOnePlusAlpha * mGamma;
        auto tElasticForcesConstant = aTimeStep * aTimeStep * tOnePlusAlpha * mBeta;

        // Copy member host constants into device
        auto tMassPropDamp = mMassPropDamp;
        auto tStiffPropDamp = mStiffPropDamp;

        // Copy member host functors and cubature rule into device
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute viscous elastic forces
            tComputeGradient(aCellOrdinal, tGradient, aConfig, tVolume);
            tVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeVoigtStrain(aCellOrdinal, tStrain, aStateA, tGradient);
            tComputeVoigtStress(aCellOrdinal, tStress, tStrain);
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl); /* apply projection operator */
            tApplyPenalty(aCellOrdinal, tCellDensity, tStress); /* apply penalty to stiffness proportional damping */
            tComputeStressDivergence(aCellOrdinal, tElasticForces , tStress, tGradient, tVolume);
            auto tConstant = tDampingForcesConstant * tStiffPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tElasticForces, aOutput);

            // Compute viscous inertial forces
            tComputeValues(aCellOrdinal, tBasisFunctions, aStateA, tValues);
            tComputeInertialForces(aCellOrdinal, tVolume, tBasisFunctions, tValues, tInertialForces);
            tApplyPenalty(aCellOrdinal, tCellDensity, tInertialForces); /* apply penalty to mass proportional damping */
            tConstant = tDampingForcesConstant * tMassPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tInertialForces, aOutput);

            // Add elastic and inertial forces to LHS
            Plato::Experimental::axpy(aCellOrdinal, tElasticForcesConstant, tElasticForces, aOutput);
            Plato::Experimental::axpy(aCellOrdinal, static_cast<Plato::Scalar>(1.0), tInertialForces, aOutput);
        }, "ElastodynamicsLHS");
    }

    /*****************************************************************************//**
     *
     * \brief Evaluate elastodynamics right-hand-side (RHS) using Newmark \f$/alpha\f$-method.
     *
     * The elastodynamics RHS formulation derived from the generalized Newmark
     * \f$/alpha\f$-method is given by
     *
     * \f$\mathbf{R}_{rhs} = -1 * \left( \mathbf{f}_{\alpha}(t) + \mathbf{b} -\alpha
     * \mathbf{C}\dot{\mathbf{u}}_{n-1} -\alpha\mathbf{K}\mathbf{u}_{n-1} + \bar{\alpha}
     * \mathbf{K}\bigg( \mathbf{u}_{n-1} + \Delta{t}_{n}\dot{\mathbf{u}}_{n-1} +
     * \Delta{t}_{n}^2\left(\frac{1}{2} - \beta\right)\ddot{\mathbf{u}}_{n-1} \bigg)
     * + \bar{\alpha}\mathbf{C}\bigg( \dot{\mathbf{u}}_{n-1} + \Delta{t}_{n}
     * \left(1 - \gamma\right)\ddot{\mathbf{u}}_{n-1} \bigg)\f$,
     *
     * where \f$\mathbf{f}_{\alpha}(t) = \left(1 + \alpha\right)\mathbf{f}_n - \alpha
     * \mathbf{f}_{n-1}.\f$ and \f$\mathbf{f}\f$ is the external force vector, \f$n\f$ is
     * the time step index, \f$\Delta{t}\f$ is the time step, \f$\gamma\f$ and \f$\beta\f$
     * are numerical factors used to damp higher modes in the solution, \f$\alpha\f$ is
     * a numerical factor used to introduce algorithmic damping such that \f$\frac{-1}{3}
     * \leq\alpha\leq{0}\f$, \f$\mathbf{M}$\f,\f$\mathbf{C}\f$ and \f$\mathbf{K}\f$
     * are the mass, damping and stiffness matrices. Finally, \f$\ddot{\mathbf{u}}\f$,
     * \f$\dot{\mathbf{u}}\f$ and \f$\mathbf{u}\f$ are the acceleration, velocity and
     * displacement vectors.
     *
     * Input and output arguments:
     *
     * @param [in] aTimeStep current time step
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aOutput 2D array of cell forces
     *
    **********************************************************************************/
    void evaluateRHS(const Plato::Scalar & aTimeStep,
                     const Plato::ScalarMultiVectorT<StateAScalarType> & aStateA,
                     const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                     const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                     Plato::ScalarMultiVectorT<ResultScalarType> & aOutput)
    {
        // Initialize elastic force functors
        Strain<mSpaceDim> tComputeVoigtStrain;
        StressDivergence<mSpaceDim> tComputeStressDivergence;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        LinearStress<mSpaceDim> tComputeVoigtStress(mCellStiffness);
        // Initialize inertial force functors
        Plato::StateValues tComputeValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        // Initialize non-AD type containers
        auto tNumCells = aStateA.extent(0);
        Plato::ScalarMultiVector tValues("CellValues", tNumCells, m_numDofsPerNode);
        Plato::ScalarMultiVector tIntrplValues("CellIntrplValues", tNumCells, m_numDofsPerCell);
        // Initialize configuration containers (e.g. gradient and volume)
        Plato::ScalarVectorT<ConfigScalarType> tVolume("CellVolume", tNumCells);
        Plato::ScalarArray3DT<ConfigScalarType> tGradient("CellGradient",tNumCells, m_numNodesPerCell, mSpaceDim);
        // Initialize stress and strain tensor containers
        Plato::ScalarMultiVectorT<ResultScalarType> tStress("CellStress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ConfigScalarType> tStrain("CellStrain", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ResultScalarType> tIntrplStress("CellIntrplStress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ConfigScalarType> tIntrplStrain("CellIntrplStrain", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ResultScalarType> tCurrentStress("CellCurrentStress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ConfigScalarType> tCurrentStrain("CellCurrentStrain", tNumCells, m_numVoigtTerms);
        // Initialize elastic and inertial force containers
        Plato::ScalarMultiVectorT<ResultScalarType> tIntrplElasticForces("CellIntrplElasticForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tCurrentElasticForces("CellCurrentElasticForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tIntrplMassPropDamping("CellIntrplMassPropDamping", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tIntrplStiffPropDamping("CellIntrplStiffPropDamping", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tCurrentMassPropDamping("CellCurrentMassPropDamping", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tCurrentStiffPropDamping("CellCurrentStiffPropDamping", tNumCells, m_numDofsPerCell);

        // Copy member host constants into device
        auto tBeta = mBeta;
        auto tGamma = mGamma;
        auto tAlpha = mAlpha;
        auto tMassPropDamp = mMassPropDamp;
        auto tStiffPropDamp = mStiffPropDamp;
        auto tOnePlusAlpha = static_cast<Plato::Scalar>(1.0) + mAlpha;

        // Copy member views and functors into device
        auto & tCurrentVel = mPrevVel;
        auto & tCurrentAcc = mPrevAcc;
        auto & tCurrentDisp = mPrevDisp;
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, aConfig, tVolume);
            tVolume(aCellOrdinal) *= tQuadratureWeight;
            // Compute viscous matrix times current velocities //
            tComputeVoigtStrain(aCellOrdinal, tCurrentStrain, tCurrentVel, tGradient);
            tComputeVoigtStress(aCellOrdinal, tCurrentStress, tCurrentStrain);
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl); /* apply projection operator */
            tApplyPenalty(aCellOrdinal, tCellDensity, tCurrentStress); /* apply penalty to stiffness proportional damping */
            tComputeStressDivergence(aCellOrdinal, tCurrentStiffPropDamping , tCurrentStress, tGradient, tVolume);
            auto tConstant = static_cast<Plato::Scalar>(-1.0) * tAlpha * tStiffPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tCurrentStiffPropDamping, aOutput);
            tComputeValues(aCellOrdinal, tBasisFunctions, tCurrentVel, tValues);
            tComputeInertialForces(aCellOrdinal, tVolume, tBasisFunctions, tValues, tCurrentMassPropDamping);
            tApplyPenalty(aCellOrdinal, tCellDensity, tCurrentMassPropDamping); /* apply penalty to mass proportional damping */
            tConstant = static_cast<Plato::Scalar>(-1.0) * tAlpha * tMassPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tCurrentMassPropDamping, aOutput);

            // Compute stiffness matrix times current displacements //
            tComputeVoigtStrain(aCellOrdinal, tStrain, tCurrentDisp, tGradient);
            tComputeVoigtStress(aCellOrdinal, tStress, tStrain);
            tApplyPenalty(aCellOrdinal, tCellDensity, tStress); /* apply penalty to current elastic force */
            tComputeStressDivergence(aCellOrdinal, tCurrentElasticForces /* elastic forces */, tStress, tGradient, tVolume);
            tConstant = static_cast<Plato::Scalar>(-1.0) * tAlpha;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tCurrentElasticForces, aOutput);

            // Compute interpolated velocities //
            Plato::Experimental::compute_interpolated_vel<m_numDofsPerCell>
                (aCellOrdinal, aTimeStep, tGamma, tCurrentVel, tCurrentAcc, tIntrplValues);
            // Compute viscous matrix times interpolated velocities //
            tComputeVoigtStrain(aCellOrdinal, tIntrplStrain, tIntrplValues, tGradient);
            tComputeVoigtStress(aCellOrdinal, tIntrplStress, tIntrplStrain);
            tApplyPenalty(aCellOrdinal, tCellDensity, tIntrplStress); /* apply penalty to stiffness proportional damping */
            tComputeStressDivergence(aCellOrdinal, tIntrplStiffPropDamping , tIntrplStress, tGradient, tVolume);
            tConstant = tOnePlusAlpha * tStiffPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tIntrplStiffPropDamping, aOutput);
            tComputeValues(aCellOrdinal, tBasisFunctions, tIntrplValues, tValues);
            tComputeInertialForces(aCellOrdinal, tVolume, tBasisFunctions, tValues, tIntrplMassPropDamping);
            tApplyPenalty(aCellOrdinal, tCellDensity, tIntrplMassPropDamping); /* apply penalty to mass proportional damping */
            tConstant = tOnePlusAlpha * tMassPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tIntrplMassPropDamping, aOutput);

            // Compute interpolated displacements //
            Plato::Experimental::compute_interpolated_disp<m_numDofsPerCell>
                (aCellOrdinal, aTimeStep, tBeta, tCurrentDisp, tCurrentVel, tCurrentAcc, tIntrplValues);
            // Compute stiffness matrix times interpolated displacements //
            tComputeVoigtStrain(aCellOrdinal, tIntrplStrain, tIntrplValues, tGradient);
            tComputeVoigtStress(aCellOrdinal, tIntrplStress, tIntrplStrain);
            tApplyPenalty(aCellOrdinal, tCellDensity, tIntrplStress); /* apply penalty to stiffness proportional damping */
            tComputeStressDivergence(aCellOrdinal, tIntrplElasticForces , tIntrplStress, tGradient, tVolume);
            Plato::Experimental::axpy(aCellOrdinal, tOnePlusAlpha, tIntrplElasticForces, aOutput);
        }, "ElastodynamicsRHS");

        if(mBodyLoads != nullptr)
        {
            mBodyLoads->get(mMesh, aStateA, aControl, aOutput);
        }

        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get( &mMesh, mMeshSets, aStateA, aControl, aOutput );
        }
    }

    /****************************************************************************//**
     *
     * \brief Set current state.
     *
     * Set current state (i.e. displacements, velocities and accelerations) at \f$t = n\f$,
     * where \f$t\f$ is time and \f$n\f$ is the time step index.
     *
     * @param aOldDisp   2D array of previous cell displacements, i.e.
     *                     \f$\{\ddot{\mathbf{D}}\}_{n-1}\f$, where \f$n\f$ is the current
     *                     time step index
     * @param aOldVel    2D array of previous cell velocities, i.e.
     *                     \f$\{\dot{\mathbf{D}}\}_{n-1}\f$, where \f$n\f$ is the current
     *                     time step index
     * @param aOldAcc    2D array of previous cell accelerations, i.e.
     *                     \f$\{\mathbf{D}\}_{n-1}\f$, where \f$n\f$ is the current
     *                     time step index
     *
    ********************************************************************************/
    void setPreviousState(const Plato::Scalar & aTimeStep,
                          const Plato::ScalarMultiVector & aDisp,
                          const Plato::ScalarMultiVector & aVel,
                          const Plato::ScalarMultiVector & aAcc)
    {
        mPrevVel = aVel;
        mPrevAcc = aAcc;
        mPrevDisp = aDisp;
        mPrevTimeStep = aTimeStep;
    }
};
// class ElastodynamicsResidual

/******************************************************************************//**
 *
 * \brief Evaluate the adjoint elastodynamics residual.
 *
 * Evaluate the adjoint elastodynamics residual. This adjoint equations are used
 * to compute the gradient of a criterion with respect to the control variables.
 * The adjoint elastodynamics residual is given by
 *
 * \f$\mathbf{R} = \bigg[\mathbf{M} + \Delta{t}_{N-n}\gamma\bar{\alpha}
 * \mathbf{C} + \Delta{t}^2_{N-n}\bar{\alpha}\beta\mathbf{K}\bigg]\Lambda_{n}
 * = -\left(\frac{\partial{f_{N-n}}}{\partial\mathbf{a}_{N-n}}\right)^{T}
 * - \Delta{t}^2_{N-n}\beta\Phi_{n} - \Delta{t}_{N-n}\gamma\ \Theta_{n}
 * - H(n)\bigg[\Delta{t}^2_{N-n+1}\left(\frac{1}{2} - \beta\right)\Big[
 * \bar{\alpha}\mathbf{K}\Lambda_{n-1} + \Phi_{n-1}\Big] + \Delta{t}_{N-n+1}
 * \left(1-\gamma\right)\Big[\bar{\alpha}\mathbf{C}\Lambda_{n-1} + \Theta_{n-1}
 * \Big] \bigg]\f$,
 *
 * where \f$n=0,\dots,N-1\f$ and \f$N\f$ is the number if time steps, \f$\alpha\f$
 * is algorithmic damping (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\bar{\alpha} =
 * 1+\alpha\f$, \f$\Delta{t}\f$ is the time step, \f$\gamma\f$ and \f$\beta\f$
 * are numerical factors used to damp higher modes in the solution, \f$\mathbf{M}$\f,
 * \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping and stiffness matrices.
 * Finally, \f$\Lambda\f$, \f$\Theta\f$ and \f$\Phi\f$ are the adjoint acceleration,
 * velocities and displacement vectors.
 *
 *********************************************************************************/
template<typename EvaluationType, class PenaltyFunctionType, class ProjectionType>
class AdjointElastodynamicsResidual :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim, EvaluationType::NumControls>,
        public Plato::Experimental::AbstractDynamicsVectorFunction<EvaluationType>
{
private:
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Simplex<mSpaceDim>::m_numNodesPerCell;                 /*!< number of nodes per cell (i.e. element) */
    using Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms;   /*!< number of stress-strain components */
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerCell;  /*!< number of degrees of freedom per cell */
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerNode;  /*!< number of degrees of freedom per node */

    /*!< Automatic differentiation types */
    using StateUScalarType  = typename EvaluationType::StateUScalarType;
    using StateVScalarType  = typename EvaluationType::StateVScalarType;
    using StateAScalarType  = typename EvaluationType::StateAScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

private:
    Plato::Scalar mBeta;           /*!< Newmark beta coefficient - controls accuracy, stability and amount of algorithmic damping */
    Plato::Scalar mGamma;          /*!< Newmark gamma coefficient - controls accuracy, stability and amount of algorithmic damping */
    Plato::Scalar mAlpha;          /*!< algorithmic damping */
    Plato::Scalar mDensity;        /*!< material density */
    Plato::Scalar mMassPropDamp;   /*!< mass proportional damping coefficient */
    Plato::Scalar mStiffPropDamp;  /*!< stiffness proportional damping coefficient */
    Plato::Scalar mPrevTimeStep;   /*!< previous iteration time step */

    Plato::ScalarMultiVector mPrevAdjVel;  /*!< adjoint velocities at the previous time step */
    Plato::ScalarMultiVector mPrevAdjAcc;  /*!< adjoint accelerations at the previous time step */
    Plato::ScalarMultiVector mPrevAdjDisp; /*!< adjoint displacements at the previous time step */

    Omega_h::Mesh& mMesh;           /*!< omega_h mesh data base */
    Plato::DataMap& mDataMap;       /*!< map used to access physics data at runtime */
    Omega_h::MeshSets& mMeshSets;   /*!< omega_h mesh side sets data base */
    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> mCellStiffness; /*!< matrix of material constants (i.e. Lam\'e constants) */

    ProjectionType mProjectionFunction;                      /*!< projection operator */
    PenaltyFunctionType mPenaltyFunction;                    /*!< material penalization operator */
    Plato::ApplyPenalty<PenaltyFunctionType> mApplyPenalty;  /*!< applies material penalization operator */
    Plato::ApplyProjection<ProjectionType> mApplyProjection; /*!< applies projection operator */

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule; /*!< instance with access to cubature rule */

private:
    /**************************************************************************//**
     *
     * \brief Initialize material model and body forces (if active).
     *
     * Input arguments
     *
     * @param [in] aParamList Teuchos parameter list with material and external forces data.
     *
     *****************************************************************************/
    void initialize(Teuchos::ParameterList & aParamList)
    {
        if(aParamList.isSublist("Time Integration") == true)
        {
            auto tSublist = aParamList.sublist("Time Integration");
            mAlpha = tSublist.get<Plato::Scalar>("Alpha", -0.1);
        }
        Plato::Experimental::compute_newmark_damping_coeff(mAlpha, mBeta, mGamma);

        if(aParamList.isSublist("Material Model") == false)
        {
            Plato::Scalar tPoisson = aParamList.get<Plato::Scalar>("Poissons Ratio", 0.3);
            Plato::Scalar tModulus = aParamList.get<Plato::Scalar>("Youngs Modulus", 1.0);
            auto tMaterialModel = Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<mSpaceDim>(tModulus,tPoisson));
            mCellStiffness = tMaterialModel->getStiffnessMatrix();
        }
        else
        {
            // Parse inertial and viscous forces parameters
            auto tMaterialParamList = aParamList.sublist("Material Model");
            Plato::Experimental::is_parameter_defined("Density", tMaterialParamList);
            mDensity = tMaterialParamList.get<Plato::Scalar>("Density");
            Plato::Experimental::is_parameter_defined("Mass Proportional Damping", tMaterialParamList);
            mMassPropDamp = tMaterialParamList.get<Plato::Scalar>("Mass Proportional Damping", 0.);
            Plato::Experimental::is_parameter_defined("Stiffness Proportional Damping", tMaterialParamList);
            mStiffPropDamp = tMaterialParamList.get<Plato::Scalar>("Stiffness Proportional Damping", 0.);

            // parse lame coefficients and create material model
            Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aParamList);
            auto tMaterialModel = tMaterialModelFactory.create();
            mCellStiffness = tMaterialModel->getStiffnessMatrix();
        }
    }

public:
    /**************************************************************************//**
     *
     * \brief Constructor
     *
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh side set data base
     * @param [in] aDataMap map used to access problem-specific data at runtime
     * @param [in] aProblemParams Teuchos parameter list that provides access
     *             to material properties
     * @param [in] aPenaltyParams Teuchos parameter list that provides access
     *             to parameters associated with the penalty function
     *
     *****************************************************************************/
    AdjointElastodynamicsResidual(Omega_h::Mesh& aMesh,
                                  Omega_h::MeshSets& aMeshSets,
                                  Plato::DataMap & aDataMap,
                                  Teuchos::ParameterList & aProblemParams,
                                  Teuchos::ParameterList & aPenaltyParams) :
            mBeta(0.3025),
            mGamma(0.6),
            mAlpha(-0.1),
            mDensity(1.0),
            mMassPropDamp(0.025),
            mStiffPropDamp(0.023),
            mPrevTimeStep(0),
            mMesh(aMesh),
            mMeshSets(aMeshSets),
            mDataMap(aDataMap),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    {
        this->initialize(aProblemParams);
    }

    /**************************************************************************//**
     *
     * \brief Constructor
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh side set data base
     * @param [in] aDataMap map used to access problem-specific data at runtime
     *
    ******************************************************************************/
    AdjointElastodynamicsResidual(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Plato::DataMap & aDataMap) :
            mBeta(0.3025),
            mGamma(0.6),
            mAlpha(-0.1),
            mDensity(1.0),
            mMassPropDamp(0.025),
            mStiffPropDamp(0.023),
            mPrevTimeStep(0),
            mMesh(aMesh),
            mMeshSets(aMeshSets),
            mDataMap(aDataMap),
            mProjectionFunction(),
            mPenaltyFunction(),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    {
    }

    /**************************************************************************//**
     *
     * \brief Destructor
     *
     *****************************************************************************/
    virtual ~AdjointElastodynamicsResidual(){}

    /******************************************************************************//**
     *
     * @brief Set material density
     * @param [in] aInput material density
     *
    **********************************************************************************/
    void setMaterialDensity(const Plato::Scalar& aInput)
    {
        mDensity = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set mass proportional damping constant
     * @param [in] aInput mass proportional damping constant
     *
    **********************************************************************************/
    void setMassPropDamping(const Plato::Scalar& aInput)
    {
        mMassPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set stiffness proportional damping constant
     * @param [in] aInput stiffness proportional damping constant
     *
    **********************************************************************************/
    void setStiffPropDamping(const Plato::Scalar& aInput)
    {
        mStiffPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set algorithmic damping (\f$\alpha | \frac{-1}{3}\leq\alpha\leq{0}\f$.
     * @param [in] aInput algorithmic damping
     *
    **********************************************************************************/
    void setAlgorithmicDamping(const Plato::Scalar& aInput)
    {
        mAlpha = aInput;
        Plato::Experimental::compute_newmark_damping_coeff(mAlpha, mBeta, mGamma);
    }

    /******************************************************************************//**
     *
     * @brief Set material stiffness constants (i.e. Lame constants)
     * @param [in] aInput material stiffness constants
     *
    **********************************************************************************/
    void setMaterialStiffnessConstants(const Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms>& aInput)
    {
        mCellStiffness = aInput;
    }

    /******************************************************************************//**
     *
     * @brief Set isotropic linear elastic material constants (i.e. Lame constants)
     * @param [in] aYoungsModulus Young's modulus
     * @param [in] aPoissonsRatio Poisson's ratio
     *
    **********************************************************************************/
    void setIsotropicLinearElasticMaterial(const Plato::Scalar& aYoungsModulus, const Plato::Scalar& aPoissonsRatio)
    {
        Plato::IsotropicLinearElasticMaterial<mSpaceDim> tMaterialModel(aYoungsModulus, aPoissonsRatio);
        mCellStiffness = tMaterialModel.getStiffnessMatrix();
    }

    /****************************************************************************//**
     *
     * \brief Returns reference to Omega_h mesh data base
     *
    ********************************************************************************/
    decltype(mMesh) getMesh() const
    {
        return (mMesh);
    }

    /****************************************************************************//**
     *
     * \brief Returns reference to Omega_h mesh side sets data base
     *
    ********************************************************************************/
    decltype(mMeshSets) getMeshSets() const
    {
        return (mMeshSets);
    }

    /****************************************************************************//**
     *
     * \brief Evaluate elastodynamics residual.
     *
     * Input and output arguments
     *
     * @param [in] aTimeStep current time instance
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aResidual 2D array of cell residuals
     *
    ********************************************************************************/
    void evaluate(const Plato::Scalar & aTimeStep,
                  const Plato::ScalarMultiVectorT<StateUScalarType> & aStateU,
                  const Plato::ScalarMultiVectorT<StateVScalarType> & aStateV,
                  const Plato::ScalarMultiVectorT<StateAScalarType> & aStateA,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType> & aResult)
    {
        // evaluate left-hand side contribution to residual
        this->evaluateLHS(aTimeStep, aStateA, aControl, aConfig, aResult);

        // evaluate right-hand side contribution to residual
        this->evaluateRHS(aTimeStep, aStateU, aStateV, aControl, aConfig, aResult);
    }

    /*****************************************************************************//**
     *
     * \brief Evaluate the left-hand-side contribution to the adjoint elastodynamics residual
     *
     * The left-hand-side (LHS) contribution to the adjoint elastodynamics residual
     * derived from the generalized Newmark-\f$\alpha\f$ method is given by
     *
     * \f$\mathbf{R}_{lhs} = \left(\mathbf{M} \Delta{t}_n\bar{\alpha}\gamma\mathbf{C}
     * + \Delta{t}_n^2\bar{\alpha}\beta\mathbf{K}\right)\Lambda_{n},
     *
     * where \f$n=0,\dots,N-1\f$ and \f$N\f$ is the number if time steps, \f$\Delta{t}\f$
     * is the time step, \f$\gamma\f$ and \f$\beta\f$ are numerical factors used to
     * damp higher modes in the solution \f$\alpha\f$ is a numerical factor used to
     * introduce algorithmic damping in the generalized Newmark \f$\alpha\f$-method
     * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\mathbf{M}$\f, \f$\mathbf{C}\f$ and
     * \f$\mathbf{K}\f$ are the mass, damping and stiffness matrices. Finally,
     * \f$\Lambda\f$ is the adjoint acceleration vector.
     *
     * Input and output arguments:
     *
     * @param [in] aTimeStep current time step
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aOutput 2D array of cell LHS
     *
    **********************************************************************************/
    void evaluateLHS(const Plato::Scalar & aTimeStep,
                     const Plato::ScalarMultiVectorT<StateAScalarType> & aStateA,
                     const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                     const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                     Plato::ScalarMultiVectorT<ResultScalarType> & aOutput)
    {
        // Elastic force functors
        Strain<mSpaceDim> tComputeVoigtStrain;
        StressDivergence<mSpaceDim> tComputeStressDivergence;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        LinearStress<mSpaceDim> tComputeVoigtStress(mCellStiffness);
        // Inertial force functors
        Plato::StateValues tComputeValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        // Define strain-scalar AD-type (AD = automatic differentiation) //
        using StrainScalarType =
                typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateAScalarType, ConfigScalarType>;

        // Effective internal forces forces containers
        auto tNumCells = aStateA.extent(0);
        Plato::ScalarVectorT<ConfigScalarType> tVolume("CellVolume", tNumCells);
        Plato::ScalarMultiVectorT<StateAScalarType> tValues("CellValues", tNumCells, m_numDofsPerNode);
        Plato::ScalarMultiVectorT<ResultScalarType> tStress("CellStress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<StrainScalarType> tStrain("CellStrain", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ResultScalarType> tElasticForces("CellElasticForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tInertialForces("CellInertialForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarArray3DT<ConfigScalarType> tGradient("CellGradient",tNumCells, m_numNodesPerCell, mSpaceDim);

        // Update constants used to compute LHS contribution to residual
        auto tOnePlusAlpha = static_cast<Plato::Scalar>(1.0) + mAlpha;
        auto tDampingForcesConstant = aTimeStep * tOnePlusAlpha * mGamma;
        auto tElasticForcesConstant = aTimeStep * aTimeStep * tOnePlusAlpha * mBeta;

        // Copy member host constants into device
        auto tMassPropDamp = mMassPropDamp;
        auto tStiffPropDamp = mStiffPropDamp;

        // Copy member host functors and cubature rule into device
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute viscous elastic forces
            tComputeGradient(aCellOrdinal, tGradient, aConfig, tVolume);
            tVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeVoigtStrain(aCellOrdinal, tStrain, aStateA, tGradient);
            tComputeVoigtStress(aCellOrdinal, tStress, tStrain);
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl); /* apply projection operator */
            tApplyPenalty(aCellOrdinal, tCellDensity, tStress); /* apply penalty to stiffness proportional damping */
            tComputeStressDivergence(aCellOrdinal, tElasticForces , tStress, tGradient, tVolume);
            auto tConstant = tDampingForcesConstant * tStiffPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tElasticForces, aOutput);

            // Compute viscous inertial forces
            tComputeValues(aCellOrdinal, tBasisFunctions, aStateA, tValues);
            tComputeInertialForces(aCellOrdinal, tVolume, tBasisFunctions, tValues, tInertialForces);
            tApplyPenalty(aCellOrdinal, tCellDensity, tInertialForces); /* apply penalty to mass proportional damping */
            tConstant = tDampingForcesConstant * tMassPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tInertialForces, aOutput);

            // Add elastic and inertial forces to LHS
            Plato::Experimental::axpy(aCellOrdinal, tElasticForcesConstant, tElasticForces, aOutput);
            Plato::Experimental::axpy(aCellOrdinal, static_cast<Plato::Scalar>(1.0), tInertialForces, aOutput);
        }, "AdjointElastodynamicsRHS");
    }

    /*****************************************************************************//**
     *
     * \brief Evaluate the right-hand-side contribution to the adjoint elastodynamics residual
     *
     * The right-hand-side (RHS) contribution to the adjoint elastodynamics residual
     * derived from the generalized Newmark-\f$\alpha\f$ method is given by
     *
     * \f$\mathbf{R}_{rhs} = -H\left(n\right) * \Bigg[ \Delta_{N-n}^{2}\left(\frac{1}{2}
     * - \beta\right)\bigg[ \bar{\alpha}\mathbf{K}\Lambda_{n-1} + \Phi_{n-1} \bigg]
     * + \Delta_{N-n}\left(1 - \gamma\right)\bigg[ \bar{\alpha}\mathbf{C}\Lambda_{n-1}
     * + \Theta_{n-1} \bigg] \Bigg]\f$,
     *
     * where \f$n=0,\dots,N-1\f$ and \f$N\f$ is the number if time steps, \f$\Delta{t}\f$
     * is the time step, \f$\gamma\f$ and \f$\beta\f$ are numerical factors used to
     * damp higher modes in the solution, \f$\bar{\alpha}=1+\alpha\f$, where \f$\alpha\f$
     * is a numerical factor used to introduce algorithmic damping in the generalized
     * Newmark-\f$\alpha\f$ method (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\mathbf{C}\f$
     * and \f$\mathbf{K}\f$ are the damping and stiffness matrices. Finally, \f$\Lambda\f$,
     * \f$\Theta\f$ and \f$\Phi\f$ are the adjoint acceleration, velocities and displacement
     * vectors.
     *
     * Input and output arguments:
     *
     * @param [in] aTimeStep current time step
     * @param [in] aState 2D array of cell solutions
     * @param [in] aControl 2D array of cell controls (e.g. design variables)
     * @param [in] aConfig 3D array of cell coordinates
     * @param [in,out] aOutput 2D array of cell forces
     *
    **********************************************************************************/
    void evaluateRHS(const Plato::Scalar & aTimeStep,
                     const Plato::ScalarMultiVectorT<StateUScalarType> & aStateU,
                     const Plato::ScalarMultiVectorT<StateVScalarType> & aStateV,
                     const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                     const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                     Plato::ScalarMultiVectorT<ResultScalarType> & aOutput)
    {
        // Initialize elastic force functors
        Strain<mSpaceDim> tComputeVoigtStrain;
        StressDivergence<mSpaceDim> tComputeStressDivergence;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        LinearStress<mSpaceDim> tComputeVoigtStress(mCellStiffness);
        // Initialize inertial force functors
        Plato::StateValues tComputeValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        // Initialize non-AD type containers
        auto tNumCells = aStateU.extent(0);
        Plato::ScalarMultiVector tValues("CellValues", tNumCells, m_numDofsPerNode);
        // Initialize configuration containers (e.g. gradient and volume)
        Plato::ScalarVectorT<ConfigScalarType> tVolume("CellVolume", tNumCells);
        Plato::ScalarArray3DT<ConfigScalarType> tGradient("CellGradient",tNumCells, m_numNodesPerCell, mSpaceDim);
        // Initialize stress and strain tensor containers
        Plato::ScalarMultiVectorT<ResultScalarType> tStress("CellStress", tNumCells, m_numVoigtTerms);
        Plato::ScalarMultiVectorT<ConfigScalarType> tStrain("CellStrain", tNumCells, m_numVoigtTerms);
        // Initialize elastic and inertial force containers
        Plato::ScalarMultiVectorT<ResultScalarType> tElasticForce("CellElasticForces", tNumCells, m_numDofsPerCell);
        Plato::ScalarMultiVectorT<ResultScalarType> tInertialForce("CellInertialForces", tNumCells, m_numDofsPerCell);

        // Update constants
        auto tCurrentAdjVelConstant = aTimeStep * mGamma;
        auto tCurrentAdjDispConstant = aTimeStep * aTimeStep * mBeta;
        auto tOnePlusAlpha = static_cast<Plato::Scalar>(1.0) + mAlpha;
        auto tPrevAdjVelConstant = mPrevTimeStep * (static_cast<Plato::Scalar>(1.0) - mGamma);
        auto tDampingForceConstant = tPrevAdjVelConstant * tOnePlusAlpha;
        auto tPrevAdjDispConstant = mPrevTimeStep * mPrevTimeStep * (static_cast<Plato::Scalar>(0.5) - mBeta);
        auto tElasticForcesConstant = tPrevAdjDispConstant * tOnePlusAlpha;

        // Copy member host constants into device
        auto tMassPropDamp = mMassPropDamp;
        auto tStiffPropDamp = mStiffPropDamp;

        // Copy member views and functors into device
        auto tPrevAdjVel = mPrevAdjVel;
        auto tPrevAdjAcc = mPrevAdjAcc;
        auto tPrevAdjDisp = mPrevAdjDisp;
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute cell gradients and volumes
            tComputeGradient(aCellOrdinal, tGradient, aConfig, tVolume);
            tVolume(aCellOrdinal) *= tQuadratureWeight;

            // Add elastic forces contribution, i.e. \Delta{t}^2_{N-n+1}(\frac{1}{2} - \beta)\bar{\alpha}\mathbf{K}\Lambda_{n-1}
            tComputeVoigtStrain(aCellOrdinal, tStrain, tPrevAdjAcc, tGradient);
            tComputeVoigtStress(aCellOrdinal, tStress, tStrain);
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl); /* apply projection operator */
            tApplyPenalty(aCellOrdinal, tCellDensity, tStress); /* apply penalty to stiffness proportional damping */
            tComputeStressDivergence(aCellOrdinal, tElasticForce , tStress, tGradient, tVolume);
            Plato::Experimental::axpy(aCellOrdinal, tElasticForcesConstant, tElasticForce, aOutput);

            // Add damping forces contribution, i.e. \Delta{t}_{N-n+1}(1 - \gamma)\bar{\alpha}\mathbf{C}\Lambda_{n-1}
            tComputeValues(aCellOrdinal, tBasisFunctions, tPrevAdjAcc, tValues);
            tComputeInertialForces(aCellOrdinal, tVolume, tBasisFunctions, tValues, tInertialForce);
            tApplyPenalty(aCellOrdinal, tCellDensity, tInertialForce); /* apply penalty to mass proportional damping */
            auto tConstant = tDampingForceConstant * tMassPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tInertialForce, aOutput);
            tConstant = tDampingForceConstant * tStiffPropDamp;
            Plato::Experimental::axpy(aCellOrdinal, tConstant, tElasticForce, aOutput);

            // Add previous adjoint displacements contribution, i.e. \Delta{t}^2_{N-n+1}(\frac{1}{2} - \beta)\Phi_{n-1}
            Plato::Experimental::axpy(aCellOrdinal, tPrevAdjDispConstant, tPrevAdjDisp, aOutput);

            // Add previous adjoint velocities contribution, i.e. \Delta{t}_{N-n+1}(1 - \gamma)\Theta_{n-1}
            Plato::Experimental::axpy(aCellOrdinal, tPrevAdjVelConstant, tPrevAdjVel, aOutput);

            // Add current adjoint velocities contribution, i.e. \Delta{t}_{N-n}\gamma\Theta_{n}
            Plato::Experimental::axpy(aCellOrdinal, tCurrentAdjVelConstant, aStateV, aOutput);

            // Add current adjoint displacements contribution, i.e. \Delta{t}^2_{N-n+1}\beta\Phi_{n}
            Plato::Experimental::axpy(aCellOrdinal, tCurrentAdjDispConstant, aStateU, aOutput);
        }, "AdjointElastodynamicsRHS");
    }

    /****************************************************************************//**
     *
     * \brief Set current adjoint state.
     *
     * Set current adjoint displacements, velocities and accelerations.
     *
     * @param [in] aOldDisp 2D array of previous cell adjoint displacements
     * @param [in] aOldVel 2D array of previous cell adjoint velocities
     * @param [in] aOldAcc 2D array of previous cell adjoint accelerations
     *
    ********************************************************************************/
    void setPreviousState(const Plato::Scalar & aTimeStep,
                          const Plato::ScalarMultiVector & aDisp,
                          const Plato::ScalarMultiVector & aVel,
                          const Plato::ScalarMultiVector & aAcc)
    {
        mPrevAdjVel = aVel;
        mPrevAdjAcc = aAcc;
        mPrevAdjDisp = aDisp;
        mPrevTimeStep = aTimeStep;
    }
};
// class AdjointElastodynamicsResidual

namespace ElastodynamicsFactory
{

/****************************************************************************//**
 *
 * \brief Create elastodynamics vector function.
 *
 * Create elastodynamics vector function, which is needed for the evaluation of
 * the residual and corresponding first- and second-order partial derivatives.
 *
 * @param [in] aMesh mesh data base
 * @param [in] aMeshSets mesh side sets data base
 * @param [in] aDataMap provides access to simulation data (e.g. displacements)
 * @param [in] aParamList Teuchos parameter list with user-defined input parameters
 * @param [in] aName dynamic vector function name
 *
********************************************************************************/
template<typename EvaluationType>
std::shared_ptr<AbstractDynamicsVectorFunction<EvaluationType>>
inline create_elastodynamics_residual(Omega_h::Mesh& aMesh,
                                      Omega_h::MeshSets& aMeshSets,
                                      Plato::DataMap& aDataMap,
                                      Teuchos::ParameterList& aParamList,
                                      const std::string & aName)
{
    Plato::Experimental::is_sublist_defined(aName, aParamList);
    auto tVecFuncSubList = aParamList.sublist(aName);
    Plato::Experimental::is_sublist_defined("Penalty Function", tVecFuncSubList);
    auto tPenaltyParams = tVecFuncSubList.sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
    if(tPenaltyType == "SIMP")
    {
        return std::make_shared<Plato::Experimental::ElastodynamicsResidual<EvaluationType, ::SIMP, Plato::HyperbolicTangentProjection>>
                (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
    }
    else if(tPenaltyType == "RAMP")
    {
        return std::make_shared<Plato::Experimental::ElastodynamicsResidual<EvaluationType, ::RAMP, Plato::HyperbolicTangentProjection>>
                (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
    }
    else
    {
        std::ostringstream tErrorMessage;
        tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n" << "\nMESSAGE: UNKNOWN 'TYPE = "
                << tPenaltyType.c_str() << "' SPECIFIED IN 'PENALTY FUNCTION' KEYWORD."
                << " USER SHOULD MAKE SURE THAT PROVIDED PENALTY FUNCTION FOR 'PDE CONSTRAINT = " << aName.c_str()
                << "' IS SUPPORTED.\n" << "**************\n\n";
        throw std::runtime_error(tErrorMessage.str().c_str());
    }
}

/****************************************************************************//**
 *
 * \brief Create elastodynamics adjoint vector function.
 *
 * Create elastodynamics adjoint vector function, which is needed for the evaluation
 * of the residual and corresponding first- and second-order partial derivatives.
 *
 * @param [in] aMesh mesh data base
 * @param [in] aMeshSets mesh side sets data base
 * @param [in] aDataMap provides access to simulation data (e.g. displacements)
 * @param [in] aParamList Teuchos parameter list with user-defined input parameters
 * @param [in] aName dynamic vector function name
 *
********************************************************************************/
template<typename EvaluationType>
std::shared_ptr<AbstractDynamicsVectorFunction<EvaluationType>>
inline create_adjoint_elastodynamics_residual(Omega_h::Mesh& aMesh,
                                              Omega_h::MeshSets& aMeshSets,
                                              Plato::DataMap& aDataMap,
                                              Teuchos::ParameterList& aParamList,
                                              const std::string & aName)
{
    Plato::Experimental::is_sublist_defined("Elastodynamics", aParamList);
    auto tVecFuncSubList = aParamList.sublist("Elastodynamics");
    Plato::Experimental::is_sublist_defined("Penalty Function", tVecFuncSubList);
    auto tPenaltyParams = tVecFuncSubList.sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type");
    if(tPenaltyType == "SIMP")
    {
        return std::make_shared<Plato::Experimental::AdjointElastodynamicsResidual<EvaluationType, ::SIMP, Plato::HyperbolicTangentProjection>>
                (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
    }
    else if(tPenaltyType == "RAMP")
    {
        return std::make_shared<Plato::Experimental::AdjointElastodynamicsResidual<EvaluationType, ::RAMP, Plato::HyperbolicTangentProjection>>
                (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
    }
    else
    {
        std::ostringstream tErrorMessage;
        tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n" << "\nMESSAGE: UNKNOWN 'TYPE = "
                << tPenaltyType.c_str() << "' SPECIFIED IN 'PENALTY FUNCTION' KEYWORD."
                << " USER SHOULD MAKE SURE THAT PROVIDED PENALTY FUNCTION FOR 'PDE CONSTRAINT = " << aName.c_str()
                << "' IS SUPPORTED.\n" << "**************\n\n";
        throw std::runtime_error(tErrorMessage.str().c_str());
    }
}

/****************************************************************************//**
 *
 * \brief Create linear elastic force function.
 *
 * @param [in] aProblemParam Teuchos parameter list with input data
 * @param [in] aPenaltyParam Teuchos parameter list with penalty function parameters
 *
********************************************************************************/
template<typename EvaluationType>
std::shared_ptr<AbstractForceFunction<EvaluationType>>
inline create_linear_elastic_force(Teuchos::ParameterList & aProblemParam, Teuchos::ParameterList & aPenaltyParam)
{
    const std::string tPenaltyType = aPenaltyParam.get<std::string>("Type");
    if(tPenaltyType == "SIMP")
    {
        return std::make_shared<LinearElasticForce<EvaluationType, ::SIMP, Plato::HyperbolicTangentProjection>>(aProblemParam, aPenaltyParam);
    }
    else if(tPenaltyType == "RAMP")
    {
        return std::make_shared<LinearElasticForce<EvaluationType, ::RAMP, Plato::HyperbolicTangentProjection>>(aProblemParam, aPenaltyParam);
    }
    else
    {
        std::ostringstream tErrorMessage;
        tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n"
                << " MESSAGE: UNKNOWN PENALTY FUNCTION WITH 'TYPE' = " << tPenaltyType.c_str()
                << "'. USER SHOULD MAKE SURE THIS PENALTY FUNCTION IS SUPPORTED.\n"
                << "**************\n\n";
        throw std::runtime_error(tErrorMessage.str().c_str());
    }
}

/****************************************************************************//**
 *
 * \brief Create rayleigh viscous force function.
 *
 * @param [in] aProblemParam Teuchos parameter list with input data
 * @param [in] aPenaltyParam Teuchos parameter list with penalty function parameters
 *
********************************************************************************/
template<typename EvaluationType>
std::shared_ptr<AbstractForceFunction<EvaluationType>>
inline create_rayleigh_viscous_force(Teuchos::ParameterList & aProblemParam, Teuchos::ParameterList & aPenaltyParam)
{
    const std::string tPenaltyType = aPenaltyParam.get<std::string>("Type");
    if(tPenaltyType == "SIMP")
    {
        return std::make_shared<RayleighViscousForce<EvaluationType, ::SIMP, Plato::HyperbolicTangentProjection>>(aProblemParam, aPenaltyParam);
    }
    else if(tPenaltyType == "RAMP")
    {
        return std::make_shared<RayleighViscousForce<EvaluationType, ::RAMP, Plato::HyperbolicTangentProjection>>(aProblemParam, aPenaltyParam);
    }
    else
    {
        std::ostringstream tErrorMessage;
        tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n"
                << " MESSAGE: UNKNOWN PENALTY FUNCTION WITH 'TYPE' = " << tPenaltyType.c_str()
                << "'. USER SHOULD MAKE SURE THIS PENALTY FUNCTION IS SUPPORTED.\n"
                << "**************\n\n";
        throw std::runtime_error(tErrorMessage.str().c_str());
    }
}

/****************************************************************************//**
 *
 * \brief Factory for elastodynamics-based simulations or optimization problems.
 *
********************************************************************************/
struct FunctionFactory
{
    /****************************************************************************//**
     *
     * \brief Create vector function.
     *
     * Create vector function that enables the residual, first- and second-order
     * partial derivatives evaluations.
     *
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh side sets data base
     * @param [in] aDataMap provides access to simulation data (e.g. displacements)
     * @param [in] aParamList Teuchos parameter list with user-defined input parameters
     * @param [in] aName vector function name
     *
    ********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<AbstractDynamicsVectorFunction<EvaluationType>>
    createVectorFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aParamList,
                         const std::string & aName)
    {
        if(aName == "Elastodynamics")
        {
            return (Plato::Experimental::ElastodynamicsFactory::create_elastodynamics_residual<EvaluationType>(aMesh, aMeshSets, aDataMap, aParamList, aName));
        }
        else if(aName == "Adjoint Elastodynamics")
        {
            return (Plato::Experimental::ElastodynamicsFactory::create_adjoint_elastodynamics_residual<EvaluationType>(aMesh, aMeshSets, aDataMap, aParamList, aName));
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                    << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n" << " MESSAGE: UNKNOWN 'PDE CONSTRAINT = "
                    << aName.c_str() << "' SPECIFIED IN THE 'PLATO PROBLEM' BLOCK."
                    << " USER SHOULD MAKE SURE THIS VECTOR FUNCTION IS SUPPORTED.\n" << "**************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }

    /****************************************************************************//**
     *
     * \brief Create scalar function.
     *
     * Create scalar function that enables criterion evaluation of its value and
     * corresponding first- and second-order partial derivatives evaluations.
     *
     * @param [in] aMesh mesh data base
     * @param [in] aMeshSets mesh side sets data base
     * @param [in] aDataMap provides access to simulation data (e.g. displacements)
     * @param [in] aParamList Teuchos parameter list with user-defined input parameters
     * @param [in] aName scalar function name
     *
    ********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<AbstractDynamicsScalarFunction<EvaluationType>>
    createScalarFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList & aParamList,
                         const std::string & aName)
    {
        std::ostringstream tErrorMessage;
        tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n" << " MESSAGE: UNKNOWN SCALAR FUNCTION 'TYPE = "
                << aName.c_str() << "' SPECIFIED IN THE 'PLATO PROBLEM' BLOCK."
                << " USER SHOULD MAKE SURE THIS SCALAR FUNCTION IS SUPPORTED.\n" << "**************\n\n";
        throw std::runtime_error(tErrorMessage.str().c_str());
    }

    /****************************************************************************//**
     *
     * \brief Create force function.
     *
     * Create force function - enables evaluation of internal forces (e.g. elastic,
     * inertial and viscous forces).
     *
     * @param [in] aParamList Teuchos parameter list with input data
     * @param [in] aName force function name
     *
    ********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<AbstractForceFunction<EvaluationType>>
    createForceFunction(Teuchos::ParameterList & aParamList, const std::string & aName)
    {
        Plato::Experimental::is_sublist_defined("Elastodynamics", aParamList);
        auto tElastodynamicsSublist = aParamList.sublist("Elastodynamics");
        Plato::Experimental::is_sublist_defined("Penalty Function", tElastodynamicsSublist);
        auto tPenaltyFuncSublist = tElastodynamicsSublist.sublist("Penalty Function");
        if(aName == "Linear Elastic Force")
        {
            return (Plato::Experimental::ElastodynamicsFactory::create_linear_elastic_force<EvaluationType>(aParamList, tPenaltyFuncSublist));
        }
        if(aName == "Rayleigh Viscous Force")
        {
            return (Plato::Experimental::ElastodynamicsFactory::create_rayleigh_viscous_force<EvaluationType>(aParamList, tPenaltyFuncSublist));
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                    << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n" << " MESSAGE: UNKNOWN FORCE FUNCTION WITH 'TYPE = "
                    << aName.c_str() << ". USER SHOULD MAKE SURE THIS FORCE FUNCTION IS SUPPORTED.\n" << "**************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
};
// struct FunctionFactory

} // namespace ElastodynamicsFactory

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class Elastodynamics: public SimplexMechanics<SpaceDim, NumControls>
{
public:
    using FunctionFactory = typename Plato::Experimental::ElastodynamicsFactory::FunctionFactory;
};
// class Elastodynamics

/****************************************************************************//**
 *
 * \brief Force function interface
 *
 * Manages allocation, evaluation and deallocation of a generic force function.
 *
********************************************************************************/
template<typename PhysicsT>
class ForceFunction : public WorksetBase<PhysicsT>
{
private:
    using WorksetBase<PhysicsT>::m_numNodes;
    using WorksetBase<PhysicsT>::m_numCells;
    using WorksetBase<PhysicsT>::m_numControl;
    using WorksetBase<PhysicsT>::m_numDofsPerCell;
    using WorksetBase<PhysicsT>::m_numDofsPerNode;
    using WorksetBase<PhysicsT>::m_numSpatialDims;
    using WorksetBase<PhysicsT>::m_numNodesPerCell;

    using Residual = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::Residual;
    std::shared_ptr<AbstractForceFunction<Residual>> mForceFunctionValue;

public:
    /****************************************************************************//**
     *
     * \brief Constructor
     *
     * Create a force function
     *
     * Input arguments
     *
     * @param [in] aMesh mesh data base
     * @param [in] aParamList Teuchos parameter list with input data
     * @param [in] aType force function type
     *
    ********************************************************************************/
    ForceFunction(Omega_h::Mesh& aMesh, Teuchos::ParameterList& aParamList, std::string& aType) :
        WorksetBase<PhysicsT>(aMesh)
    {
        typename PhysicsT::FunctionFactory tFactory;
        mForceFunctionValue = tFactory.template createForceFunction<Residual>(aParamList, aType);
    }

    /****************************************************************************//**
     *
     * \brief Destructor
     *
    ********************************************************************************/
    ~ForceFunction(){}

    /****************************************************************************//**
     *
     * \brief Evaluate force function
     *
     * Input arguments
     *
     * @param [in] aState state vector (e.g. displacement vector)
     * @param [in] aControl design variables vector
     * @param [in] aTimeStep current time step
     *
    ********************************************************************************/
    Plato::ScalarVector
    value(const Plato::ScalarVector & aState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar  = typename Residual::ConfigScalarType;
      using StateUScalar  = typename Residual::StateUScalarType;
      using ResultScalar  = typename Residual::ResultScalarType;
      using ControlScalar = typename Residual::ControlScalarType;

      Plato::ScalarMultiVectorT<StateUScalar> tStateWS("State Workset", m_numCells, m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", m_numCells, m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      Plato::ScalarMultiVectorT<ResultScalar> tForceWS("Force Workset", m_numCells, m_numDofsPerCell);
      mForceFunctionValue->evaluate( tStateWS, tControlWS, tConfigWS, tForceWS, aTimeStep );

      auto tTotalNumDofs = m_numDofsPerNode*m_numNodes;
      Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tAssembledForce("Assembled Force", tTotalNumDofs);
      WorksetBase<PhysicsT>::assembleResidual( tForceWS, tAssembledForce );

      return tAssembledForce;
    }
};
// class class ForceFunction

/******************************************************************************//**
 *
 * \brief Manages time-dependent vector function evaluations.
 *
 * Manages the evaluation of the residual and first- and second-order (if necessary)
 * partial derivatives with respect to states, controls and configuration variables
 * for all time-dependent vector functions.
 *
 *********************************************************************************/
template<typename PhysicsT>
class DynamicsVectorFunction : public WorksetBase<PhysicsT>
{
private:
    /*!< aid determine automatic differentiation types associated with respective evaluation type */
    using Residual  = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::Residual;
    using GradientU = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientU;
    using GradientV = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientV;
    using GradientA = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientA;
    using GradientX = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientX;
    using GradientZ = typename Plato::Experimental::DynamicsEvaluation<PhysicsT>::GradientZ;

    using WorksetBase<PhysicsT>::m_numNodes; /*!< number of nodes per cell (i.e. element) */
    using WorksetBase<PhysicsT>::m_numCells; /*!< number of cells */
    using WorksetBase<PhysicsT>::m_numControl; /*!< number of controls (i.e. design variables) */
    using WorksetBase<PhysicsT>::m_numDofsPerCell; /*!< number of degrees of freedom per cell */
    using WorksetBase<PhysicsT>::m_numDofsPerNode; /*!< number of degrees of freedom per node */
    using WorksetBase<PhysicsT>::m_numSpatialDims; /*!< number of spatial dimensions */
    using WorksetBase<PhysicsT>::m_numNodesPerCell; /*!< number of nodes per cell */

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = m_numSpatialDims * m_numNodesPerCell; /*!< number of configuration degrees of freedom */

    Plato::DataMap& mDataMap; /*!< map holding problem-specific data */

    /*!< time-dependent vector functions associated with an evaluation type */
    std::shared_ptr<AbstractDynamicsVectorFunction<Residual>>  mVectorFunctionResidual;
    std::shared_ptr<AbstractDynamicsVectorFunction<GradientU>> mVectorFunctionJacobianU;
    std::shared_ptr<AbstractDynamicsVectorFunction<GradientV>> mVectorFunctionJacobianV;
    std::shared_ptr<AbstractDynamicsVectorFunction<GradientA>> mVectorFunctionJacobianA;
    std::shared_ptr<AbstractDynamicsVectorFunction<GradientX>> mVectorFunctionJacobianX;
    std::shared_ptr<AbstractDynamicsVectorFunction<GradientZ>> mVectorFunctionJacobianZ;

public:
    /**************************************************************************//**
     *
     * \brief Constructor
     *
     * Input arguments:
     *
     * @param aMesh          mesh data base
     * @param aMeshSets      mesh side sets data base
     * @param aDataMap       map used to access problem-specific data at runtime
     * @param aParamList     parameter list with user defined input parameters
     * @param aVecFuncType   time-dependent vector function name
     *
    ******************************************************************************/
    DynamicsVectorFunction(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           Teuchos::ParameterList& aParamList,
                           std::string& aName) :
            WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;
        mVectorFunctionResidual = tFunctionFactory.template createVectorFunction<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aName);
        mVectorFunctionJacobianA = tFunctionFactory.template createVectorFunction<GradientA>(aMesh, aMeshSets, aDataMap, aParamList, aName);
    }

    /**************************************************************************//**
     * \brief Constructor
     *
     * Input arguments:
     *
     * @param aMesh    mesh data base
     * @param aDataMap map used to access problem-specific data at runtime
    ******************************************************************************/
    DynamicsVectorFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mVectorFunctionResidual(nullptr),
            mVectorFunctionJacobianU(nullptr),
            mVectorFunctionJacobianV(nullptr),
            mVectorFunctionJacobianA(nullptr),
            mVectorFunctionJacobianX(nullptr),
            mVectorFunctionJacobianZ(nullptr)
    {
    }

    /**************************************************************************//**
     *
     * \brief Destructor
     *
     *****************************************************************************/
    ~DynamicsVectorFunction()
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual residual evaluator
    * @param [in] aJacobian Jacobian evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<Plato::Experimental::AbstractDynamicsVectorFunction<Residual>>& aResidual,
                          const std::shared_ptr<Plato::Experimental::AbstractDynamicsVectorFunction<GradientA>>& aJacobian)
    {
        mVectorFunctionResidual = aResidual;
        mVectorFunctionJacobianA = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to state-u evaluator
    * @param [in] GradientU partial derivative with respect to state-u (e.g. displacements) evaluator
    *
    ******************************************************************************/
    void allocateJacobianU(const std::shared_ptr<Plato::Experimental::AbstractDynamicsVectorFunction<GradientU>>& GradientU)
    {
        mVectorFunctionJacobianU = GradientU;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to state-v evaluator
    * @param [in] GradientV partial derivative with respect to state-v (e.g. velocities) evaluator
    *
    ******************************************************************************/
    void allocateJacobianV(const std::shared_ptr<Plato::Experimental::AbstractDynamicsVectorFunction<GradientV>>& GradientV)
    {
        mVectorFunctionJacobianV = GradientV;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateJacobianZ(const std::shared_ptr<Plato::Experimental::AbstractDynamicsVectorFunction<GradientZ>>& aGradientZ)
    {
        mVectorFunctionJacobianZ = aGradientZ;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateJacobianX(const std::shared_ptr<Plato::Experimental::AbstractDynamicsVectorFunction<GradientX>>& aGradientX)
    {
        mVectorFunctionJacobianX = aGradientX;
    }

    void allocateJacobianU(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           Teuchos::ParameterList& aParamList,
                           std::string& aName)
    {
        mVectorFunctionJacobianU.reset();
        typename PhysicsT::FunctionFactory tFunctionFactory;
        mVectorFunctionJacobianU = tFunctionFactory.template createVectorFunction<GradientU>(aMesh, aMeshSets, aDataMap, aParamList, aName);
    }

    void allocateJacobianV(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           Teuchos::ParameterList& aParamList,
                           std::string& aName)
    {
        mVectorFunctionJacobianV.reset();
        typename PhysicsT::FunctionFactory tFunctionFactory;
        mVectorFunctionJacobianV = tFunctionFactory.template createVectorFunction<GradientV>(aMesh, aMeshSets, aDataMap, aParamList, aName);
    }

    void allocateJacobianZ(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           Teuchos::ParameterList& aParamList,
                           std::string& aName)
    {
        mVectorFunctionJacobianZ.reset();
        typename PhysicsT::FunctionFactory tFunctionFactory;
        mVectorFunctionJacobianZ = tFunctionFactory.template createVectorFunction<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aName);
    }

    void allocateJacobianX(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           Teuchos::ParameterList& aParamList,
                           std::string& aName)
    {
        mVectorFunctionJacobianX.reset();
        typename PhysicsT::FunctionFactory tFunctionFactory;
        mVectorFunctionJacobianX = tFunctionFactory.template createVectorFunction<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aName);
    }

    /**************************************************************************//**
     *
     * \brief Returns the total number of degrees of freedom
     *
     * The total number of degrees of freedom is given by \f$N_{dofs} =
     * N_{nodes}*n_{dofs}\f$, where \f$N_{dofs}\f$ is the total number of degrees
     * of freedom, \f$N_{nodes}\f$ is the total number of nodes and \f$n_{dofs}\f$
     * is the number of degrees of freedom per node.
     *
    ******************************************************************************/
    int size() const
    {
        return (m_numNodes*m_numDofsPerNode);
    }

    /**************************************************************************//**
     *
     * \brief Set current state (i.e. accelerations, velocities and displacements)
     *
     * @param aDisp    current set of displacements
     * @param aVel     current set of velocities
     * @param aAcc     current set of accelerations
     *
    ******************************************************************************/
    void setPreviousState(const Plato::Scalar & aTimeStep,
                          const Plato::ScalarVector & aOldDisp,
                          const Plato::ScalarVector & aOldVel,
                          const Plato::ScalarVector & aOldAcc)
    {
        // Current state workset
        Plato::ScalarMultiVector tOldDispWS("Old State Workset",m_numCells,m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aOldDisp, tOldDispWS);

        // Current velocity workset
        Plato::ScalarMultiVector tOldVelWS("Old Velocity Workset",m_numCells,m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aOldVel, tOldVelWS);

        // Current acceleration workset
        Plato::ScalarMultiVector tOldAccWS("Old Acceleration Workset",m_numCells,m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aOldAcc, tOldAccWS);

        mVectorFunctionResidual->setPreviousState(aTimeStep, tOldDispWS, tOldVelWS, tOldAccWS);
        mVectorFunctionJacobianA->setPreviousState(aTimeStep, tOldDispWS, tOldVelWS, tOldAccWS);

        if(mVectorFunctionJacobianZ != nullptr)
        {
            mVectorFunctionJacobianZ->setPreviousState(aTimeStep, tOldDispWS, tOldVelWS, tOldAccWS);
        }

        if(mVectorFunctionJacobianX != nullptr)
        {
            mVectorFunctionJacobianX->setPreviousState(aTimeStep, tOldDispWS, tOldVelWS, tOldAccWS);
        }

        if(mVectorFunctionJacobianU != nullptr)
        {
            mVectorFunctionJacobianU->setPreviousState(aTimeStep, tOldDispWS, tOldVelWS, tOldAccWS);
        }

        if(mVectorFunctionJacobianV != nullptr)
        {
            mVectorFunctionJacobianV->setPreviousState(aTimeStep, tOldDispWS, tOldVelWS, tOldAccWS);
        }
    }

    /**************************************************************************//**
     *
     * \brief Evaluate time-dependent vector function
     *
     * @param aState      current set of state (e.g. displacements) variables
     * @param aControl    current set of control (i.e. design variables)
     * @param aTimeStep   current time step
     *
    ******************************************************************************/
    Plato::ScalarVector
    value(const Plato::Scalar & aTimeStep,
          const Plato::ScalarVector & aStateU,
          const Plato::ScalarVector & aStateV,
          const Plato::ScalarVector & aStateA,
          const Plato::ScalarVector & aControl) const
    {
         using StateUScalar  = typename Residual::StateUScalarType;
         using StateVScalar  = typename Residual::StateVScalarType;
         using StateAScalar  = typename Residual::StateAScalarType;
         using ConfigScalar  = typename Residual::ConfigScalarType;
         using ResultScalar  = typename Residual::ResultScalarType;
         using ControlScalar = typename Residual::ControlScalarType;

         // Displacement State workset
         Plato::ScalarMultiVectorT<StateUScalar> tStateU_WS("Current StateU Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateU, tStateU_WS);

         // Velocity State workset
         Plato::ScalarMultiVectorT<StateVScalar> tStateV_WS("Current StateV Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateV, tStateV_WS);

         // Acceleration State workset
         Plato::ScalarMultiVectorT<StateAScalar> tStateA_WS("Current StateA Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateA, tStateA_WS);

         // Control workset
         Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", m_numCells, m_numNodesPerCell);
         WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

         // Configuration workset
         Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
         WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

         // Residual workset
         Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", m_numCells, m_numDofsPerCell);

         // evaluate function
         mVectorFunctionResidual->evaluate(aTimeStep, tStateU_WS, tStateV_WS, tStateA_WS, tControlWS, tConfigWS, tResidual);

         // create and assemble to return view
         const Plato::OrdinalType tTotalNumDofs = this->size();
         Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tAssembledResidual("Assembled Residual", tTotalNumDofs);
         WorksetBase<PhysicsT>::assembleResidual( tResidual, tAssembledResidual );

         return tAssembledResidual;
    }

    /**************************************************************************//**
     *
     * \brief Evaluate partial derivative of a time-dependent vector function with respect to configuration
     *
     * @param aState      current set of state (e.g. displacements) variables
     * @param aControl    current set of control (i.e. design variables)
     * @param aTimeStep   current time step
     *
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::Scalar & aTimeStep,
               const Plato::ScalarVector & aStateU,
               const Plato::ScalarVector & aStateV,
               const Plato::ScalarVector & aStateA,
               const Plato::ScalarVector & aControl) const
    {
        using StateUScalar  = typename GradientX::StateUScalarType;
        using StateVScalar  = typename GradientX::StateVScalarType;
        using StateAScalar  = typename GradientX::StateAScalarType;
        using ConfigScalar  = typename GradientX::ConfigScalarType;
        using ResultScalar  = typename GradientX::ResultScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;

        // Workset config
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Displacement State workset
        Plato::ScalarMultiVectorT<StateUScalar> tStateU_WS("Current StateU Workset", m_numCells, m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aStateU, tStateU_WS);

        // Velocity State workset
        Plato::ScalarMultiVectorT<StateVScalar> tStateV_WS("Current StateV Workset", m_numCells, m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aStateV, tStateV_WS);

        // Acceleration State workset
        Plato::ScalarMultiVectorT<StateAScalar> tStateA_WS("Current StateA Workset", m_numCells, m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aStateA, tStateA_WS);

        // Workset control
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", m_numCells, m_numNodesPerCell);
        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Configuration", m_numCells, m_numDofsPerCell);

        // evaluate function
        mVectorFunctionJacobianX->evaluate(aTimeStep, tStateU_WS, tStateV_WS, tStateA_WS, tControlWS, tConfigWS, tJacobian);

        // create return matrix
        auto tMesh = mVectorFunctionJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numSpatialDims>(&tMesh);

        // assembly to return matrix
        Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numDofsPerNode, m_numSpatialDims>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tJacobianMatEntries = tJacobianMat->entries();
        WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /**************************************************************************//**
     *
     * \brief Evaluate partial derivative with respect to state_u variable
     *
     * @param aState      current set of state (e.g. displacements) variables
     * @param aControl    current set of control (i.e. design variables)
     * @param aTimeStep   current time step
     *
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::Scalar & aTimeStep,
               const Plato::ScalarVector & aStateU,
               const Plato::ScalarVector & aStateV,
               const Plato::ScalarVector & aStateA,
               const Plato::ScalarVector & aControl) const
    {
         using StateUScalar  = typename GradientU::StateUScalarType;
         using StateVScalar  = typename GradientU::StateVScalarType;
         using StateAScalar  = typename GradientU::StateAScalarType;
         using ConfigScalar  = typename GradientU::ConfigScalarType;
         using ResultScalar  = typename GradientU::ResultScalarType;
         using ControlScalar = typename GradientU::ControlScalarType;

         // Workset config
         Plato::ScalarArray3DT<ConfigScalar>
             tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
         WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

         // Displacement State workset
         Plato::ScalarMultiVectorT<StateUScalar> tStateU_WS("Current StateU Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateU, tStateU_WS);

         // Velocity State workset
         Plato::ScalarMultiVectorT<StateVScalar> tStateV_WS("Current StateV Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateV, tStateV_WS);

         // Acceleration State workset
         Plato::ScalarMultiVectorT<StateAScalar> tStateA_WS("Current StateA Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateA, tStateA_WS);

         // Workset control
         Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
         WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

         // create return view
         Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian StateU",m_numCells,m_numDofsPerCell);

         // evaluate function
         mVectorFunctionJacobianU->evaluate(aTimeStep, tStateU_WS, tStateV_WS, tStateA_WS, tControlWS, tConfigWS, tJacobian);

         // create return matrix
         auto tMesh = mVectorFunctionJacobianU->getMesh();
         Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                 Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numDofsPerNode>(&tMesh);

         // assembly to return matrix
         Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numDofsPerNode, m_numDofsPerNode>
             tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

         auto tJacobianMatEntries = tJacobianMat->entries();
         WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, m_numDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

         return tJacobianMat;
    }

    /**************************************************************************//**
     *
     * \brief Evaluate partial derivative with respect to state_v variable
     *
     * @param aState      current set of state (e.g. displacements) variables
     * @param aControl    current set of control (i.e. design variables)
     * @param aTimeStep   current time step
     *
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_v(const Plato::Scalar & aTimeStep,
               const Plato::ScalarVector & aStateU,
               const Plato::ScalarVector & aStateV,
               const Plato::ScalarVector & aStateA,
               const Plato::ScalarVector & aControl) const
    {
         using StateUScalar  = typename GradientV::StateUScalarType;
         using StateVScalar  = typename GradientV::StateVScalarType;
         using StateAScalar  = typename GradientV::StateAScalarType;
         using ConfigScalar  = typename GradientV::ConfigScalarType;
         using ResultScalar  = typename GradientV::ResultScalarType;
         using ControlScalar = typename GradientV::ControlScalarType;

         // Workset config
         Plato::ScalarArray3DT<ConfigScalar>
             tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
         WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

         // Displacement State workset
         Plato::ScalarMultiVectorT<StateUScalar> tStateU_WS("Current StateU Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateU, tStateU_WS);

         // Velocity State workset
         Plato::ScalarMultiVectorT<StateVScalar> tStateV_WS("Current StateV Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateV, tStateV_WS);

         // Acceleration State workset
         Plato::ScalarMultiVectorT<StateAScalar> tStateA_WS("Current StateA Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateA, tStateA_WS);

         // Workset control
         Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
         WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

         // create return view
         Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian StateV",m_numCells,m_numDofsPerCell);

         // evaluate function
         mVectorFunctionJacobianV->evaluate(aTimeStep, tStateU_WS, tStateV_WS, tStateA_WS, tControlWS, tConfigWS, tJacobian);

         // create return matrix
         auto tMesh = mVectorFunctionJacobianV->getMesh();
         Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                 Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numDofsPerNode>(&tMesh);

         // assembly to return matrix
         Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numDofsPerNode, m_numDofsPerNode>
             tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

         auto tJacobianMatEntries = tJacobianMat->entries();
         WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, m_numDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

         return tJacobianMat;
    }

    /**************************************************************************//**
     *
     * \brief Evaluate partial derivative with respect to state_a variable
     *
     * @param aState      current set of state (e.g. displacements) variables
     * @param aControl    current set of control (i.e. design variables)
     * @param aTimeStep   current time step
     *
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_a(const Plato::Scalar & aTimeStep,
               const Plato::ScalarVector & aStateU,
               const Plato::ScalarVector & aStateV,
               const Plato::ScalarVector & aStateA,
               const Plato::ScalarVector & aControl) const
    {
         using StateUScalar  = typename GradientA::StateUScalarType;
         using StateVScalar  = typename GradientA::StateVScalarType;
         using StateAScalar  = typename GradientA::StateAScalarType;
         using ConfigScalar  = typename GradientA::ConfigScalarType;
         using ResultScalar  = typename GradientA::ResultScalarType;
         using ControlScalar = typename GradientA::ControlScalarType;

         // Workset config
         Plato::ScalarArray3DT<ConfigScalar>
             tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
         WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

         // Displacement State workset
         Plato::ScalarMultiVectorT<StateUScalar> tStateU_WS("Current StateU Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateU, tStateU_WS);

         // Velocity State workset
         Plato::ScalarMultiVectorT<StateVScalar> tStateV_WS("Current StateV Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateV, tStateV_WS);

         // Acceleration State workset
         Plato::ScalarMultiVectorT<StateAScalar> tStateA_WS("Current StateA Workset", m_numCells, m_numDofsPerCell);
         WorksetBase<PhysicsT>::worksetState(aStateA, tStateA_WS);

         // Workset control
         Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
         WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

         // create return view
         Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian StateA",m_numCells,m_numDofsPerCell);

         // evaluate function
         mVectorFunctionJacobianA->evaluate(aTimeStep, tStateU_WS, tStateV_WS, tStateA_WS, tControlWS, tConfigWS, tJacobian);

         // create return matrix
         auto tMesh = mVectorFunctionJacobianA->getMesh();
         Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                 Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numDofsPerNode>(&tMesh);

         // assembly to return matrix
         Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numDofsPerNode, m_numDofsPerNode>
             tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

         auto tJacobianMatEntries = tJacobianMat->entries();
         WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, m_numDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

         return tJacobianMat;
    }

    /**************************************************************************//**
     *
     * \brief Evaluate partial derivative of a time-dependent vector function with respect to control variables
     *
     * @param aState      current set of state (e.g. displacements) variables
     * @param aControl    current set of control (i.e. design variables)
     * @param aTimeStep   current time step
     *
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::Scalar & aTimeStep,
               const Plato::ScalarVector & aStateU,
               const Plato::ScalarVector & aStateV,
               const Plato::ScalarVector & aStateA,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl) const
    {
        using StateUScalar  = typename GradientZ::StateUScalarType;
        using StateVScalar  = typename GradientZ::StateVScalarType;
        using StateAScalar  = typename GradientZ::StateAScalarType;
        using ConfigScalar  = typename GradientZ::ConfigScalarType;
        using ResultScalar  = typename GradientZ::ResultScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;

      // Workset config
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // Displacement State workset
      Plato::ScalarMultiVectorT<StateUScalar> tStateU_WS("Current StateU Workset", m_numCells, m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aStateU, tStateU_WS);

      // Velocity State workset
      Plato::ScalarMultiVectorT<StateVScalar> tStateV_WS("Current StateV Workset", m_numCells, m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aStateV, tStateV_WS);

      // Acceleration State workset
      Plato::ScalarMultiVectorT<StateAScalar> tStateA_WS("Current StateA Workset", m_numCells, m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aStateA, tStateA_WS);

      // create result
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Control",m_numCells,m_numDofsPerCell);

      // evaluate function
      mVectorFunctionJacobianZ->evaluate(aTimeStep, tStateU_WS, tStateV_WS, tStateA_WS, tControlWS, tConfigWS, tJacobian);

      // create return matrix
      auto tMesh = mVectorFunctionJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numControl, m_numDofsPerNode>(&tMesh);

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numControl, m_numDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      WorksetBase<PhysicsT>::assembleTransposeJacobian(m_numDofsPerCell, m_numNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }
}; // class DynamicsVectorFunction

/*************************************************************************//**
 *
 * \brief Set initial condition vector
 *
 * Set initial condition vector at time \f$t = 0\f$. The initial conditions are
 * defined by the user as Dirichlet boundary conditions.
 *
 * Input and output arguments
 *
 * @param [in] aDofs degrees of freedom associated with initial conditions
 * @param [in] aValues values associated with initial conditions' degrees of freedom
 * @param [in,out] aOutput scalar vector of initial conditions
 *
******************************************************************************/
template<typename OutputType>
inline void set_initial_conditions_vector(const Plato::LocalOrdinalVector & aDofs,
                                          const Plato::ScalarVector & aValues,
                                          OutputType & aOutput)
{
    auto tNumBCs = aDofs.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBCs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tDofOrdinal = aDofs(aIndex);
        aOutput(tDofOrdinal) = aValues(aIndex);
    },"Set initial condition vector");
}

class AbstractDynamicsProblem
{
public:
    virtual ~AbstractDynamicsProblem()
    {
    }

    virtual void
    getState(Plato::ScalarMultiVector & aStateU,
             Plato::ScalarMultiVector & aStateV,
             Plato::ScalarMultiVector & aStateA)=0;
    virtual void
    setState(const Plato::ScalarMultiVector & aStateU,
             const Plato::ScalarMultiVector & aStateV,
             const Plato::ScalarMultiVector & aStateA)=0;

    // Functions associated with residual
    virtual void
    applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)=0;

    virtual void solve(const Plato::ScalarVector & aControl)=0;

    // Functions associated with constraint
    virtual Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl)=0;

    virtual Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl,
                    const Plato::ScalarMultiVector & aStateU,
                    const Plato::ScalarMultiVector & aStateV,
                    const Plato::ScalarMultiVector & aStateA)=0;

    virtual Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl,
                       const Plato::ScalarMultiVector & aStateU,
                       const Plato::ScalarMultiVector & aStateV,
                       const Plato::ScalarMultiVector & aStateA)=0;

    virtual Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl,
                        const Plato::ScalarMultiVector & aStateU,
                        const Plato::ScalarMultiVector & aStateV,
                        const Plato::ScalarMultiVector & aStateA)=0;

    // Functions associated with objective
    virtual Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl)=0;

    virtual Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl,
                   const Plato::ScalarMultiVector & aStateU,
                   const Plato::ScalarMultiVector & aStateV,
                   const Plato::ScalarMultiVector & aStateA)=0;

    virtual Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl,
                      const Plato::ScalarMultiVector & aStateU,
                      const Plato::ScalarMultiVector & aStateV,
                      const Plato::ScalarMultiVector & aStateA)=0;

    virtual Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl,
                       const Plato::ScalarMultiVector & aStateU,
                       const Plato::ScalarMultiVector & aStateV,
                       const Plato::ScalarMultiVector & aStateA)=0;

    Plato::DataMap mDataMap;
    decltype(mDataMap)& getDataMap()
    {
        return mDataMap;
    }
};
// class AbstractDynamicsProblem

template<typename SimplexPhysics>
class ElastodynamicsProblem: public Plato::Experimental::AbstractDynamicsProblem
{
    static constexpr Plato::OrdinalType mSpatialDim = SimplexPhysics::m_numSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::m_numDofsPerNode;

    Plato::Scalar mBeta;
    Plato::Scalar mGamma;
    Plato::Scalar mAlpha;

    Plato::OrdinalType mNumStates;
    Plato::OrdinalType mNumConfig;
    Plato::OrdinalType mNumControls;
    Plato::OrdinalType mNumTimeSteps;
    Plato::OrdinalType mNumIterationsAmgX;

    Plato::ScalarVector mBcValues;
    Plato::LocalOrdinalVector mBcDofs;

    Plato::ScalarVector mResidual;
    Plato::ScalarVector mExternalForce;

    Plato::ScalarMultiVector mVelocities;
    Plato::ScalarMultiVector mDisplacements;
    Plato::ScalarMultiVector mAccelerations;

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian;

    std::vector<Plato::Scalar> mTimeSteps;

    // required
    std::shared_ptr<DynamicsVectorFunction<SimplexPhysics>> mEquality;

    // optional
    std::shared_ptr<ForceFunction<SimplexPhysics>> mElasticForce;
    std::shared_ptr<ForceFunction<SimplexPhysics>> mViscousForce;
    std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> mObjective;
    std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> mConstraint;
    std::shared_ptr<DynamicsVectorFunction<SimplexPhysics>> mAdjointEquation;

private:
    /*************************************************************************//**
     *
     * \brief Initialize objective function and related dependencies.
     *
     * Initialize objective function, adjoint problem and force functors needed
     * for the adjoint displacement and velocity vectors transition equations.
     *
     * Input arguments
     *
     * @param aMesh       Omega_h mesh data base
     * @param MeshSets    Omega_h mesh side sets data base
     * @param aParamList  Teuchos parameter list with user define problem input information
     *
    ******************************************************************************/
    void initializeObjective(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    {
        if(aParamList.isType<std::string>("Objective"))
        {
            std::string tName = aParamList.get<std::string>("Objective");
            mObjective = std::make_shared<DynamicScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);

            tName = "Linear Elastic Force";
            mElasticForce = std::make_shared<ForceFunction<SimplexPhysics>>(aMesh, aParamList, tName);
            tName = "Rayleigh Viscous Force";
            mViscousForce = std::make_shared<ForceFunction<SimplexPhysics>>(aMesh, aParamList, tName);

            tName = "Adjoint Elastodynamics";
            mAdjointEquation = std::make_shared<DynamicsVectorFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);
            auto tMyPDE = aParamList.get<std::string>("PDE Constraint");
            mEquality->allocateJacobianZ(aMesh, aMeshSets, mDataMap, aParamList, tName);
            mEquality->allocateJacobianX(aMesh, aMeshSets, mDataMap, aParamList, tName);
        }
    }

    /*************************************************************************//**
     *
     * \brief Initialize constraint and related dependencies.
     *
     * Input arguments
     *
     * @param aMesh       Omega_h mesh data base
     * @param MeshSets    Omega_h mesh side sets data base
     * @param aParamList  Teuchos parameter list with user define problem input information
     *
    ******************************************************************************/
    void initializeConstraint(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    {
        if(aParamList.isType<std::string>("Constraint"))
        {
            std::string tName = aParamList.get<std::string>("Constraint");
            mConstraint = std::make_shared<DynamicScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);
        }
    }

    /*************************************************************************//**
     *
     * \brief Initialize forward and adjoint problems.
     *
     * Input arguments
     *
     * @param aMesh       Omega_h mesh data base
     * @param MeshSets    Omega_h mesh side sets data base
     * @param aParamList  Teuchos parameter list with user define problem input information
     *
    ******************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    {
        Plato::Experimental::is_sublist_defined("Time Integration", aParamList);
        auto tSublist = aParamList.sublist("Time Integration");
        auto tTimeStep = tSublist.get<Plato::Scalar>("Time Step", 1e-4);
        mNumTimeSteps = tSublist.get<Plato::Scalar>("Number Time Steps", 10);
        mTimeSteps.clear();
        mTimeSteps.resize(mNumTimeSteps + 1);
        std::fill(mTimeSteps.begin() + 1, mTimeSteps.end(), tTimeStep);
        mAlpha = tSublist.get<Plato::Scalar>("Alpha", -0.1);
        Plato::Experimental::compute_newmark_damping_coeff(mAlpha, mBeta, mGamma);

        Plato::Experimental::is_parameter_defined("PDE Constraint", aParamList);
        auto tMyPDE = aParamList.get<std::string>("PDE Constraint");
        mEquality = std::make_shared<DynamicsVectorFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tMyPDE);

        this->initializeObjective(aMesh, aMeshSets, aParamList);
        this->initializeConstraint(aMesh, aMeshSets, aParamList);

        // Initialize state multivectors //
        const Plato::OrdinalType tSize = mNumTimeSteps + 1;
        mVelocities = Plato::ScalarMultiVector("Velocities", tSize, mNumStates);
        mDisplacements = Plato::ScalarMultiVector("Displacements", tSize, mNumStates);
        mAccelerations = Plato::ScalarMultiVector("Accelerations", tSize, mNumStates);

        Plato::Experimental::is_sublist_defined("Essential Boundary Conditions", aParamList);
        Plato::EssentialBCs<SimplexPhysics> tDirichletConditions(aParamList.sublist("Essential Boundary Conditions",false));
        tDirichletConditions.get(aMeshSets, mBcDofs, mBcValues);
        this->setInitialConditions(aMeshSets, aParamList);
    }

    /*************************************************************************//**
     *
     * \brief Read initial conditions, i.e. initial displacements and velocities.
     *
     * Input arguments
     *
     * @param aMeshSets Omega_h mesh side sets data base
     * @param aParamList Teuchos parameter list with input data
     *
    ******************************************************************************/
    void setInitialConditions(Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    {
        Plato::ScalarVector tInitialDispValues;
        Plato::LocalOrdinalVector tInitialDispDofs;
        Plato::Experimental::is_sublist_defined("Initial Displacements", aParamList);
        Plato::EssentialBCs<SimplexPhysics> tInitialDispConditions(aParamList.sublist("Initial Displacements",false));
        tInitialDispConditions.get(aMeshSets, tInitialDispDofs, tInitialDispValues);
        const Plato::OrdinalType tINITIAL_TIME_STEP_INDEX = 0;
        auto tInitialDispVector = Kokkos::subview(mDisplacements, tINITIAL_TIME_STEP_INDEX, Kokkos::ALL());
        Plato::Experimental::set_initial_conditions_vector(tInitialDispDofs, tInitialDispValues, tInitialDispVector);

        Plato::ScalarVector tInitialVelValues;
        Plato::LocalOrdinalVector tInitialVelDofs;
        Plato::Experimental::is_sublist_defined("Initial Velocities", aParamList);
        Plato::EssentialBCs<SimplexPhysics> tInitialVelConditions(aParamList.sublist("Initial Velocities",false));
        tInitialVelConditions.get(aMeshSets, tInitialVelDofs, tInitialVelValues);
        auto tInitialVelVector = Kokkos::subview(mVelocities, tINITIAL_TIME_STEP_INDEX, Kokkos::ALL());
        Plato::Experimental::set_initial_conditions_vector(tInitialVelDofs, tInitialVelValues, tInitialVelVector);
    }

    /*************************************************************************//**
     *
     * \brief Solve system of equations, /f$\mathbf{A}\mathbf{x} = \mathbf{b}/f$.
     *
     * Solve system of equations, /f$\mathbf{A}\mathbf{x} = \mathbf{b}/f$, where
     * /f$\mathbf{x}/f$ is the solution vector, /f$\mathbf{b}/f$ is the right-hand-side
     * vector and /f$\mathbf{A}/f$ is a matrix.
     *
     * Input and output arguments
     *
     * @param aMatrix scalar matrix /f$\mathbf{A}/f$
     * @param aForce scalar right-hand-side vector /f$\mathbf{b}/f$
     * @param aSolution solution /f$\mathbf{x}/f$
     *
    ******************************************************************************/
    template<typename MatrixType, typename ForceType, typename SolutionType>
    void solve(MatrixType & aMatrix, ForceType & aForce, SolutionType & aSolution)
    {
#ifdef HAVE_AMGX
        using AmgXLinearProblem = lgr::AmgXSparseLinearProblem<Plato::OrdinalType, mNumDofsPerNode>;
        auto tConfigString = AmgXLinearProblem::getConfigString();
        auto tSolver = std::make_shared<AmgXLinearProblem>(aMatrix, aSolution, aForce, tConfigString);
        tSolver->solve();
#endif
    }

    /*************************************************************************//**
     *
     * \brief Compute initial accelerations, /f$\dots{\mathbf{u}}_{t=0}/f$.
     *
     * Compute initial accelerations vector /f$\dots{\mathbf{u}}_{t=0}/f$ by solving
     * the equation of motion
     *
     * \f$\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{K}\mathbf{u}
     * - \mathbf{f} = 0,\f$
     *
     * where \f$\mathbf{f}\f$ is the external force vector, \f$\mathbf{M}\f$,
     * \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass. damping and stiffness
     * matrices, and \f$\ddot{\mathbf{u}}\f$, \f$\dot{\mathbf{u}}\f$ and
     * \f$\mathbf{u}\f$ are the acceleration, velocity and displacement vectors.
     *
     * Input and output parameters:
     *
     * @param [in] aControl control variables (i.e. design variables)
     * @param [in,out] aInitialDisp initial displacement vector
     * @param [in,out] aInitialVel initial velocity vector
     *
    ******************************************************************************/
    void computeInitialAccelerations(const Plato::ScalarVector & aControl)
    {
        const Plato::OrdinalType tINITIAL_TIME_STEP_INDEX = 0;
        auto tInitialVel = Kokkos::subview(mVelocities, tINITIAL_TIME_STEP_INDEX, Kokkos::ALL());
        auto tInitialAcc = Kokkos::subview(mAccelerations, tINITIAL_TIME_STEP_INDEX, Kokkos::ALL());
        auto tInitialDisp = Kokkos::subview(mDisplacements, tINITIAL_TIME_STEP_INDEX, Kokkos::ALL());
        Kokkos::deep_copy(tInitialAcc, static_cast<Plato::Scalar>(0));
        auto tTimeStep = mTimeSteps[tINITIAL_TIME_STEP_INDEX];
        mEquality->setPreviousState(tTimeStep, tInitialDisp, tInitialVel, tInitialAcc);
        mResidual = mEquality->value(tTimeStep, tInitialDisp, tInitialVel, tInitialAcc, aControl);
        mJacobian = mEquality->gradient_u(tTimeStep, tInitialDisp, tInitialVel, tInitialAcc, aControl);
        this->solve(*mJacobian, mResidual, tInitialAcc);
    }

    void setPreviousBackwardState(const Plato::OrdinalType & aBackwardTimeIndex,
                                  const Plato::ScalarMultiVector & aStateU,
                                  const Plato::ScalarMultiVector & aStateV,
                                  const Plato::ScalarMultiVector & aStateA)
    {
        auto tPreviousTimeIndex = aBackwardTimeIndex - static_cast<Plato::OrdinalType>(1);
        if(tPreviousTimeIndex >= static_cast<Plato::OrdinalType>(0))
        {
            auto tPreviousTimeStep = mTimeSteps[tPreviousTimeIndex];
            auto tPreviousAcc = Kokkos::subview(aStateA, tPreviousTimeIndex, Kokkos::ALL());
            auto tPreviousVel = Kokkos::subview(aStateV, tPreviousTimeIndex, Kokkos::ALL());
            auto tPreviousDisp = Kokkos::subview(aStateU, tPreviousTimeIndex, Kokkos::ALL());
            mEquality->setPreviousState(tPreviousTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);
        }
        else
        {
            Plato::Scalar tPreviousTimeStep = 0.;
            Plato::ScalarVector tPreviousAcc("Previous Acc",mNumStates);
            Plato::ScalarVector tPreviousVel("Previous Vel",mNumStates);
            Plato::ScalarVector tPreviousDisp("Previous Disp",mNumStates);
            mEquality->setPreviousState(tPreviousTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);
        }
    }

    void updateAdjointState(std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> aCriterion,
                            const Plato::OrdinalType & aBackwardTimeIndex,
                            const Plato::ScalarVector & aControl,
                            const Plato::ScalarMultiVector & aStateU,
                            const Plato::ScalarMultiVector & aStateV,
                            const Plato::ScalarMultiVector & aStateA,
                            const Plato::ScalarVector & aOldAdjDisp,
                            const Plato::ScalarVector & aOldAdjVel,
                            const Plato::ScalarVector & aOldAdjAcc,
                            Plato::ScalarVector & aNewAdjDisp,
                            Plato::ScalarVector & aNewAdjVel,
                            Plato::ScalarVector & aNewAdjAcc)
    {
        // Get primal state associated with current backward time step
        auto tBackwardAcc = Kokkos::subview(aStateA, aBackwardTimeIndex, Kokkos::ALL());
        auto tBackwardVel = Kokkos::subview(aStateV, aBackwardTimeIndex, Kokkos::ALL());
        auto tBackwardDisp = Kokkos::subview(aStateU, aBackwardTimeIndex, Kokkos::ALL());

        // Update adjoint displacement and velocity vectors
        auto tBackwardTimeStep = mTimeSteps[aBackwardTimeIndex];
        auto tPrevBackwardTimeIndex = aBackwardTimeIndex + static_cast<Plato::OrdinalType>(1);
        auto tPrevBackwardTimeStep = aBackwardTimeIndex != mNumTimeSteps ? mTimeSteps[tPrevBackwardTimeIndex] : static_cast<Plato::Scalar>(0);
        auto tDfDu = aCriterion->gradient_u(tBackwardTimeStep, tBackwardDisp, tBackwardVel, tBackwardAcc, aControl);
        auto tOldElasticForce = mElasticForce->value(aOldAdjAcc, aControl, tPrevBackwardTimeStep);
        Plato::Experimental::adjoint_displacement_update(aOldAdjDisp, tOldElasticForce, tDfDu, aNewAdjDisp);
        auto tOldViscousForce = mViscousForce->value(aOldAdjAcc, aControl, tPrevBackwardTimeStep);
        auto tDfDv = aCriterion->gradient_v(tBackwardTimeStep, tBackwardDisp, tBackwardVel, tBackwardAcc, aControl);
        Plato::Experimental::adjoint_velocity_update(tPrevBackwardTimeStep, mAlpha, aOldAdjDisp, aOldAdjVel, tOldElasticForce, tOldViscousForce, tDfDv, aNewAdjVel);

        // compute adjoint accelerations
        Kokkos::deep_copy(aNewAdjAcc, static_cast<Plato::Scalar>(0));
        mAdjointEquation->setPreviousState(tPrevBackwardTimeStep, aOldAdjDisp, aOldAdjVel, aOldAdjAcc);
        mResidual = mAdjointEquation->value(tBackwardTimeStep, aNewAdjDisp, aNewAdjVel, aNewAdjAcc, aControl);
        auto tDfDa = aCriterion->gradient_a(tBackwardTimeStep, tBackwardDisp, tBackwardVel, tBackwardAcc, aControl);
        Plato::axpy(static_cast<Plato::Scalar>(1.0), tDfDa, mResidual);
        mJacobian = mAdjointEquation->gradient_a(tBackwardTimeStep, aNewAdjDisp, aNewAdjVel, aNewAdjAcc, aControl);
        this->applyConstraints(mJacobian, mResidual);
        this->solve(*mJacobian, mResidual, aNewAdjAcc);
    }

    void computeCriterionGradientZ(std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> aCriterion,
                                   const Plato::ScalarVector & aControl,
                                   const Plato::ScalarMultiVector & aStateU,
                                   const Plato::ScalarMultiVector & aStateV,
                                   const Plato::ScalarMultiVector & aStateA,
                                   Plato::ScalarVector & tGradient)
    {
        Plato::ScalarVector tOldAdjVel("OldAdjVel",mNumStates);
        Plato::ScalarVector tNewAdjVel("NewAdjVel",mNumStates);
        Plato::ScalarVector tOldAdjAcc("OldAdjAcc", mNumStates);
        Plato::ScalarVector tNewAdjAcc("NewAdjAcc", mNumStates);
        Plato::ScalarVector tOldAdjDisp("OldAdjDisp",mNumStates);
        Plato::ScalarVector tNewAdjDisp("NewAdjDisp",mNumStates);

        for(auto tIterator = mTimeSteps.begin(); tIterator != mTimeSteps.end(); ++tIterator)
        {
            // update adjoint state
            Plato::OrdinalType tBackwardTimeIndex = std::distance(std::end(mTimeSteps), tIterator) - 1;
            this->updateAdjointState(aCriterion, tBackwardTimeIndex, aControl, aStateU, aStateV, aStateA, tOldAdjDisp, tOldAdjVel, tOldAdjAcc, tNewAdjDisp, tNewAdjVel, tNewAdjAcc);
            // add gradient contribution for this time step
            this->addCriterionGradientZ(aCriterion, tBackwardTimeIndex, aControl, aStateU, aStateV, aStateA, tNewAdjAcc, tGradient);
            // store current adjoint state
            Kokkos::deep_copy(tOldAdjAcc, tNewAdjAcc);
            Kokkos::deep_copy(tOldAdjVel, tNewAdjVel);
            Kokkos::deep_copy(tOldAdjDisp, tNewAdjDisp);
        }
    }

    void addCriterionGradientZ(std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> aCriterion,
                               const Plato::OrdinalType & aBackwardTimeIndex,
                               const Plato::ScalarVector & aControl,
                               const Plato::ScalarMultiVector & aStateU,
                               const Plato::ScalarMultiVector & aStateV,
                               const Plato::ScalarMultiVector & aStateA,
                               const Plato::ScalarVector & aAdjoint,
                               Plato::ScalarVector & aGradient)
    {
        // add residual contribution
        this->setPreviousBackwardState(aBackwardTimeIndex, aStateU, aStateV, aStateA);

        auto tCurrentVel = Kokkos::subview(aStateV, aBackwardTimeIndex, Kokkos::ALL());
        auto tCurrentAcc = Kokkos::subview(aStateA, aBackwardTimeIndex, Kokkos::ALL());
        auto tCurrentDisp = Kokkos::subview(aStateU, aBackwardTimeIndex, Kokkos::ALL());
        auto tCurrentTimeStep = mTimeSteps[aBackwardTimeIndex];
        auto tDhDz = mEquality->gradient_z(tCurrentTimeStep, tCurrentDisp, tCurrentVel, tCurrentAcc, aControl);
        Plato::MatrixTimesVectorPlusVector(tDhDz, aAdjoint, aGradient);

        // add objective contribution
        auto tDfDz = aCriterion->gradient_z(tCurrentTimeStep, tCurrentDisp, tCurrentVel, tCurrentAcc, aControl);
        Plato::axpy(static_cast<Plato::Scalar>(1), tDfDz, aGradient);
    }

    void computeCriterionGradientX(std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> aCriterion,
                                   const Plato::ScalarVector & aControl,
                                   const Plato::ScalarMultiVector & aStateU,
                                   const Plato::ScalarMultiVector & aStateV,
                                   const Plato::ScalarMultiVector & aStateA,
                                   Plato::ScalarVector & tGradient)
    {
        Plato::ScalarVector tOldAdjVel("OldAdjVel",mNumStates);
        Plato::ScalarVector tNewAdjVel("NewAdjVel",mNumStates);
        Plato::ScalarVector tOldAdjAcc("OldAdjAcc", mNumStates);
        Plato::ScalarVector tNewAdjAcc("NewAdjAcc", mNumStates);
        Plato::ScalarVector tOldAdjDisp("OldAdjDisp",mNumStates);
        Plato::ScalarVector tNewAdjDisp("NewAdjDisp",mNumStates);

        for(auto tIterator = mTimeSteps.begin(); tIterator != mTimeSteps.end(); ++tIterator)
        {
            // update adjoint state
            Plato::OrdinalType tBackwardTimeIndex = std::distance(std::end(mTimeSteps), tIterator) - 1;
            this->updateAdjointState(aCriterion, tBackwardTimeIndex, aControl, aStateU, aStateV, aStateA, tOldAdjDisp, tOldAdjVel, tOldAdjAcc, tNewAdjDisp, tNewAdjVel, tNewAdjAcc);
            // add gradient contribution for this time step
            this->addCriterionGradientX(aCriterion, tBackwardTimeIndex, aControl, aStateU, aStateV, aStateA, tNewAdjAcc, tGradient);
            // store current adjoint state
            Kokkos::deep_copy(tOldAdjAcc, tNewAdjAcc);
            Kokkos::deep_copy(tOldAdjVel, tNewAdjVel);
            Kokkos::deep_copy(tOldAdjDisp, tNewAdjDisp);
        }
    }

    void addCriterionGradientX(std::shared_ptr<DynamicScalarFunction<SimplexPhysics>> aCriterion,
                               const Plato::OrdinalType & aBackwardTimeIndex,
                               const Plato::ScalarVector & aControl,
                               const Plato::ScalarMultiVector & aStateU,
                               const Plato::ScalarMultiVector & aStateV,
                               const Plato::ScalarMultiVector & aStateA,
                               const Plato::ScalarVector & aAdjoint,
                               Plato::ScalarVector & aGradient)
    {
        // add residual contribution
        this->setPreviousBackwardState(aBackwardTimeIndex, aStateU, aStateV, aStateA);

        auto tCurrentAcc = Kokkos::subview(aStateA, aBackwardTimeIndex, Kokkos::ALL());
        auto tCurrentVel = Kokkos::subview(aStateV, aBackwardTimeIndex, Kokkos::ALL());
        auto tCurrentDisp = Kokkos::subview(aStateU, aBackwardTimeIndex, Kokkos::ALL());
        auto tCurrentTimeStep = mTimeSteps[aBackwardTimeIndex];
        auto tDhDz = mEquality->gradient_x(tCurrentTimeStep, tCurrentDisp, tCurrentVel, tCurrentAcc, aControl);
        Plato::MatrixTimesVectorPlusVector(tDhDz, aAdjoint, aGradient);

        // add objective contribution
        auto tDfDz = aCriterion->gradient_x(tCurrentTimeStep, tCurrentDisp, tCurrentVel, tCurrentAcc, aControl);
        Plato::axpy(static_cast<Plato::Scalar>(1), tDfDz, aGradient);
    }

public:
    /*************************************************************************//**
     *
     * \brief Constructor
     *
     * @param aMesh       Omega_h mesh data base
     * @param aMeshSets   Omega_h mesh side sets data base
     * @param aParamList  Teuchos parameter list with user define problem input information
     *
    ******************************************************************************/
    ElastodynamicsProblem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList & aParamList) :
        mBeta(0.3025),
        mGamma(0.6),
        mAlpha(-0.1),
        mNumStates(aMesh.nverts() * mNumDofsPerNode),
        mNumConfig(aMesh.nverts() * mSpatialDim),
        mNumControls(aMesh.nverts()),
        mNumTimeSteps(10),
        mNumIterationsAmgX(1000),
        mResidual("Residual", mNumStates),
        mExternalForce("ExternalForce", mNumStates),
        mJacobian(Teuchos::null),
        mTimeSteps(),
        mEquality(nullptr),
        mObjective(nullptr),
        mConstraint(nullptr),
        mAdjointEquation(nullptr)
    {
        this->initialize(aMesh, aMeshSets, aParamList);
    }

    /******************************************************************************//**
     *
     * @brief Constructor
     * @param aMesh mesh data base
     * @param aEquality equality constraint vector function
     *
    **********************************************************************************/
    ElastodynamicsProblem(Omega_h::Mesh& aMesh, const std::shared_ptr<Plato::Experimental::DynamicsVectorFunction<SimplexPhysics>> & aEquality) :
        mBeta(0.3025),
        mGamma(0.6),
        mAlpha(-0.1),
        mNumStates(aMesh.nverts() * mNumDofsPerNode),
        mNumConfig(aMesh.nverts() * mSpatialDim),
        mNumControls(aMesh.nverts()),
        mNumTimeSteps(10),
        mNumIterationsAmgX(1000),
        mResidual("Residual", mNumStates),
        mExternalForce("ExternalForce", mNumStates),
        mJacobian(Teuchos::null),
        mTimeSteps(),
        mEquality(aEquality),
        mObjective(nullptr),
        mConstraint(nullptr),
        mAdjointEquation(nullptr)    
    {
    }

    /*************************************************************************//**
     *
     * \brief Destructor
     *
    *****************************************************************************/
    virtual ~ElastodynamicsProblem()
    {
    }

    /*************************************************************************//**
     *
     * @brief Set maximum number of AmgX solver iterations.
     * @ param [in] aInput number of iterations 
     *
    *****************************************************************************/
    void setMaxNumIterationsAmgX(const Plato::OrdinalType& aInput)
    {
        mNumIterationsAmgX = aInput;   
    }

    /******************************************************************************//**
     *
     * @brief Set time steps and allocate state containers
     *
     * @param[in] aInput angular frequencies
     *
    **********************************************************************************/
    void setTimeSteps(const Plato::OrdinalType& aInput)
    {
        assert(aInput > static_cast<Plato::OrdinalType>(0));

        mNumTimeSteps = aInput;
        Plato::Scalar tTimeStep = static_cast<Plato::Scalar>(1/mNumTimeSteps);
        mTimeSteps.clear();
        mTimeSteps.resize(mNumTimeSteps + 1);
        std::fill(mTimeSteps.begin() + 1, mTimeSteps.end(), tTimeStep);

        auto tNumTimeInstances = mTimeSteps.size();
        mVelocities = Plato::ScalarMultiVector("Velocities", tNumTimeInstances, mNumStates);
        mDisplacements = Plato::ScalarMultiVector("Displacements", tNumTimeInstances, mNumStates);
        mAccelerations = Plato::ScalarMultiVector("Accelerations", tNumTimeInstances, mNumStates);
    }

    /******************************************************************************//**
     *
     * @brief Set essential boundary conditions
     *
     * @param[in] aBcDofs degrees of freedom associated with essential boundary conditions
     * @param[in] aBcValues values associated with essential boundary conditions
     *
    **********************************************************************************/
    void setEssentialBoundaryConditions(const Plato::LocalOrdinalVector & aBcDofs,
                                        const Plato::ScalarVector & aBcValues)
    {
        assert(aBcDofs.size() > 0);
        assert(aBcValues.size() > 0);
        Kokkos::resize(mBcDofs, aBcDofs.size());
        Kokkos::deep_copy(mBcDofs, aBcDofs);
        Kokkos::resize(mBcValues, aBcValues.size());
        Kokkos::deep_copy(mBcValues, aBcValues);
    }

    /******************************************************************************//**
     * 
     * @brief Set external force vector
     *
     * @param[in] aInput external force vector
     *
    **********************************************************************************/
    void setExternalForce(const Plato::ScalarVector & aInput)
    {
        assert(static_cast<Plato::OrdinalType>(aInput.size()) == mNumStates);
        assert(static_cast<Plato::OrdinalType>(mExternalForce.size()) == mNumStates);
        Kokkos::deep_copy(mExternalForce, aInput);
    }

    /***********************************************************************//**
     *
     * \brief Set state multi-vector
     *
     * @param aState multi-vector with state solutions at each time step
     *
    *****************************************************************************/
    void setState(const Plato::ScalarMultiVector & aStateU,
                  const Plato::ScalarMultiVector & aStateV,
                  const Plato::ScalarMultiVector & aStateA)
    {
        assert(aStateV.size() == mVelocities.size());
        assert(aStateU.size() == mDisplacements.size());
        assert(aStateA.size() == mAccelerations.size());
        Kokkos::deep_copy(mVelocities, aStateV);
        Kokkos::deep_copy(mDisplacements, aStateU);
        Kokkos::deep_copy(mAccelerations, aStateA);
    }

    /***********************************************************************//**
     *
     * \brief Returns state multi-vector with state solutions at each time step
     *
    *****************************************************************************/
    void getState(Plato::ScalarMultiVector & aStateU,
                  Plato::ScalarMultiVector & aStateV,
                  Plato::ScalarMultiVector & aStateA)
    {
        assert(aStateV.size() == mVelocities.size());
        assert(aStateU.size() == mDisplacements.size());
        assert(aStateA.size() == mAccelerations.size());
        Kokkos::deep_copy(aStateV, mVelocities);
        Kokkos::deep_copy(aStateU, mDisplacements);
        Kokkos::deep_copy(aStateA, mAccelerations);
    }

    /***********************************************************************//**
     *
     * \brief Applies Dirichlet boundary conditions
     *
     * Modify left-hand-side (LHS) matrix and right-hand (RHS) vector due to the
     * enforcement of the Dirichlet boundary conditions.
     *
     * @param aMatrix   LHS matrix (i.e. Jacobian of the residual equation)
     * @param aVector   RHS vector (i.e. force/load vector)
     *
    *****************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mSpatialDim>(aMatrix, aVector, mBcDofs, mBcValues);
        }
        else
        {
            Plato::applyConstraints<mSpatialDim>(aMatrix, aVector, mBcDofs, mBcValues);
        }
    }

    /***********************************************************************//**
     *
     * \brief Solve partial differential equation using the Newmark-/f$\alpha/f$ method.
     *
     * @param [in] aControl  current set of controls (i.e. design variables)
     *
    *****************************************************************************/
    void solve(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_equality_constraint_defined(mEquality);
        assert(aControl.size() == mNumControls);

        this->computeInitialAccelerations(aControl);
        for(Plato::OrdinalType tTimeIndex = 1; tTimeIndex < mTimeSteps.size(); tTimeIndex++)
        {
            auto tPreviousTimeIndex = tTimeIndex - static_cast<Plato::OrdinalType>(1);
            auto tOldVel = Kokkos::subview(mVelocities, tPreviousTimeIndex, Kokkos::ALL());
            auto tOldAcc = Kokkos::subview(mAccelerations, tPreviousTimeIndex, Kokkos::ALL());
            auto tOldDisp = Kokkos::subview(mDisplacements, tPreviousTimeIndex, Kokkos::ALL());
            auto tPreviousTimeStep = mTimeSteps[tPreviousTimeIndex];
            mEquality->setPreviousState(tPreviousTimeStep, tOldDisp, tOldVel, tOldAcc);

            auto tNewAcc = Kokkos::subview(mAccelerations, tTimeIndex, Kokkos::ALL());
            Kokkos::deep_copy(tNewAcc, static_cast<Plato::Scalar>(0));
            auto tNewVel = Kokkos::subview(mVelocities, tTimeIndex, Kokkos::ALL());
            Kokkos::deep_copy(tNewVel, static_cast<Plato::Scalar>(0));
            auto tNewDisp = Kokkos::subview(mDisplacements, tTimeIndex, Kokkos::ALL());
            Kokkos::deep_copy(tNewDisp, static_cast<Plato::Scalar>(0));

            auto tCurrentTimeStep = mTimeSteps[tTimeIndex];
            mResidual = mEquality->value(tCurrentTimeStep, tNewDisp, tNewVel, tNewAcc, aControl);
            mJacobian = mEquality->gradient_a(tCurrentTimeStep, tNewDisp, tNewVel, tNewAcc, aControl);
            this->applyConstraints(mJacobian, mResidual);
            this->solve(*mJacobian, mResidual, tNewAcc);
            Plato::Experimental::newmark_update(tCurrentTimeStep, mBeta, mGamma, tOldAcc, tOldVel, tOldDisp, tNewAcc, tNewVel, tNewDisp);
        }
    }

    /***********************************************************************//**
     *
     * \brief Evaluate objective function, /f$f(z)/f$
     *
     * Evaluate objective function. The objective function only depends on the
     * controls (i.e. design variables). Thus, it does not depend on the state
     * variables.
     *
     * @param [in] aControl current set of controls /f$z/f$ (i.e. design variables)
     *
    *****************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_objective_func_defined(mObjective);
        assert(aControl.size() == mNumControls);
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tMyVel = Kokkos::subview(mVelocities, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyAcc = Kokkos::subview(mAccelerations, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyDisp = Kokkos::subview(mDisplacements, tTIME_STEP_INDEX, Kokkos::ALL());
        const Plato::Scalar tTIME_STEP = 0;
        Plato::Scalar tValue = mObjective->value(tTIME_STEP, tMyDisp, tMyVel, tMyAcc, aControl);
        return tValue;
    }

    /***********************************************************************//**
     *
     * \brief Evaluate inequality constraint, /f$g(z)/f$
     *
     * Evaluate inequality constraint. The inequality constraint only depends on
     * the controls (i.e. design variables). Thus, it does not depend on the state
     * variables.
     *
     * @param [in] aControl current set of controls /f$z/f$ (i.e. design variables)
     *
    *****************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_inequality_constraint_defined(mConstraint);
        assert(aControl.size() == mNumControls);
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tMyVel = Kokkos::subview(mVelocities, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyAcc = Kokkos::subview(mAccelerations, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyDisp = Kokkos::subview(mDisplacements, tTIME_STEP_INDEX, Kokkos::ALL());
        const Plato::Scalar tTIME_STEP = 0;
        Plato::Scalar tValue = mConstraint->value(tTIME_STEP, tMyDisp, tMyVel, tMyAcc, aControl);
        return tValue;
    }

    /***********************************************************************//**
     *
     * \brief Evaluate objective function, /f$f(z,u)/f$
     *
     * Evaluate objective function. The objective function depends on the control
     * variables /f$f(z/f$ (i.e. design variables) and the state variables /f$u/f$.
     *
     * @param [in] aControl current set of control variables /f$z/f$
     * @param [in] aState current set of state variables /f$u/f$
     *
    *****************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl,
                                 const Plato::ScalarMultiVector & aStateU,
                                 const Plato::ScalarMultiVector & aStateV,
                                 const Plato::ScalarMultiVector & aStateA)
    {
        Plato::Experimental::is_objective_func_defined(mObjective);
        assert(aControl.size() == mNumControls);
        assert(aStateU.size() == mDisplacements.size());
        assert(aStateV.size() == mVelocities.size());
        assert(aStateA.size() == mAccelerations.size());

        Plato::Scalar tValue = 0.;
        for(Plato::OrdinalType tTimeIndex = 0; tTimeIndex < mTimeSteps.size(); tTimeIndex++)
        {
            auto tMyAcc = Kokkos::subview(aStateA, tTimeIndex, Kokkos::ALL());
            auto tMyVel = Kokkos::subview(aStateV, tTimeIndex, Kokkos::ALL());
            auto tMyDisp = Kokkos::subview(aStateU, tTimeIndex, Kokkos::ALL());
            auto tTimeStep = mTimeSteps[tTimeIndex];
            tValue += mObjective->value(tTimeStep, tMyDisp, tMyVel, tMyAcc, aControl);
        }

        return tValue;
    }

    /***********************************************************************//**
     *
     * \brief Evaluate inequality constraint, /f$g(z,u)/f$
     *
     * Evaluate inequality constraint. The inequality constraint depends on the
     * control /f$f(z/f$ (i.e. design variables) and state variables /f$u/f$.
     *
     * @param [in] aControl current set of control variables /f$z/f$
     * @param [in] aState current set of state variables /f$u/f$
     *
    *****************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl,
                                  const Plato::ScalarMultiVector & aStateU,
                                  const Plato::ScalarMultiVector & aStateV,
                                  const Plato::ScalarMultiVector & aStateA)
    {
        Plato::Experimental::is_inequality_constraint_defined(mConstraint);
        assert(aControl.size() == mNumControls);
        assert(aStateU.size() == mDisplacements.size());
        assert(aStateV.size() == mVelocities.size());
        assert(aStateA.size() == mAccelerations.size());

        Plato::Scalar tValue = 0.;
        for(Plato::OrdinalType tTimeIndex = 0; tTimeIndex < mTimeSteps.size(); tTimeIndex++)
        {
            auto tMyAcc = Kokkos::subview(aStateA, tTimeIndex, Kokkos::ALL());
            auto tMyVel = Kokkos::subview(aStateV, tTimeIndex, Kokkos::ALL());
            auto tMyDisp = Kokkos::subview(aStateU, tTimeIndex, Kokkos::ALL());
            auto tTimeStep = mTimeSteps[tTimeIndex];
            tValue += mConstraint->value(tTimeStep, tMyDisp, tMyVel, tMyAcc, aControl);
        }

        return tValue;
    }

    /***********************************************************************//**
     *
     * \brief Compute objective function gradient, /f$\frac{df(z)}{dz}/f$
     *
     * Compute objective function gradient. The objective function only depends
     * on the control variables /f$z/f$ (i.e. design variables). Thus, it
     * does not depend on the state variables.
     *
     * @param [in] aControl current set of control variables (i.e. design variables)
     *
    *****************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_objective_func_defined(mObjective);
        assert(aControl.size() == mNumControls);
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tMyVel = Kokkos::subview(mVelocities, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyAcc = Kokkos::subview(mAccelerations, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyDisp = Kokkos::subview(mDisplacements, tTIME_STEP_INDEX, Kokkos::ALL());
        const Plato::Scalar tTIME_STEP = 0;
        return mObjective->gradient_z(tTIME_STEP, tMyDisp, tMyVel, tMyAcc, aControl);
    }

    /***********************************************************************//**
     *
     * \brief Compute objective function gradient, /f$\frac{f(z,u)}{dz}/f$
     *
     * Compute objective function gradient. The objective function depends on
     * the control /f$f(z/f$ (i.e. design variables) and state variables /f$u/f$.
     *
     * @param [in] aControl current set of control variables /f$z/f$
     * @param [in] aState current set of state variables /f$u/f$
     *
    *****************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl,
                                          const Plato::ScalarMultiVector & aStateU,
                                          const Plato::ScalarMultiVector & aStateV,
                                          const Plato::ScalarMultiVector & aStateA)
    {
        Plato::Experimental::is_objective_func_defined(mObjective);
        Plato::Experimental::is_equality_constraint_defined(mEquality);
        Plato::Experimental::is_adjoint_residual_defined(mAdjointEquation);
        assert(aControl.size() == mNumControls);
        assert(aStateU.size() == mDisplacements.size());
        assert(aStateV.size() == mVelocities.size());
        assert(aStateA.size() == mAccelerations.size());

        Plato::ScalarVector tGradient("Gradient Control", mNumControls);
        this->computeCriterionGradientZ(mObjective, aControl, aStateU, aStateV, aStateA, tGradient);

        return tGradient;
    }

    /***********************************************************************//**
     *
     * \brief Compute objective function gradient, /f$\frac{df(z)}{dx}/f$
     *
     * Compute objective function gradient with respect to configuration /f$x/f$.
     * The objective function only depends on the control variables /f$z/f$
     * (i.e. design variables). Thus, it does not depend on the state variables.
     *
     * @param aControl current set of control variables (i.e. design variables)
     *
    *****************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_objective_func_defined(mObjective);
        assert(aControl.size() == mNumControls);
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tMyVel = Kokkos::subview(mVelocities, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyAcc = Kokkos::subview(mAccelerations, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyDisp = Kokkos::subview(mDisplacements, tTIME_STEP_INDEX, Kokkos::ALL());
        const Plato::Scalar tTIME_STEP = 0;
        return mObjective->gradient_x(tTIME_STEP, tMyDisp, tMyVel, tMyAcc, aControl);
    }

    /***********************************************************************//**
     *
     * \brief Compute objective function gradient, /f$\frac{f(z,u)}{dx}/f$
     *
     * Compute objective function gradient with respect to configuration /f$x/f$.
     * The objective function depends on the control /f$f(z/f$ (i.e. design
     * variables) and state variables /f$u/f$.
     *
     * @param aControl current set of control variables /f$z/f$
     * @param aState current set of state variables /f$u/f$
     *
    *****************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl,
                                           const Plato::ScalarMultiVector & aStateU,
                                           const Plato::ScalarMultiVector & aStateV,
                                           const Plato::ScalarMultiVector & aStateA)
    {
        Plato::Experimental::is_objective_func_defined(mObjective);
        Plato::Experimental::is_equality_constraint_defined(mEquality);
        Plato::Experimental::is_adjoint_residual_defined(mAdjointEquation);
        assert(aControl.size() == mNumControls);
        assert(aStateA.size() == mAccelerations.size());

        Plato::ScalarVector tGradient("Gradient Configuration", mNumConfig);
        this->computeCriterionGradientX(mObjective, aControl, aStateU, aStateV, aStateA, tGradient);

        return tGradient;
    }

    /***********************************************************************//**
     *
     * \brief Compute inequality constraint gradient, /f$\frac{dg(z)}{dz}/f$
     *
     * Compute inequality constraint gradient. The inequality constraint only
     * depends on the control variables /f$z/f$ (i.e. design variables). Thus,
     * it does not depend on the state variables.
     *
     * @param aControl current set of control variables (i.e. design variables)
     *
    *****************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_inequality_constraint_defined(mConstraint);
        assert(aControl.size() == mNumControls);
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tMyVel = Kokkos::subview(mVelocities, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyAcc = Kokkos::subview(mAccelerations, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyDisp = Kokkos::subview(mDisplacements, tTIME_STEP_INDEX, Kokkos::ALL());
        const Plato::Scalar tTIME_STEP = 0;
        return mConstraint->gradient_z(tTIME_STEP, tMyDisp, tMyVel, tMyAcc, aControl);
    }

    /***********************************************************************//**
     *
     * \brief Compute inequality constraint gradient, /f$\frac{f(z,u)}{dz}/f$
     *
     * Compute inequality constraint gradient. The inequality constraint depends
     * on the control /f$f(z/f$ (i.e. design variables) and state variables /f$u/f$.
     *
     * @param aControl current set of control variables /f$z/f$
     * @param aState current set of state variables /f$u/f$
     *
    *****************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl,
                                           const Plato::ScalarMultiVector & aStateU,
                                           const Plato::ScalarMultiVector & aStateV,
                                           const Plato::ScalarMultiVector & aStateA)
    {
        Plato::Experimental::is_inequality_constraint_defined(mConstraint);
        Plato::Experimental::is_equality_constraint_defined(mEquality);
        Plato::Experimental::is_adjoint_residual_defined(mAdjointEquation);
        assert(aControl.size() == mNumControls);
        assert(aStateA.size() == mAccelerations.size());

        Plato::ScalarVector tGradient("Gradient Control", mNumControls);
        this->computeCriterionGradientZ(mConstraint, aControl, aStateU, aStateV, aStateA, tGradient);

        return tGradient;
    }

    /***********************************************************************//**
     *
     * \brief Compute inequality constraint gradient, /f$\frac{dg(z)}{dx}/f$
     *
     * Compute inequality constraint gradient with respect to configuration /f$x/f$.
     * The inequality constraint only depends on the control variables /f$z/f$
     * (i.e. design variables). Thus, it does not depend on the state variables.
     *
     * @param aControl current set of control variables (i.e. design variables)
     *
    *****************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    {
        Plato::Experimental::is_inequality_constraint_defined(mConstraint);
        assert(aControl.size() == mNumControls);
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tMyVel = Kokkos::subview(mVelocities, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyAcc = Kokkos::subview(mAccelerations, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tMyDisp = Kokkos::subview(mDisplacements, tTIME_STEP_INDEX, Kokkos::ALL());
        const Plato::Scalar tTIME_STEP = 0;
        return mConstraint->gradient_x(tTIME_STEP, tMyDisp, tMyVel, tMyAcc, aControl);
    }

    /***********************************************************************//**
     *
     * \brief Compute inequality constraint gradient, /f$\frac{g(z,u)}{dx}/f$
     *
     * Compute inequality constraint gradient with respect to configuration /f$x/f$.
     * The inequality constraint depends on the control /f$f(z/f$ (i.e. design
     * variables) and state variables /f$u/f$.
     *
     * @param aControl current set of control variables /f$z/f$
     * @param aState current set of state variables /f$u/f$
     *
    *****************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl,
                                            const Plato::ScalarMultiVector & aStateU,
                                            const Plato::ScalarMultiVector & aStateV,
                                            const Plato::ScalarMultiVector & aStateA)
    {
        Plato::Experimental::is_inequality_constraint_defined(mConstraint);
        Plato::Experimental::is_equality_constraint_defined(mEquality);
        Plato::Experimental::is_adjoint_residual_defined(mAdjointEquation);
        assert(aControl.size() == mNumControls);
        assert(aStateA.size() == mAccelerations.size());

        Plato::ScalarVector tGradient("Gradient Configuration", mNumConfig);
        this->computeCriterionGradientX(mConstraint, aControl, aStateU, aStateV, aStateA, tGradient);

        return tGradient;
    }

};
// end class ElastodynamicsProblem

namespace ElastodynamicsDriver
{

template<const Plato::OrdinalType NumDofsPerNode>
inline void output_data(const Omega_h::Int & aEntityDim,
                        const std::string & aName,
                        const Omega_h::Write<Omega_h::Real> & aData,
                        Omega_h::Mesh & aMesh)
{
    if(aMesh.has_tag(aEntityDim, aName) == false)
    {
        aMesh.add_tag(aEntityDim, aName, NumDofsPerNode, Omega_h::Reals(aData));
    }
    else
    {
        aMesh.set_tag(aEntityDim, aName, Omega_h::Reals(aData));
    }
}

template<const Plato::OrdinalType SpatialDim>
inline void output(Plato::Experimental::AbstractDynamicsProblem & aProblem,
                   Teuchos::ParameterList & aParamList,
                   const std::string & aVizFilePath,
                   Omega_h::Mesh& aMesh)
{
    const Plato::Scalar tRestartTime = 0.;
    const Plato::OrdinalType tCellDim = aMesh.dim();
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aVizFilePath, &aMesh, tCellDim, tRestartTime);

    auto tProblemSpecs = aParamList.sublist("Elastodynamics Problem");
    assert(tProblemSpecs.isParameter("Physics"));

    auto tSublist = tProblemSpecs.sublist("Time Integration");
    auto tTimeStep = tSublist.get<Plato::Scalar>("Time Step");
    auto tNumTimeSteps = tSublist.get<Plato::Scalar>("Number Time Steps");

    auto tNumVertices = aMesh.nverts();
    auto tNumDofs = tNumVertices * SpatialDim;
    auto tRange = tNumTimeSteps + static_cast<Plato::OrdinalType>(1);
    Plato::ScalarMultiVector tAcc("Acc", tRange, tNumDofs);
    Plato::ScalarMultiVector tVel("Vel", tRange, tNumDofs);
    Plato::ScalarMultiVector tDisp("Disp", tRange, tNumDofs);
    aProblem.getState(tDisp, tVel, tAcc);

    Omega_h::Write<Omega_h::Real> tAccOut(tNumDofs, "AccOut");
    Omega_h::Write<Omega_h::Real> tVelOut(tNumDofs, "VelOut");
    Omega_h::Write<Omega_h::Real> tDispOut(tNumDofs, "DispOut");
    const Plato::OrdinalType tINPUT_NUM_DOFS_PER_NODE = SpatialDim;
    const Plato::OrdinalType tOUTPUT_NUM_DOFS_PER_NODE = SpatialDim;
    
    for(Plato::OrdinalType tStepIndex = 0; tStepIndex < tRange; tStepIndex++)
    {
        Plato::OrdinalType tStride = 0;
        std::string tName("Accelerations");
        auto tAccSubView = Kokkos::subview(tAcc, tStepIndex, Kokkos::ALL());
        Plato::copy<tINPUT_NUM_DOFS_PER_NODE, tOUTPUT_NUM_DOFS_PER_NODE>(tStride, tNumVertices, tAccSubView, tAccOut);
        Plato::Experimental::ElastodynamicsDriver::output_data<tOUTPUT_NUM_DOFS_PER_NODE>(Omega_h::VERT, tName, tAccOut, aMesh);

        tName = "Velocities";
        auto tVelSubView = Kokkos::subview(tVel, tStepIndex, Kokkos::ALL());
        Plato::copy<tINPUT_NUM_DOFS_PER_NODE, tOUTPUT_NUM_DOFS_PER_NODE>(tStride, tNumVertices, tVelSubView, tVelOut);
        Plato::Experimental::ElastodynamicsDriver::output_data<tOUTPUT_NUM_DOFS_PER_NODE>(Omega_h::VERT, tName, tVelOut, aMesh);

        tName = "Displacements";
        auto tDispSubView = Kokkos::subview(tDisp, tStepIndex, Kokkos::ALL());
        Plato::copy<tINPUT_NUM_DOFS_PER_NODE, tOUTPUT_NUM_DOFS_PER_NODE>(tStride, tNumVertices, tDispSubView, tDispOut);
        Plato::Experimental::ElastodynamicsDriver::output_data<tOUTPUT_NUM_DOFS_PER_NODE>(Omega_h::VERT, tName, tDispOut, aMesh);

        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        auto tMyTime = static_cast<Plato::Scalar>(tStepIndex) * tTimeStep;
        tWriter.write(tStepIndex, tMyTime, tTags);
    }
}

template<const Plato::OrdinalType SpatialDim>
inline void run(Teuchos::ParameterList& aProblemSpec,
                Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                const std::string & aVizFilePath)
{
    // create mesh based density from host data
    std::vector<Plato::Scalar> tControlHost(aMesh.nverts(), 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tControlHostView(tControlHost.data(), tControlHost.size());
    auto tControl = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tControlHostView);

    // Solve elastodynamics problem
    Plato::Experimental::ElastodynamicsProblem<Plato::Experimental::Elastodynamics<SpatialDim>> tProblem(aMesh, aMeshSets, aProblemSpec);
    tProblem.solve(tControl);

    Plato::Experimental::ElastodynamicsDriver::output<SpatialDim>(tProblem, aProblemSpec, aVizFilePath, aMesh);
}

template<const Plato::OrdinalType SpatialDim>
inline void driver(Omega_h::Library* aLibOSH,
                   Teuchos::ParameterList & aProblemSpec,
                   const std::string& aInputFilename,
                   const std::string& aVizFilePath)
{
    auto& tAssocParamList = aProblemSpec.sublist("Associations");
    auto tMesh = Omega_h::binary::read(aInputFilename, aLibOSH);
    tMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

    Omega_h::Assoc tAssoc;
    Omega_h::update_assoc(&tAssoc, tAssocParamList);
    auto tMeshSets = Omega_h::invert(&tMesh, tAssoc);

    Plato::Experimental::ElastodynamicsDriver::run<SpatialDim>(aProblemSpec, tMesh, tMeshSets, aVizFilePath);
}

inline void driver(Omega_h::Library* aLibOmegaH,
                   Teuchos::ParameterList & aProblemSpec,
                   const std::string& aInputFilename,
                   const std::string& aVizFilePath)
{
    const Plato::OrdinalType tSpaceDim = aProblemSpec.get<Plato::OrdinalType>("Spatial Dimension", 3);

    // Run Plato problem
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        Plato::Experimental::ElastodynamicsDriver::driver<3>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        Plato::Experimental::ElastodynamicsDriver::driver<2>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        Plato::Experimental::ElastodynamicsDriver::driver<1>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
    }
}

} // ElastodynamicsDriver

} // namespace Experimental

} // namespace Plato

namespace PlatoUnitTests
{

/******************************************************************************//**
 *
 * \brief Test set initial condition vector inline function.
 *
 * Test inline function used to set an initial condition vector, e.g. \f$\mathbf{u}_{t=0}\f$.
 *
 * where \f$\mathbf{u}_{t=0}\f$ is the displacement vector at time \f$t=0\f$.
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, SetInitialConditionVector)
{
    // SET TEST DATA
     const Plato::OrdinalType tNumBCs = 3;
     Plato::ScalarVector tValuesBCs("ValuesBCs", tNumBCs);
     auto tHostValuesBCs = Kokkos::create_mirror(tValuesBCs);
     tHostValuesBCs(0) = 0; tHostValuesBCs(1) = 0.5; tHostValuesBCs(2) = 0.8;
     Kokkos::deep_copy(tValuesBCs, tHostValuesBCs);

     Plato::LocalOrdinalVector tDofsBCs("DofsBCs", tNumBCs);
     auto tHostDofsBCs = Kokkos::create_mirror(tDofsBCs);
     tHostDofsBCs(0) = 0; tHostDofsBCs(1) = 5; tHostDofsBCs(2) = 8;
     Kokkos::deep_copy(tDofsBCs, tHostDofsBCs);

     // CALL FUNCTION
     const Plato::OrdinalType tNumCells = 10;
     const Plato::OrdinalType tNumDofs = tNumCells + static_cast<Plato::OrdinalType>(1); // for a 1D example
     Plato::ScalarVector tInitialDisp("InitialDisp", tNumDofs);
     Plato::Experimental::set_initial_conditions_vector(tDofsBCs, tValuesBCs, tInitialDisp);

     // TEST OUTPUT DISPLACEMENTS
     auto tHostInitialDisp = Kokkos::create_mirror(tInitialDisp);
     Kokkos::deep_copy(tHostInitialDisp, tInitialDisp);
     std::vector<Plato::Scalar> tGold(tNumDofs, 0);
     tGold[5] = 0.5; tGold[8] = 0.8;
     TEST_EQUALITY(tHostInitialDisp.size(), static_cast<Plato::OrdinalType>(tGold.size()));

     const Plato::Scalar tTolerance = 1e-6;
     for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
     {
         TEST_FLOATING_EQUALITY(tHostInitialDisp(tDofIndex), tGold[tDofIndex], tTolerance);
     }
}

/******************************************************************************//**
 *
 * \brief Test the partial derivative of the heaviside projection operator.
 *
 * Test the partial derivative of the heaviside projection operator with respect
 * to the controls (i.e. design variables) for a given cell (i.e. element) using
 * automatic differentiation and compare the results against its analytical
 * counterpart. The heaviside projection operator is defined as:
 *
 *  /f$H(\rho) = 1 - \exp(-\beta*\alpha) + \alpha\exp(-\beta)/f$, where /f$\alpha = 1 - \rho/f$
 *
 * and its corresponding partial derivative is given by
 *
 *   /f$\frac{\partial H(\rho)}{\partial\rho} =
 *     -\exp\left( -\beta\left( 1-\rho \right) \right) - \exp\left( -\beta \right),
 *
 * where /f$\rho/f$ is the control (i.e. design variable) and /f$\beta\geq{0}/f$
 * dictates the curvature of the regularization. 
 *
 * Note: the control values are averaged at the cell level and thus the analytical
 * gradient defined above is divided by the number of nodes in the cell inside 
 * the ApplyProjection class.
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, HeavisideProjection_GradZ)
{
    const Plato::OrdinalType tNumNodesPerCell = 2;
    typedef Sacado::Fad::SFad<Plato::Scalar, tNumNodesPerCell> FadType;

    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tOutputVal("OutputVal", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tOutputGrad("OutputGrad", tNumNodesPerCell);
    Plato::ScalarMultiVectorT<FadType> tControl("Control", tNumCells, tNumNodesPerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tControl(aCellOrdinal, 0) = FadType(tNumNodesPerCell, 0, 1.0);
        tControl(aCellOrdinal, 1) = FadType(tNumNodesPerCell, 1, 1.0);
    }, "Set Controls");

    // SET EVALUATION TYPES FOR UNIT TEST
    Plato::HeavisideProjection tProjection;
    Plato::ApplyProjection<Plato::HeavisideProjection> tApplyProjection(tProjection);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        FadType tValue = tApplyProjection(aCellOrdinal, tControl);
        tOutputVal(aCellOrdinal) = tValue.val();
        tOutputGrad(0) = tValue.dx(0);
        tOutputGrad(1) = tValue.dx(1);
    }, "UnitTest: HeavisideProjection_GradZ");

    // TEST OUTPUT
    auto tHostVal = Kokkos::create_mirror(tOutputVal);
    Kokkos::deep_copy(tHostVal, tOutputVal);
    auto tHostGrad = Kokkos::create_mirror(tOutputGrad);
    Kokkos::deep_copy(tHostGrad, tOutputGrad);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldVal = { 0 };
    std::vector<Plato::Scalar> tGoldGrad = { -5.000022699964881, -5.000022699964881 };
    TEST_FLOATING_EQUALITY(tHostVal(0), tGoldVal[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(0), tGoldGrad[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(1), tGoldGrad[1], tTolerance);
}

/******************************************************************************//**
 *
 * \brief Test the partial derivative of the hyperbolic projection operator.
 *
 * Test the partial derivative of the hyperbolic projection operator with respect
 * to the controls (i.e. design variables) for a given cell (i.e. element) using
 * automatic differentiation and compare the results against its analytical counterpart.
 * The hyperbolic projection operator is defined as:
 *
 *  /f$H_{\eta}(\rho) = \frac{\tanh(\beta\eta) + \tanh(\beta(\rho - \eta))}
 *                      {\tanh(\beta\eta) + \tanh(\beta(1 - \eta))}/f$
 *
 * and its partial derivative is given by
 *
 *   /f$\frac{\partial H_{\eta}(\rho)}{\partial\rho} =
 *     \frac{\beta\left( 1 - \tanh^2\left( \beta\left( \rho-\eta \right) \right) \right)}
 *     {\tanh\left( \beta\eta \right) + tanh\left( \beta\left( 1 - \eta \right) \right)},
 *
 * where /f$\eta/f$ is the threshold level on the controls /f$\rho/f$ and /f$\beta\geq{0}/f$
 * dictates the curvature of the regularization.
 *
 * Note: the control values are averaged at the cell level and thus the analytical
 * gradient defined above is divided by the number of nodes in the cell inside 
 * the ApplyProjection class.
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, HyperbolicTangentProjection_GradZ)
{
    const Plato::OrdinalType tNumNodesPerCell = 2;
    typedef Sacado::Fad::SFad<Plato::Scalar, tNumNodesPerCell> FadType;

    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tOutputVal("OutputVal", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tOutputGrad("OutputGrad", tNumNodesPerCell);
    Plato::ScalarMultiVectorT<FadType> tControl("Control", tNumCells, tNumNodesPerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tControl(aCellOrdinal, 0) = FadType(tNumNodesPerCell, 0, 1.0);
        tControl(aCellOrdinal, 1) = FadType(tNumNodesPerCell, 1, 1.0);
    }, "Set Controls");

    // SET EVALUATION TYPES FOR UNIT TEST
    Plato::HyperbolicTangentProjection tProjection;
    Plato::ApplyProjection<Plato::HyperbolicTangentProjection> tApplyProjection(tProjection);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        FadType tValue = tApplyProjection(aCellOrdinal, tControl);
        tOutputVal(aCellOrdinal) = tValue.val();
        tOutputGrad(0) = tValue.dx(0);
        tOutputGrad(1) = tValue.dx(1);
    }, "UnitTest: HyperbolicTangentProjection_GradZ");

    // TEST OUTPUT
    auto tHostVal = Kokkos::create_mirror(tOutputVal);
    Kokkos::deep_copy(tHostVal, tOutputVal);
    auto tHostGrad = Kokkos::create_mirror(tOutputGrad);
    Kokkos::deep_copy(tHostGrad, tOutputGrad);
    
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldVal = { 1.0 };
    std::vector<Plato::Scalar> tGoldGrad = { 4.539992985607449e-4, 4.539992985607449e-4 };
    TEST_FLOATING_EQUALITY(tHostVal(0), tGoldVal[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(0), tGoldGrad[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(1), tGoldGrad[1], tTolerance);
}

/******************************************************************************//**
 *
 * \brief Test element (i.e. cell) damping force calculation for elastodynamics
 *        application.
 *
 * Example is from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, example's formulation can be found on page 387, Example #2.
 *
 *   \f$left(\frac{AE}{L} * \left[ {\begin{array}{cc} 1 & -1 \\ -1 & 1 \\ \end{array}}
 *   \right] - \omega^{2}\frac{m}{6} * \left[ {\begin{array}{cc} 2 & 1 \\ 1 & 2 \\
 *   \end{array} } \right] \left( {\begin{array}{c} u_1 \\ u_2 \end{array} } \right)
 *   \right)\f$,
 *
 *   where \f$A\f$ denotes area, \f$E\f$ is the elastic modulus, \f$L\f$ is the
 *   bar length, \f$\omega\f$ is the angular frequency and \f$m\f$ is mass.
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, RayleighDamping_Elastodynamics)
{
    Plato::Scalar tArea = 1;
    Plato::Scalar tMass = 1;
    Plato::Scalar tLength = 2;
    Plato::Scalar tElasticModulus = 10;
    Plato::Scalar tStiffnessCoeff = (tArea * tElasticModulus) / tLength;
    Plato::Scalar tInertiaCoeff = (tArea * tElasticModulus) / (tMass * tLength);
    Plato::Scalar tAngularFrequency = static_cast<Plato::Scalar>(3.464) * std::sqrt(tInertiaCoeff);
    Plato::Scalar tMassCoeff = tAngularFrequency * tAngularFrequency * (tMass / static_cast<Plato::Scalar>(6));

    // Set internal forces, where the displacement vector (U) is set to U = {1 2}.
    const Plato::OrdinalType tSize = 2;
    std::vector<Plato::Scalar> tData(tSize);
    tData[0] = -1.0 * tStiffnessCoeff; tData[1] = 1.0 * tStiffnessCoeff;

    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarMultiVector tInternalForces("InternalForces", tNumCells, tSize);
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    tHostInternalForces(0, 0) = tData[0]; tHostInternalForces(0, 1) = tData[1];
    Kokkos::deep_copy(tInternalForces, tHostInternalForces);

    // Set inertial forces, where the displacement vector (U) is set to U = {1 2}.
    tData[0] = 4.0 * tMassCoeff; tData[1] = 5.0 * tMassCoeff;

    Plato::ScalarMultiVector tInertialForces("InertialForces", tNumCells, tSize);
    auto tHostInertialForces = Kokkos::create_mirror(tInertialForces);
    tHostInertialForces(0, 0) = tData[0]; tHostInertialForces(0, 1) = tData[1];
    Kokkos::deep_copy(tInertialForces, tHostInertialForces);

    // Compute Rayleigh damping forces from internal and inertial forces
    // note: default mass (0.025) and stiffness (0.023) damping coefficients are used
    const Plato::OrdinalType tSpaceDim = 2;
    Plato::RayleighDamping<tSpaceDim> tComputeDampingForces;
    Plato::ScalarMultiVector tDampingForces("DampingForces", tNumCells, tSize);

    // RUN KERNEL
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal){
        tComputeDampingForces(aCellOrdinal, tInternalForces, tInertialForces, tDampingForces);
    }, "UnitTest::RayleighDamping_Elastodynamics");

    // TEST OUTPUT
    auto tHostDampingForces = Kokkos::create_mirror(tDampingForces);
    Kokkos::deep_copy(tHostDampingForces, tDampingForces);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar>> tGold = { {0.884941333333334, 1.36492666666667} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tElemIndex = 0; tElemIndex < tSize; tElemIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostDampingForces(tCellIndex, tElemIndex), tGold[tCellIndex][tElemIndex], tTolerance);
       }
    }
}

/******************************************************************************//**
 *
 * \brief Test computation of the interpolated cell displacement vector.
 *
 * Test the function used to compute the interpolated cell displacements vector  used
 * to compute the elastodynamics residual when the generalized Newmark \f$\alpha\f$-method
 * is used. The interpolated displacements are given by
 *
 * \f$\hat{\mathbf{u}}_{n} = \mathbf{u}_{n-1} + \Delta{t}_n\dot{\mathbf{u}}_{n-1}
 * + \Delta{t}_n^2\left(\frac{1}{2} - \beta\right)\ddot{\mathbf{u}}_{n-1}\f$,
 *
 * where \f$\Delta{t}_n\f$ is the current time step, \f$\beta\f$ is an algorithmic
 * damping parameter. Finally, \f$\ddot{\mathbf{u}}_{n-1}\f$, \f$\dot{\mathbf{u}}_{n-1}\f$
 * and \f$\mathbf{u}_{n-1}\f$ are the current acceleration, velocity and displacement
 * vectors,
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeInterpolatedDispNewmark)
{
    // SET STATE MULTIVECTORS
     const Plato::OrdinalType tNumCells = 1;
     const Plato::OrdinalType tNumDofsPerCell = 2;
     Plato::ScalarMultiVector tOut("Out", tNumCells, tNumDofsPerCell);
     Plato::ScalarMultiVector tVel("Vel", tNumCells, tNumDofsPerCell);
     auto tHostVel = Kokkos::create_mirror(tVel);
     tHostVel(0,0) = 0.5; tHostVel(0,1) = 0.2;
     Kokkos::deep_copy(tVel, tHostVel);
     Plato::ScalarMultiVector tAcc("Acc", tNumCells, tNumDofsPerCell);
     auto tHostAcc = Kokkos::create_mirror(tAcc);
     tHostAcc(0,0) = -0.1; tHostAcc(0,1) = 0.2;
     Kokkos::deep_copy(tAcc, tHostAcc);
     Plato::ScalarMultiVector tDisp("Disp", tNumCells, tNumDofsPerCell);
     auto tHostDisp = Kokkos::create_mirror(tDisp);
     tHostDisp(0,0) = 0.1; tHostDisp(0,1) = 0.2;
     Kokkos::deep_copy(tDisp, tHostDisp);

     // CONSTANTS
     Plato::Scalar tBeta = 0.3025;
     Plato::Scalar tTimeStep = 2e-1;

     // COMPUTE INTERPOLATED DISPLACEMENTS
     Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
     {
         Plato::Experimental::compute_interpolated_disp<tNumDofsPerCell>(aCellOrdinal, tTimeStep, tBeta, tDisp, tVel, tAcc, tOut);
     }, "UnitTest::compute_interpolated_disp");

     // TEST OUTPUT VELOCITIES
     auto tHostOut = Kokkos::create_mirror(tOut);
     Kokkos::deep_copy(tHostOut, tOut);
     const Plato::Scalar tTolerance = 1e-6;
     std::vector<std::vector<Plato::Scalar>> tGold = { {0.19921, 0.24158} };
     for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
     {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerCell; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostOut(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
     }
}

/******************************************************************************//**
 *
 * \brief Test computation of the interpolated cell velocity vector.
 *
 * Test the function used to compute the interpolated cell velocity vector  used
 * to compute the elastodynamics residual when the generalized Newmark \f$\alpha\f$-method
 * is used. The interpolated velocity are given by
 *
 * \f$\hat{\dot{\mathbf{u}}}_{n} =
 *   \dot{\mathbf{u}}_{n-1} + \Delta{t}_n\left(1-\gamma\right)\ddot{\mathbf{u}}_{n-1}\f$,
 *
 * where \f$\Delta{t}_n\f$ is the current time step, \f$\gamma\f$ is an algorithmic
 * damping parameter. Finally, \f$\ddot{\mathbf{u}}_{n-1}\f$ and \f$\dot{\mathbf{u}}_{n-1}\f$
 * are the current acceleration and velocity vectors,
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeInterpolatedVelNewmark)
{
    // SET STATE MULTIVECTORS
     const Plato::OrdinalType tNumCells = 1;
     const Plato::OrdinalType tNumDofsPerCell = 2;
     Plato::ScalarMultiVector tOut("Out", tNumCells, tNumDofsPerCell);
     Plato::ScalarMultiVector tVel("Vel", tNumCells, tNumDofsPerCell);
     auto tHostVel = Kokkos::create_mirror(tVel);
     tHostVel(0,0) = 0.5; tHostVel(0,1) = 0.2;
     Kokkos::deep_copy(tVel, tHostVel);
     Plato::ScalarMultiVector tAcc("Acc", tNumCells, tNumDofsPerCell);
     auto tHostAcc = Kokkos::create_mirror(tAcc);
     tHostAcc(0,0) = -0.1; tHostAcc(0,1) = 0.2;
     Kokkos::deep_copy(tAcc, tHostAcc);

     // CONSTANTS
     Plato::Scalar tGamma = 0.6;
     Plato::Scalar tTimeStep = 2e-1;

     // COMPUTE INTERPOLATED DISPLACEMENTS
     Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
     {
         Plato::Experimental::compute_interpolated_vel<tNumDofsPerCell>(aCellOrdinal, tTimeStep, tGamma, tVel, tAcc, tOut);
     }, "UnitTest::compute_interpolated_vel");

     // TEST OUTPUT VELOCITIES
     auto tHostOut = Kokkos::create_mirror(tOut);
     Kokkos::deep_copy(tHostOut, tOut);
     const Plato::Scalar tTolerance = 1e-6;
     std::vector<std::vector<Plato::Scalar>> tGold = { {0.492, 0.216} };
     for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
     {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerCell; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostOut(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
     }

}

/******************************************************************************//**
 *
 * \brief Test the Newmark update scheme used for explicit-dynamics applications.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.13-4a) and (11.13-4b) on page 418.
 *
 *   \f$\{\ddot{\mathbf{D}}\}_{n+1} = \frac{\beta}{\Delta{t}^2}\left( \{\mathbf{D}\}_{n+1}
 *   - \{\mathbf{D}\}_{n} -\Delta{t}\{\{\dot{\mathbf{D}}\}\}_n - left(\frac{1}{2\beta}
 *   - 1 \right)\{\ddot{\mathbf{D}}\}_n \right)\f$
 * and
 *   \f$\{\dot{\mathbf{D}}\}_{n+1} = \frac{\gamma}{\beta\Delta{t}}\left( \{\mathbf{D}\}_{n+1}
 *   - \{\mathbf{D}\}_n \right) - \left(\frac{\gamma}{\beta} - 1\right)\{\dot{\mathbf{D}}\}_n
 *   - \Delta{t}\left(\frac{\gamma}{2\beta} - 1\right)\{\ddot{\mathbf{D}}\}_n\f$,
 *
 *   where \f$\gamma\f$ and \f$\beta\f$ are numerical factors used in the Newmark method,
 *   \f$\Delta{t}$\fdenotes is the time step, \f$\{\mathbf{D}\}\f$ denotes displacements,
 *   \f$\{\dot{\mathbf{D}}\}\f$ denotes velocities and \f$\{\ddot{\mathbf{D}}\}\f$ denotes
 *   accelerations.
 *
**********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, NewmarkUpdateScheme)
{
    const Plato::Scalar tGamma = 0.6;
    const Plato::Scalar tBeta = 0.3025;
    const Plato::Scalar tTimeStep = 2e-1;
    const Plato::OrdinalType tNumDofs = 2;

    // Set vector data
    std::vector<Plato::Scalar> tOldVelData = { 0.5, 0.2 };
    Plato::ScalarVector tOldVel("OldVelocities", tNumDofs);
    Plato::Experimental::copy(tOldVelData, tOldVel);

    std::vector<Plato::Scalar> tOldAccelData = { -0.1, 0.2 };
    Plato::ScalarVector tOldAccel("OldAccelerations", tNumDofs);
    Plato::Experimental::copy(tOldAccelData, tOldAccel);

    std::vector<Plato::Scalar> tOldDispData = { 0.1, 0.2 };
    Plato::ScalarVector tOldDisp("OldDisplacements", tNumDofs);
    Plato::Experimental::copy(tOldDispData, tOldDisp);

    std::vector<Plato::Scalar> tNewAccelData = { 0.25, -0.125 };
    Plato::ScalarVector tNewAccel("NewAccelerations", tNumDofs);
    Plato::Experimental::copy(tNewAccelData, tNewAccel);

    Plato::ScalarVector tNewVel("NewVelocities", tNumDofs);
    Plato::ScalarVector tNewDisp("NewDisplacements", tNumDofs);
    Plato::Experimental::newmark_update(tTimeStep, tBeta, tGamma, tOldAccel, tOldVel, tOldDisp, tNewAccel, tNewVel, tNewDisp);

    // TEST DISPLACEMENTS AND VELOCITIES OUTPUT
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tNewDispGold = {0.202235, 0.2400675};
    auto tHostNewDisp = Kokkos::create_mirror(tNewDisp);
    Kokkos::deep_copy(tHostNewDisp, tNewDisp);

    std::vector<Plato::Scalar> tNewVelGold = {0.522, 0.201};
    auto tHostNewVel = Kokkos::create_mirror(tNewVel);
    Kokkos::deep_copy(tHostNewVel, tNewVel);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostNewVel(tIndex), tNewVelGold[tIndex], tTolerance);
        TEST_FLOATING_EQUALITY(tHostNewDisp(tIndex), tNewDispGold[tIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the evaluation of the equation of motion and its residual
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 *   \f$[\mathbf{K}]\{\mathbf{D}\} + [\mathbf{M}]\{\ddot{\mathbf{D}}\}
 *   + [\mathbf{C}]\{\dot{\mathbf{D}}\} - \{\mathbf{F}^{ext}\} = 0\f$,
 *
 * where \f$\{\mathbf{F}^{ext}\}\f$ is the external force vector, \f$[\mathbf{M}]$\f
 * is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$ is the vector of accelerations,
 * \f$[\mathbf{C}]\f$ is the mass matrix, \f$\{\dot{\mathbf{D}}\}\f$ is the vector
 * of velocities, \f$[\mathbf{K}]\f$ is the stiffness matrix and \f$\{\mathbf{D}\}\f$
 * is the vector of displacements.
 *
**********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, InitialAccelerationsCalculation)
{
    const Plato::OrdinalType tSpaceDim = 1;
    using ResidualT = typename Plato::Experimental::DynamicsEvaluation<Plato::Experimental::Elastodynamics<tSpaceDim>>::Residual;

    Plato::DataMap tDataMap;
    Teuchos::ParameterList tProblemParams;
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);

    Plato::OrdinalType tNumCells = 1;
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + 1;
    Plato::ScalarMultiVectorT<ResidualT::ControlScalarType> tControl("Control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0, 0) = 1.0; tHostControl(0, 1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfig("Config", tNumCells, tNumNodesPerCell, tSpaceDim);
    auto tHostConfig = Kokkos::create_mirror(tConfig);
    tHostConfig(0, 0, 0) = 0.0; tHostConfig(0, 1, 0) = 1.0;
    Kokkos::deep_copy(tConfig, tHostConfig);

    Plato::Scalar tInitialTimeStep = 0;
    Plato::ScalarMultiVector tPreviousAcc("PreviousAcc", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tPreviousDisp("PreviousDisp", tNumCells, tNumNodesPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0, 0) = 0.0; tHostPreviousDisp(0, 1) = 1.0;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);
    Plato::ScalarMultiVector tPreviousVel("PreviousVel", tNumCells, tNumNodesPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0, 0) = 0.0; tHostPreviousVel(0, 1) = 0.5;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarMultiVectorT<ResidualT::StateUScalarType> tNewAcc("NewAcc", tNumCells, tNumNodesPerCell);
    auto tHostNewAcc = Kokkos::create_mirror(tNewAcc);
    tHostNewAcc(0, 0) = 0.0; tHostNewAcc(0, 1) = -2.0;
    Kokkos::deep_copy(tNewAcc, tHostNewAcc);

    // BUILD OMEGA_H MESH
    Omega_h::MeshSets tMeshSets;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // SET ELASTODYNAMICS VECTOR FUNCTION
    Plato::Experimental::ElastodynamicsResidual<ResidualT, ::SIMP, Plato::HyperbolicTangentProjection>
        tElastodynamics(*tMesh, tMeshSets, tDataMap, tProblemParams, tPenaltyParams);
    tElastodynamics.setPreviousState(tInitialTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tResidualElastodyn("ResidualElastodyn", tNumCells, tNumNodesPerCell);
    tElastodynamics.evaluateLHS(tInitialTimeStep, tNewAcc, tControl, tConfig, tResidualElastodyn);
    tElastodynamics.evaluateRHS(tInitialTimeStep, tNewAcc, tControl, tConfig, tResidualElastodyn);

    // TEST OUTPUT
    const Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {-1.85851, 0.86476} };
    auto tHostResidualElastodyn = Kokkos::create_mirror(tResidualElastodyn);
    Kokkos::deep_copy(tHostResidualElastodyn, tResidualElastodyn);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostResidualElastodyn(tCellIndex, tNodeIndex), tGold[tCellIndex][tNodeIndex], tTolerance);
       }
    }
}

/******************************************************************************//**
 *
 * \brief Test function that computes optimal Newmark damping parameters.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, the \f$\beta\f$ and \f$\gamma\f$ equations shown in Table 11.13-1,
 * page 420.
 *
 * The following equations for \f$\beta\f$ and \f$\gamma\f$ parameters introduce
 * algorithmic damping and retain unconditional stability,
 *
 *   \f$\beta = \frac{1}{4}\left(1 - \alpha\right)^{2}\f$
 * and
 *   \f$\gamma = \frac{1}{2}\left(1 - 2\alpha\right)\f$,
 *
 * where \f$\alpha\f$ introduces algorithmic damping to damp lower modes and
 * \f$\gamma\f$ and \f$\beta\f$ introduces algorithmic damping to damp higher
 * modes.
 *
 *********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeNewmarkAlphaDampingParameters)
{
    // CALL FUNCTION
    Plato::Scalar tBeta = 0.0;
    Plato::Scalar tGamma = 0.0;
    Plato::Scalar tAlpha = -0.1;
    Plato::Experimental::compute_newmark_damping_coeff(tAlpha, tBeta, tGamma);

    // TEST OUTPUT
    Plato::Scalar tGold = 0.6;
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tGamma, tGold, tTolerance);

    tGold = 0.3025;
    TEST_FLOATING_EQUALITY(tBeta, tGold, tTolerance);
}

/******************************************************************************//**
 *
 * \brief Test elastodynamics residual calculation using the Newmark /f$\alpha/f$-method.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 * \f$(1-\alpha)*\{\mathbf{F}^{ext}\} - \alpha*\{\mathbf{F}^{ext}\} - [\mathbf{M}]
 * \{\ddot{\mathbf{D}}\}_{n+1} - (1-\alpha)*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n+1}
 * + \alpha*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n} - (1-\alpha)*[\mathbf{K}]
 * \{\mathbf{D}\}_{n+1} + \alpha*[\mathbf{K}]\{\mathbf{D}\}_{n} = 0,
 *
 * where \f$n\f$ is the time step index, \f$\alpha\f$ introduces algorithmic damping
 * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\{\mathbf{F}^{ext}\}\f$ is the external
 * force vector, \f$[\mathbf{M}]$\f is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$
 * is the vector of accelerations, \f$[\mathbf{C}]\f$ is the mass matrix,
 * \f$\{\dot{\mathbf{D}}\}\f$ is the vector of velocities, \f$[\mathbf{K}]\f$ is
 * the stiffness matrix and \f$\{\mathbf{D}\}\f$ is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastodynamicsResidual_Residual)
{
    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tSpaceDim = 1;
    using ResidualT = typename Plato::Experimental::DynamicsEvaluation<Plato::Experimental::Elastodynamics<tSpaceDim>>::Residual;
    
    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    Plato::DataMap tDataMap;
    Teuchos::ParameterList tProblemParams;
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    
    Omega_h::MeshSets tMeshSets;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::Experimental::ElastodynamicsResidual<ResidualT, ::SIMP, Plato::HyperbolicTangentProjection>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tPenaltyParams);
    
    // INITIALIZE MULTI-DIMENSION VECTORS
    const Plato::OrdinalType tNumCells = 1;
    const Plato::OrdinalType tNumNodesPerCell = 2;
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVector tOutput("Residual", tNumCells, tNumDofsPerCell);
    
    Plato::ScalarMultiVector tState("State", tNumCells, tNumDofsPerCell);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0, 0) = 0.75; tHostState(0, 1) = 0.25;
    Kokkos::deep_copy(tState, tHostState);
    
    Plato::ScalarMultiVector tControl("Control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0, 0) = 1.0; tHostControl(0, 1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);
    
    Plato::ScalarMultiVector tPreviousVel("PreviousVel", tNumCells, tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0, 0) = 0.5; tHostPreviousVel(0, 1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);
    
    Plato::ScalarMultiVector tPreviousAcc("PreviousAcc", tNumCells, tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0, 0) = -0.1; tHostPreviousAcc(0, 1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);
    
    Plato::ScalarMultiVector tPreviousDisp("PreviousDisp", tNumCells, tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0, 0) = 0.1; tHostPreviousDisp(0, 1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);
    
    Plato::ScalarArray3D tConfig("Config", tNumCells, tNumNodesPerCell, tSpaceDim);
    auto tHostConfig = Kokkos::create_mirror(tConfig);
    tHostConfig(0, 0, 0) = 0.0; tHostConfig(0, 1, 0) = 1.0;
    Kokkos::deep_copy(tConfig, tHostConfig);

    // EVALUATE ELASTODYNAMICS RESIDUAL
    const Plato::Scalar tTimeStep = 2e-1;
    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);
    tResidual.evaluateLHS(tTimeStep, tState, tControl, tConfig, tOutput);
    tResidual.evaluateRHS(tTimeStep, tState, tControl, tConfig, tOutput);

    // TEST OUTPUT RESIDUAL
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar>> tGold = { {0.207922, 0.302268} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
       for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerCell; tDofIndex++)
       {
           TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
       }
    }
}

/******************************************************************************//**
 *
 * \brief Test elastodynamics residual evaluation.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 * \f$(1-\alpha)*\{\mathbf{F}^{ext}\} - \alpha*\{\mathbf{F}^{ext}\} - [\mathbf{M}]
 * \{\ddot{\mathbf{D}}\}_{n+1} - (1-\alpha)*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n+1}
 * + \alpha*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n} - (1-\alpha)*[\mathbf{K}]
 * \{\mathbf{D}\}_{n+1} + \alpha*[\mathbf{K}]\{\mathbf{D}\}_{n} = 0,
 *
 * where \f$n\f$ is the time step index, \f$\alpha\f$ introduces algorithmic damping
 * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\{\mathbf{F}^{ext}\}\f$ is the external
 * force vector, \f$[\mathbf{M}]$\f is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$
 * is the vector of accelerations, \f$[\mathbf{C}]\f$ is the mass matrix,
 * \f$\{\dot{\mathbf{D}}\}\f$ is the vector of velocities, \f$[\mathbf{K}]\f$ is
 * the stiffness matrix and \f$\{\mathbf{D}\}\f$ is the vector of displacements.
 *
**********************************************************************************/ 
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastodynamicsResidual_VectorFunc_Residual)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE RESIDUAL
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.value(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT RESIDUAL
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.207922, 0.302268 };
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerCell; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test elastodynamics Jacobian calculation.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 * \f$(1-\alpha)*\{\mathbf{F}^{ext}\} - \alpha*\{\mathbf{F}^{ext}\} - [\mathbf{M}]
 * \{\ddot{\mathbf{D}}\}_{n+1} - (1-\alpha)*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n+1}
 * + \alpha*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n} - (1-\alpha)*[\mathbf{K}]
 * \{\mathbf{D}\}_{n+1} + \alpha*[\mathbf{K}]\{\mathbf{D}\}_{n} = 0,
 *
 * where \f$n\f$ is the time step index, \f$\alpha\f$ introduces algorithmic damping
 * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\{\mathbf{F}^{ext}\}\f$ is the external
 * force vector, \f$[\mathbf{M}]$\f is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$
 * is the vector of accelerations, \f$[\mathbf{C}]\f$ is the mass matrix,
 * \f$\{\dot{\mathbf{D}}\}\f$ is the vector of velocities, \f$[\mathbf{K}]\f$ is
 * the stiffness matrix and \f$\{\mathbf{D}\}\f$ is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastodynamicsResidual_VectorFuncJac)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE JACOBIAN
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.gradient_a(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT JACOBIAN
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    const Plato::Scalar tTolerance = 1e-6;
    const std::vector<Plato::Scalar> tGold =
        { 0.268678461538462, 0.232671538461538, 0.232671538461538, 0.268678461538462 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tIndex), tGold[tIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test elastodynamics partial derivative wrt controls.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 * \f$(1-\alpha)*\{\mathbf{F}^{ext}\} - \alpha*\{\mathbf{F}^{ext}\} - [\mathbf{M}]
 * \{\ddot{\mathbf{D}}\}_{n+1} - (1-\alpha)*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n+1}
 * + \alpha*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n} - (1-\alpha)*[\mathbf{K}]
 * \{\mathbf{D}\}_{n+1} + \alpha*[\mathbf{K}]\{\mathbf{D}\}_{n} = 0,
 *
 * where \f$n\f$ is the time step index, \f$\alpha\f$ introduces algorithmic damping
 * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\{\mathbf{F}^{ext}\}\f$ is the external
 * force vector, \f$[\mathbf{M}]$\f is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$
 * is the vector of accelerations, \f$[\mathbf{C}]\f$ is the mass matrix,
 * \f$\{\dot{\mathbf{D}}\}\f$ is the vector of velocities, \f$[\mathbf{K}]\f$ is
 * the stiffness matrix and \f$\{\mathbf{D}\}\f$ is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastodynamicsResidual_VectorFuncJacZ)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);
    tResidual.allocateJacobianZ(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE GRADIENT Z
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.gradient_z(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT GRADIENT Z
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.000283189326466042, 0.000411688379932078, 0.000283189326466042, 0.000411688379932078 };
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumEntries; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test elastodynamics partial derivative wrt configuration.
 *
 * Equations are from Cook, R.D., Malkus, D.S., Plesha, M.E. and Witt, R.J., 1974.
 * Concepts and applications of finite element analysis (Vol. 4). New York: Wiley.
 * Specifically, equations (11.2-12) on page 376.
 *
 * \f$(1-\alpha)*\{\mathbf{F}^{ext}\} - \alpha*\{\mathbf{F}^{ext}\} - [\mathbf{M}]
 * \{\ddot{\mathbf{D}}\}_{n+1} - (1-\alpha)*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n+1}
 * + \alpha*[\mathbf{C}]\{\dot{\mathbf{D}}\}_{n} - (1-\alpha)*[\mathbf{K}]
 * \{\mathbf{D}\}_{n+1} + \alpha*[\mathbf{K}]\{\mathbf{D}\}_{n} = 0,
 *
 * where \f$n\f$ is the time step index, \f$\alpha\f$ introduces algorithmic damping
 * (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\{\mathbf{F}^{ext}\}\f$ is the external
 * force vector, \f$[\mathbf{M}]$\f is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$
 * is the vector of accelerations, \f$[\mathbf{C}]\f$ is the mass matrix,
 * \f$\{\dot{\mathbf{D}}\}\f$ is the vector of velocities, \f$[\mathbf{K}]\f$ is
 * the stiffness matrix and \f$\{\mathbf{D}\}\f$ is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastodynamicsResidual_VectorFuncJacX)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);
    tResidual.allocateJacobianX(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE GRADIENT X
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.gradient_x(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT GRADIENT X
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = { -0.302268, 0.302268, -0.207922, 0.207922 };
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumEntries; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the evaluation of the equation of motion.
 *
 * The elastodynamics vector function should produce the equation of motion at time
 * \f$t = 0\f$. This equation is used to compute the initial accelerations for an
 * elastodynamics simulation. The equation of motion is defined in Cook, R.D., Malkus,
 * D.S., Plesha, M.E. and Witt, R.J., 1974. Concepts and applications of finite
 * element analysis (Vol. 4). New York: Wiley. Specifically, equations (11.2-12)
 * on page 376.
 *
 * [\mathbf{M}]\{\ddot{\mathbf{D}}\} + [\mathbf{C}] \{\dot{\mathbf{D}}\}
 * + [\mathbf{K}]\{\mathbf{D}\} - \f$\{\mathbf{F}^{ext}\}= 0,
 *
 * where \f$\{\mathbf{F}^{ext}\}\f$ is the external force vector, \f$[\mathbf{M}]$\f
 * is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$ is the vector of accelerations,
 * \f$[\mathbf{C}]\f$ is the mass matrix, \f$\{\dot{\mathbf{D}}\}\f$ is the vector
 * of velocities, \f$[\mathbf{K}]\f$ is the stiffness matrix and \f$\{\mathbf{D}\}\f$
 * is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, EquationMotion_VectorFuncValue)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tInitialTimeStep = 0;
    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    tResidual.setPreviousState(tInitialTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE THE RESIDUAL
    auto tOutput = tResidual.value(tInitialTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST RESIDUAL VALUES
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    const Plato::OrdinalType tNumEntries = 2;
    TEST_EQUALITY(tNumEntries, tOutput.size());

    const Plato::Scalar tTolerance = 1e-6;
    const std::vector<Plato::Scalar> tGold = { 0.129048076923077, 0.379701923076923 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tIndex), tGold[tIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the Jacobian of the equation of motion using automatic differentiation.
 *
 * The elastodynamics vector function should produce the equation of motion at time
 * \f$t = 0\f$. This equation is used to compute the initial accelerations for an
 * elastodynamics simulation. The equation of motion is defined in Cook, R.D., Malkus,
 * D.S., Plesha, M.E. and Witt, R.J., 1974. Concepts and applications of finite
 * element analysis (Vol. 4). New York: Wiley. Specifically, equations (11.2-12)
 * on page 376.
 *
 * [\mathbf{M}]\{\ddot{\mathbf{D}}\} + [\mathbf{C}] \{\dot{\mathbf{D}}\}
 * + [\mathbf{K}]\{\mathbf{D}\} - \f$\{\mathbf{F}^{ext}\}= 0,
 *
 * where \f$\{\mathbf{F}^{ext}\}\f$ is the external force vector, \f$[\mathbf{M}]$\f
 * is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$ is the vector of accelerations,
 * \f$[\mathbf{C}]\f$ is the mass matrix, \f$\{\dot{\mathbf{D}}\}\f$ is the vector
 * of velocities, \f$[\mathbf{K}]\f$ is the stiffness matrix and \f$\{\mathbf{D}\}\f$
 * is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, EquationMotion_VectorFuncJac)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.15;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tInitialTimeStep = 0;
    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    tResidual.setPreviousState(tInitialTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE JACOBIAN
    auto tOutput = tResidual.gradient_a(tInitialTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT JACOBIAN
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    const Plato::Scalar tTolerance = 1e-6;
    const std::vector<Plato::Scalar> tGold = { 0.25, 0.25, 0.25, 0.25 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tIndex), tGold[tIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the partial derivative of the equation of motion wrt the controls.
 *
 * The elastodynamics vector function should produce the equation of motion at time
 * \f$t = 0\f$. This equation is used to compute the initial accelerations for an
 * elastodynamics simulation. The equation of motion is defined in Cook, R.D., Malkus,
 * D.S., Plesha, M.E. and Witt, R.J., 1974. Concepts and applications of finite
 * element analysis (Vol. 4). New York: Wiley. Specifically, equations (11.2-12)
 * on page 376.
 *
 * [\mathbf{M}]\{\ddot{\mathbf{D}}\} + [\mathbf{C}] \{\dot{\mathbf{D}}\}
 * + [\mathbf{K}]\{\mathbf{D}\} - \f$\{\mathbf{F}^{ext}\}= 0,
 *
 * where \f$\{\mathbf{F}^{ext}\}\f$ is the external force vector, \f$[\mathbf{M}]$\f
 * is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$ is the vector of accelerations,
 * \f$[\mathbf{C}]\f$ is the mass matrix, \f$\{\dot{\mathbf{D}}\}\f$ is the vector
 * of velocities, \f$[\mathbf{K}]\f$ is the stiffness matrix and \f$\{\mathbf{D}\}\f$
 * is the vector of displacements.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, EquationMotion_VectorFuncJacZ)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);
    tResidual.allocateJacobianZ(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tInitialTimeStep = 0.;
    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    tResidual.setPreviousState(tInitialTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS (CURRENT STATE)
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE JACOBIAN
    auto tOutput = tResidual.gradient_z(tInitialTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT JACOBIAN
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    const Plato::Scalar tTolerance = 1e-6;
    const std::vector<Plato::Scalar> tGold = { 0.000175763209211070, 0.000517153220217267, 0.000175763209211070, 0.000517153220217267 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tIndex), tGold[tIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the partial derivative of the equation of motion wrt configuration.
 *
 * The elastodynamics vector function should produce the equation of motion at time
 * \f$t = 0\f$. This equation is used to compute the initial accelerations for an
 * elastodynamics simulation. The equation of motion is defined in Cook, R.D., Malkus,
 * D.S., Plesha, M.E. and Witt, R.J., 1974. Concepts and applications of finite
 * element analysis (Vol. 4). New York: Wiley. Specifically, equations (11.2-12)
 * on page 376.
 *
 * [\mathbf{M}]\{\ddot{\mathbf{D}}\} + [\mathbf{C}] \{\dot{\mathbf{D}}\}
 * + [\mathbf{K}]\{\mathbf{D}\} - \f$\{\mathbf{F}^{ext}\}= 0,
 *
 * where \f$\{\mathbf{F}^{ext}\}\f$ is the external force vector, \f$[\mathbf{M}]$\f
 * is the mass matrix, \f$\{\ddot{\mathbf{D}}\}\f$ is the vector of accelerations,
 * \f$[\mathbf{C}]\f$ is the mass matrix, \f$\{\dot{\mathbf{D}}\}\f$ is the vector
 * of velocities, \f$[\mathbf{K}]\f$ is the stiffness matrix and \f$\{\mathbf{D}\}\f$
 * is the vector of displacements.
 *
**********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, EquationMotion_VectorFuncJacX)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    tName = "Elastodynamics";
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set(tName, tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);
    tResidual.allocateJacobianX(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET PREVIOUS STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.15;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    const Plato::Scalar tInitialTimeStep = 0.;
    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    tResidual.setPreviousState(tInitialTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.5; tHostStateV(1) = 0.5;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = 0.7; tHostStateD(1) = 0.2;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE JACOBIAN
    auto tOutput = tResidual.gradient_x(tInitialTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT JACOBIAN
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    const Plato::Scalar tTolerance = 1e-4;
    const std::vector<Plato::Scalar> tGold = { -0.312394, 0.312394, -0.196356, 0.196356 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tIndex), tGold[tIndex], tTolerance);
    }     
}

/******************************************************************************//**
 *
 * \brief Test the adjoint elastodynamics residual.
 *
 * Unit test the adjoint elastodynamics residual. This residual is used/needed
 * to compute the gradient of a criterion with respect to the control variables.
 * The adjoint elastodynamics residual is given by
 *
 * \f$\mathbf{R} = \bigg[\mathbf{M} + \Delta{t}_{N-n}\gamma\bar{\alpha}
 * \mathbf{C} + \Delta{t}^2_{N-n}\bar{\alpha}\beta\mathbf{K}\bigg]\Lambda_{n}
 * = -\left(\frac{\partial{f_{N-n}}}{\partial\mathbf{a}_{N-n}}\right)^{T}
 * - \Delta{t}^2_{N-n}\beta\Phi_{n} - \Delta{t}_{N-n}\gamma\ \Theta_{n}
 * - H(n)\bigg[\Delta{t}^2_{N-n+1}\left(\frac{1}{2} - \beta\right)\Big[
 * \bar{\alpha}\mathbf{K}\Lambda_{n-1} + \Phi_{n-1}\Big] + \Delta{t}_{N-n+1}
 * \left(1-\gamma\right)\Big[\bar{\alpha}\mathbf{C}\Lambda_{n-1} + \Theta_{n-1}
 * \Big] \bigg]\f$,
 *
 * where \f$n=0,\dots,N-1\f$ and \f$N\f$ is the number if time steps, \f$\alpha\f$ is
 * algorithmic damping (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\bar{\alpha} =
 * 1+\alpha\f$, \f$\Delta{t}\f$ is the time step, \f$\gamma\f$ and \f$\beta\f$
 * are numerical factors used to damp higher modes in the solution, \f$\mathbf{M}$\f,
 * \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping and stiffness matrices.
 * Finally, \f$\Lambda\f$, \f$\Theta\f$ and \f$\Phi\f$ are the adjoint acceleration,
 * velocities and displacement vectors.
 *
**********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointElastodynamics_VectorFunc_Residual)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    Teuchos::ParameterList tProblemParams;
    tProblemParams.set("Elastodynamics", tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    tName = "Adjoint Elastodynamics";
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET CURRENT STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    // SET PREVIOUS STATE
    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.7; tHostStateV(1) = 0.2;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = -0.5; tHostStateD(1) = -0.25;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE RESIDUAL
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.value(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT RESIDUAL
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.356794509615385, 0.272117990384615 };
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerCell; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the Jacobian of the adjoint elastodynamics residual.
 *
 * Unit test the Jacobian of the adjoint elastodynamics residual. The Jacobian is
 * used/needed to compute Lagrange multipliers by solving the adjoint problem.
 * The Jacobian is given by
 *
 * \f$\mathbf{Jac} = \mathbf{M} + \Delta{t}_{N-n}\gamma\bar{\alpha}\mathbf{C}
 * + \Delta{t}^2_{N-n}\bar{\alpha}\beta\mathbf{K}\bigg,
 *
 * where \f$n=0,\dots,N-1\f$ and \f$N\f$ is the number if time steps, \f$\alpha\f$
 * is algorithmic damping (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\bar{\alpha} =
 * 1+\alpha\f$, \f$\Delta{t}\f$ is the time step, \f$\gamma\f$ and \f$\beta\f$
 * are numerical factors used to damp higher modes in the solution, \f$\mathbf{M}$\f,
 * \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping and stiffness matrices.
 *
**********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointElastodynamics_VectorFuncJac)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    Teuchos::ParameterList tProblemParams;
    tProblemParams.set("Elastodynamics", tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    tName = "Adjoint Elastodynamics";
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET CURRENT STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    // SET PREVIOUS STATE
    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.7; tHostStateV(1) = 0.2;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = -0.5; tHostStateD(1) = -0.25;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE RESIDUAL
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.gradient_a(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT JACOBIAN
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    // TEST OUTPUT RESIDUAL
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.268678461538462, 0.232671538461538, 0.232671538461538, 0.268678461538462 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tIndex), tGold[tIndex], tTolerance);
    }
}

/******************************************************************************//**
 *
 * \brief Test the partial derivative of the elastodynamics adjoint residual wrt controls.
 *
 * Unit test the partial derivative of the elastodynamics adjoint residual wrt
 * controls. The adjoint elastodynamics residual is given by
 *
 * \f$\mathbf{R} = \bigg[\mathbf{M} + \Delta{t}_{N-n}\gamma\bar{\alpha}
 * \mathbf{C} + \Delta{t}^2_{N-n}\bar{\alpha}\beta\mathbf{K}\bigg]\Lambda_{n}
 * = -\left(\frac{\partial{f_{N-n}}}{\partial\mathbf{a}_{N-n}}\right)^{T}
 * - \Delta{t}^2_{N-n}\beta\Phi_{n} - \Delta{t}_{N-n}\gamma\ \Theta_{n}
 * - H(n)\bigg[\Delta{t}^2_{N-n+1}\left(\frac{1}{2} - \beta\right)\Big[
 * \bar{\alpha}\mathbf{K}\Lambda_{n-1} + \Phi_{n-1}\Big] + \Delta{t}_{N-n+1}
 * \left(1-\gamma\right)\Big[\bar{\alpha}\mathbf{C}\Lambda_{n-1} + \Theta_{n-1}
 * \Big] \bigg]\f$,
 *
 * where \f$n=0,\dots,N-1\f$ and \f$N\f$ is the number if time steps, \f$\alpha\f$ is
 * algorithmic damping (\f$\frac{-1}{3}\leq\alpha\leq{0}\f$), \f$\bar{\alpha} =
 * 1+\alpha\f$, \f$\Delta{t}\f$ is the time step, \f$\gamma\f$ and \f$\beta\f$
 * are numerical factors used to damp higher modes in the solution, \f$\mathbf{M}$\f,
 * \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping and stiffness matrices.
 * Finally, \f$\Lambda\f$, \f$\Theta\f$ and \f$\Phi\f$ are the adjoint acceleration,
 * velocities and displacement vectors.
 *
**********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointElastodynamics_VectorFuncGradZ)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // INITIALIZE ELASTODYNAMICS CONSTRUCTOR
    std::string tName("Penalty Function");
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set(tName, tPenaltyParams);

    Teuchos::ParameterList tProblemParams;
    tProblemParams.set("Elastodynamics", tElastodynamicsSubList);

    // INITIALIZE DYNAMICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    tName = "Adjoint Elastodynamics";
    Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>
        tResidual(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);
    tResidual.allocateJacobianZ(*tMesh, tMeshSets, tDataMap, tProblemParams, tName);

    // SET CURRENT STATE
    const Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tPreviousVel("PreviousVel", tNumDofsPerCell);
    auto tHostPreviousVel = Kokkos::create_mirror(tPreviousVel);
    tHostPreviousVel(0) = 0.5; tHostPreviousVel(1) = 0.2;
    Kokkos::deep_copy(tPreviousVel, tHostPreviousVel);

    Plato::ScalarVector tPreviousAcc("PreviousAcc", tNumDofsPerCell);
    auto tHostPreviousAcc = Kokkos::create_mirror(tPreviousAcc);
    tHostPreviousAcc(0) = -0.1; tHostPreviousAcc(1) = 0.2;
    Kokkos::deep_copy(tPreviousAcc, tHostPreviousAcc);

    Plato::ScalarVector tPreviousDisp("PreviousDisp", tNumDofsPerCell);
    auto tHostPreviousDisp = Kokkos::create_mirror(tPreviousDisp);
    tHostPreviousDisp(0) = 0.1; tHostPreviousDisp(1) = 0.2;
    Kokkos::deep_copy(tPreviousDisp, tHostPreviousDisp);

    // SET PREVIOUS STATE
    const Plato::Scalar tPrevTimeStep = 1e-1;
    tResidual.setPreviousState(tPrevTimeStep, tPreviousDisp, tPreviousVel, tPreviousAcc);

    // SET INPUT ARGUMENTS
    Plato::ScalarVector tStateA("StateA", tNumDofsPerCell);
    auto tHostStateA = Kokkos::create_mirror(tStateA);
    tHostStateA(0) = 0.75; tHostStateA(1) = 0.25;
    Kokkos::deep_copy(tStateA, tHostStateA);

    Plato::ScalarVector tStateV("StateV", tNumDofsPerCell);
    auto tHostStateV = Kokkos::create_mirror(tStateV);
    tHostStateV(0) = 0.7; tHostStateV(1) = 0.2;
    Kokkos::deep_copy(tStateV, tHostStateV);

    Plato::ScalarVector tStateD("StateD", tNumDofsPerCell);
    auto tHostStateD = Kokkos::create_mirror(tStateD);
    tHostStateD(0) = -0.5; tHostStateD(1) = -0.25;
    Kokkos::deep_copy(tStateD, tHostStateD);

    Plato::ScalarVector tControl("Control", tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1.0; tHostControl(1) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // EVALUATE RESIDUAL
    const Plato::Scalar tTimeStep = 2e-1;
    auto tOutput = tResidual.gradient_z(tTimeStep, tStateD, tStateV, tStateA, tControl);

    // TEST OUTPUT JACOBIAN
    auto tEntries = tOutput->entries();
    auto tHostEntries = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tHostEntries, tEntries);
    const Plato::OrdinalType tNumEntries = 4;
    TEST_EQUALITY(tNumEntries, tEntries.size());

    // TEST OUTPUT RESIDUAL
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.000352276682820657, 0.000330622252084938, 0.000352276682820657, 0.000330622252084938 };
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntries; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostEntries(tIndex), tGold[tIndex], tTolerance);
    }
}

/****************************************************************************//**
 *
 * \brief Test linear elastic force function
 *
 * Test linear elastic force function, which is defined as \f$\mathbf{K}\mathbf{u}\f$,
 * where \f$\mathbf{K}\f$ is the stiffness matrix and \f$mathbf{u}\f$ is the
 * displacement vector.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, LinearElasticForce_Evaluate)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    const Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    const Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // SET MATERIAL PROPERTIES
    Teuchos::ParameterList tIsotropicMaterialSubList;
    tIsotropicMaterialSubList.set<double>("Poissons Ratio", 0.3);
    tIsotropicMaterialSubList.set<double>("Youngs Modulus", 1.0);
    Teuchos::ParameterList tMaterialModelParamList;
    tMaterialModelParamList.set("Isotropic Linear Elastic", tIsotropicMaterialSubList);

    // SET PENALTY MODEL PROPERTIES
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set("Penalty Function", tPenaltyParams);

    // SET PROBLEM PARAMETER LIST
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set("Material Model", tMaterialModelParamList);
    tProblemParams.set("Elastodynamics", tElastodynamicsSubList);

    // ALLOCATE LINEAR ELASTIC FORCE FUNCTOR
    std::string tName("Linear Elastic Force");
    Plato::Experimental::ForceFunction<Plato::Experimental::Elastodynamics<tSpaceDim>> tForceFunction(*tMesh, tProblemParams, tName);

    // ALLOCATE DATA CONTAINERS FOR TEST
    const Plato::OrdinalType tNumDofs = tNumCells * tNumNodesPerCell;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.25; tHostState(1) = -0.125;
    Kokkos::deep_copy(tState, tHostState);

    Plato::ScalarVector tControl("Control", tNumDofs);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1; tHostControl(1) = 1;
    Kokkos::deep_copy(tControl, tHostControl);

    // TEST VALUE FUNCTION
    auto tElasticForce = tForceFunction.value(tState, tControl);

    // TEST OUTPUT RESIDUAL
    const Plato::Scalar tTolerance = 1e-6;
    auto tHostElasticForce = Kokkos::create_mirror(tElasticForce);
    Kokkos::deep_copy(tHostElasticForce, tElasticForce);
    std::vector<Plato::Scalar> tGold = { 0.504807692307692, -0.504807692307692 };
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostElasticForce(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/****************************************************************************//**
 *
 * \brief Test linear elastic force function
 *
 * Test Rayleigh viscous force function, which is given by \f$\mathbf{f}_d = \left(
 * \alpha\mathbf{M} + \beta\mathbf{K}\right)\dot{\mathbf{u}}\f$, where \f$\mathbf{f}_d\f$
 * is the damping force vector, \f$\alpha\f$ is the mass proportional damping
 * coefficient, \f$\beta\f$ is the stiffness proportional damping coefficient
 * \f$\mathbf{K}\f$ is the stiffness matrix, \f$\mathbf{M}\f$ is the mass matrix
 * and \f$\dot{\mathbf{u}}\f$ is the velocity vector.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, RayleighViscousForce_Evaluate)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    const Plato::OrdinalType tNumCells = 1;
    TEST_EQUALITY(tNumCells, tMesh->nelems());
    const Plato::OrdinalType tNumNodesPerCell = 2;
    TEST_EQUALITY(tNumNodesPerCell, tMesh->nverts());

    // SET MATERIAL PROPERTIES
    Teuchos::ParameterList tMaterialModelParamList;
    tMaterialModelParamList.set<double>("Density", 1);
    tMaterialModelParamList.set<double>("Mass Proportional Damping", 0.025);
    tMaterialModelParamList.set<double>("Stiffness Proportional Damping", 0.023);
    Teuchos::ParameterList tIsotropicMaterialSubList;
    tIsotropicMaterialSubList.set<double>("Poissons Ratio", 0.3);
    tIsotropicMaterialSubList.set<double>("Youngs Modulus", 2.0);
    tMaterialModelParamList.set("Isotropic Linear Elastic", tIsotropicMaterialSubList);

    // SET PENALTY MODEL PROPERTIES
    Teuchos::ParameterList tPenaltyParams;
    tPenaltyParams.set<std::string>("Type", "SIMP");
    tPenaltyParams.set<double>("Exponent", 3.0);
    tPenaltyParams.set<double>("Minimum Value", 0.0);
    Teuchos::ParameterList tElastodynamicsSubList;
    tElastodynamicsSubList.set("Penalty Function", tPenaltyParams);

    // SET PROBLEM PARAMETER LIST
    Teuchos::ParameterList tProblemParams;
    tProblemParams.set("Material Model", tMaterialModelParamList);
    tProblemParams.set("Elastodynamics", tElastodynamicsSubList);

    // ALLOCATE LINEAR ELASTIC FORCE FUNCTOR
    std::string tName("Rayleigh Viscous Force");
    Plato::Experimental::ForceFunction<Plato::Experimental::Elastodynamics<tSpaceDim>> tForceFunction(*tMesh, tProblemParams, tName);

    // ALLOCATE DATA CONTAINERS FOR TEST
    const Plato::OrdinalType tNumDofs = tNumCells * tNumNodesPerCell;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.25; tHostState(1) = -0.125;
    Kokkos::deep_copy(tState, tHostState);

    Plato::ScalarVector tControl("Control", tNumDofs);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0) = 1; tHostControl(1) = 1;
    Kokkos::deep_copy(tControl, tHostControl);

    // TEST VALUE FUNCTION
    auto tElasticForce = tForceFunction.value(tState, tControl);

    // TEST OUTPUT RESIDUAL
    const Plato::Scalar tTolerance = 1e-6;
    auto tHostElasticForce = Kokkos::create_mirror(tElasticForce);
    Kokkos::deep_copy(tHostElasticForce, tElasticForce);
    std::vector<Plato::Scalar> tGold = { 0.0240024038461538, -0.0224399038461538 };
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostElasticForce(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/****************************************************************************//**
 *
 * \brief Test adjoint displacement update function
 *
 * Test adjoint displacement vector update, i.e. \f$\Phi_n\f$ for \f$n = 0,
 * \dots,N-1\f$, where \f$N\f$ is the total number of time steps, is given by
 *
 * \f$\Phi_n = \left(\frac{\partial{f}_{N-n}}{\partial\mathbf{u}_{N-n}}\right)^{T}
 * + H(n)\bigg[ \mathbf{K}\Lambda_{n-1} + \Phi_{n-1} \bigg]\f$.
 *
 * \f$f_{N-n}\equiv{f}(\mathbf{u}_{N-n},\mathbf{v}_{N-n},\mathbf{a}_{N-n},
 * \mathbf{z})\f$ is a criterion of interest, \f$H(n)\f$ is the Heaviside step
 * function, \f$\mathbf{K}\f$ is the stiffness matrix, \f$\mathbf{u}\f$ is the
 * displacement vector. Finally, \f$Phi\f$ and \f$\Lambda\f$ are the adjoint
 * displacement and acceleration vectors.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointDisplacementNewmarkUpdate)
{
    // DEFINE CONTAINER SIZE 
    const Plato::OrdinalType tNumCells = 1;
    const Plato::OrdinalType tNumNodesPerCell = 2;
    const Plato::OrdinalType tNumDofs = tNumCells * tNumNodesPerCell;
                 
    // ALLOCATE DATA CONTAINERS FOR TEST
    Plato::ScalarVector tOldDisp("OldDisp", tNumDofs);
    auto tHostOldDisp = Kokkos::create_mirror(tOldDisp);
    tHostOldDisp(0) = 0.25; tHostOldDisp(1) = -0.125;
    Kokkos::deep_copy(tOldDisp, tHostOldDisp);
    
    Plato::ScalarVector tOldForce("OldForce", tNumDofs);
    auto tHostOldForce = Kokkos::create_mirror(tOldForce);
    tHostOldForce(0) = -0.1; tHostOldForce(1) = 0.15;
    Kokkos::deep_copy(tOldForce, tHostOldForce);
    
    Plato::ScalarVector tDfDu("DfDu", tNumDofs);
    auto tHostDfDu = Kokkos::create_mirror(tDfDu);
    tHostDfDu(0) = 0.5; tHostDfDu(1) = 0.125;
    Kokkos::deep_copy(tDfDu, tHostDfDu);
    
    // CALL FUNCTION
    Plato::ScalarVector tNewDisp("NewDisp", tNumDofs);
    Plato::Experimental::adjoint_displacement_update(tOldDisp, tOldForce, tDfDu, tNewDisp);
    
    // TEST OUTPUT
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.65, 0.15 };
    auto tHostNewDisp = Kokkos::create_mirror(tNewDisp);
    Kokkos::deep_copy(tHostNewDisp, tNewDisp);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostNewDisp(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/****************************************************************************//**
 *
 * \brief Test adjoint velocity update function
 *
 * Test adjoint velocity vector update, i.e. \f$\Theta_n\f$ for \f$n=0,\dots,N-1\f$,
 * where \f$N\f$ is the total number of time steps, is given by
 *
 * \f$ \Theta_n = \left(\frac{\partial{f}_{N-n}}{\partial\mathbf{v}_{N-n}}\right)^{T}
 * + H(n)\bigg[ \Delta{t}_{N-n+1}\left(\bar{\alpha}\mathbf{K}\Lambda_{n-1}+\Phi_{n-1}
 * \right) \bigg] + \mathbf{C}\Lambda_{n-1}+\Theta_{n-1} \f$.
 *
 * \f$f_{N-n}\equiv{f}(\mathbf{u}_{N-n},\mathbf{v}_{N-n},\mathbf{a}_{N-n},\mathbf{z})\f$
 * is a criterion of interest, \f$\Delta{t}\f$ is the time step, \f$\bar{\alpha}=\left(
 * 1 + \alpha\right)\f$ is an algorithmic damping parameter, \f$H(n)\f$ is the Heaviside
 * step function, \f$\mathbf{K}\f$ and \f$\mathbf{C}\f$ are the stiffness and damping
 * matrices and \f$\mathbf{v}\f$ is the velocity vector. Finally, \f$\Phi\f$ and
 * \f$\Lambda\f$ are the adjoint displacement and acceleration vectors.
 *
*********************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AdjointVelocityNewmarkUpdate)
{
    // DEFINE CONTAINER SIZE
    const Plato::OrdinalType tNumCells = 1;
    const Plato::OrdinalType tNumNodesPerCell = 2;
    const Plato::OrdinalType tNumDofs = tNumCells * tNumNodesPerCell;

    // ALLOCATE DATA CONTAINERS FOR TEST
    Plato::ScalarVector tOldDisp("OldDisp", tNumDofs);
    auto tHostOldDisp = Kokkos::create_mirror(tOldDisp);
    tHostOldDisp(0) = 0.25; tHostOldDisp(1) = -0.125;
    Kokkos::deep_copy(tOldDisp, tHostOldDisp);

    Plato::ScalarVector tOldVel("OldVel", tNumDofs);
    auto tHostOldVel = Kokkos::create_mirror(tOldVel);
    tHostOldVel(0) = 0.045; tHostOldVel(1) = -0.075;
    Kokkos::deep_copy(tOldVel, tHostOldVel);

    Plato::ScalarVector tOldElasticForce("OldElasticForce", tNumDofs);
    auto tHostOldElasticForce = Kokkos::create_mirror(tOldElasticForce);
    tHostOldElasticForce(0) = -0.1; tHostOldElasticForce(1) = 0.15;
    Kokkos::deep_copy(tOldElasticForce, tHostOldElasticForce);

    Plato::ScalarVector tOldViscousForce("OldViscousForce", tNumDofs);
    auto tHostOldViscousForce = Kokkos::create_mirror(tOldViscousForce);
    tHostOldViscousForce(0) = -0.01; tHostOldViscousForce(1) = 0.015;
    Kokkos::deep_copy(tOldViscousForce, tHostOldViscousForce);

    Plato::ScalarVector tDfDv("DfDv", tNumDofs);
    auto tHostDfDv = Kokkos::create_mirror(tDfDv);
    tHostDfDv(0) = 0.5; tHostDfDv(1) = 0.125;
    Kokkos::deep_copy(tDfDv, tHostDfDv);

    // CALL FUNCTION
    const Plato::Scalar tAlpha = -1e-1;
    const Plato::Scalar tTimeStep = 5e-1;
    Plato::ScalarVector tNewVel("NewVel", tNumDofs);
    Plato::Experimental::adjoint_velocity_update(tTimeStep, tAlpha, tOldDisp, tOldVel, tOldElasticForce, tOldViscousForce, tDfDv, tNewVel);

    // TEST OUTPUT
    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = { 0.615, 0.07 };
    auto tHostNewVel = Kokkos::create_mirror(tNewVel);
    Kokkos::deep_copy(tHostNewVel, tNewVel);
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostNewVel(tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/*
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ElastodynamicsSolve_1D)
{
    // CREATE 2D-MESH
    Omega_h::LO aNx = 10;
    Omega_h::Real aX = 1;
    std::shared_ptr<Omega_h::Mesh> tMesh = PlatoUtestHelpers::build_1d_box_mesh(aX, aNx);

    // PROBLEM INPUTS
    const Plato::Scalar tDensity = 1000;
    const Plato::Scalar tPoissonRatio = 0.3;
    const Plato::Scalar tYoungsModulus = 1e9;
    const Plato::Scalar tMassPropDamping = 0.000025;
    const Plato::Scalar tStiffPropDamping = 0.000023;

    // ALLOCATE ELASTODYNAMICS RESIDUAL
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    const Plato::OrdinalType tSpaceDim = 1;
    using ResidualT = typename Plato::Experimental::DynamicsEvaluation<Plato::Experimental::Elastodynamics<tSpaceDim>>::Residual;
    using GradientA = typename Plato::Experimental::DynamicsEvaluation<Plato::Experimental::Elastodynamics<tSpaceDim>>::GradientA;
    std::shared_ptr<Plato::Experimental::ElastodynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>> tResidual;
    tResidual = std::make_shared<Plato::Experimental::ElastodynamicsResidual<ResidualT, SIMP, Plato::HyperbolicTangentProjection>>
            (*tMesh, tMeshSets, tDataMap);
    tResidual->setMaterialDensity(tDensity);
    tResidual->setMassPropDamping(tMassPropDamping);
    tResidual->setStiffPropDamping(tStiffPropDamping);
    tResidual->setIsotropicLinearElasticMaterial(tYoungsModulus, tPoissonRatio);

    std::shared_ptr<Plato::Experimental::ElastodynamicsResidual<GradientA, SIMP, Plato::HyperbolicTangentProjection>> tJacobianA;
    tJacobianA = std::make_shared<Plato::Experimental::ElastodynamicsResidual<GradientA, SIMP, Plato::HyperbolicTangentProjection>>
            (*tMesh, tMeshSets, tDataMap);
    tResidual->setMaterialDensity(tDensity);
    tJacobianA->setMassPropDamping(tMassPropDamping);
    tJacobianA->setStiffPropDamping(tStiffPropDamping);
    tJacobianA->setIsotropicLinearElasticMaterial(tYoungsModulus, tPoissonRatio);

    // ALLOCATE VECTOR FUNCTION
    std::shared_ptr<Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>> tVectorFunction =
        std::make_shared<Plato::Experimental::DynamicsVectorFunction<Plato::Experimental::Elastodynamics<tSpaceDim>>>(*tMesh, tDataMap);
    tVectorFunction->allocateResidual(tResidual, tJacobianA);

    // ALLOCATE STRUCTURAL DYNAMICS PROBLEM
    Plato::Experimental::ElastodynamicsProblem<Plato::Experimental::Elastodynamics<tSpaceDim>> tProblem(*tMesh, tVectorFunction);

    // SET DIRICHLET BOUNDARY CONDITIONS
    Plato::Scalar tValue = 0;
    auto tNumDofsPerNode = 2*tSpaceDim;
    Omega_h::LOs tCoordsX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    auto tNumDirichletDofs = tNumDofsPerNode*tCoordsX0.size();
    Plato::ScalarVector tDirichletValues("DirichletValues", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("DirichletDofs", tNumDirichletDofs);
    PlatoUtestHelpers::set_dirichlet_boundary_conditions(tNumDofsPerNode, tValue, tCoordsX0, tDirichletDofs, tDirichletValues);
    tProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // SET TIME STEPS
    Plato::Scalar tNumTimeSteps = 20;
    tProblem.setTimeSteps(tNumTimeSteps);

    // SET EXTERNAL FORCE
    auto tNumDofs = tVectorFunction->size();
    auto tNumDofsGold = tNumDofsPerNode * tMesh->nverts();
    TEST_EQUALITY(tNumDofsGold, tNumDofs);
    Plato::ScalarVector tPointLoad("PointLoad", tNumDofs);

    Plato::ScalarMultiVector tValues("Values", 2, tSpaceDim);
    auto tHostValues = Kokkos::create_mirror(tValues);
    tHostValues(0,0) = 0;    tHostValues(1,0) = 0;
    tHostValues(0,1) = -1e5; tHostValues(1,1) = -1e5;
    Kokkos::deep_copy(tValues, tHostValues);

    auto tTopOrdinalIndex = 0;
    auto tNodeOrdinalsX1 = PlatoUtestHelpers::get_2D_boundary_nodes_x1(*tMesh);
    PlatoUtestHelpers::set_point_load(tTopOrdinalIndex, tNodeOrdinalsX1, tValues, tPointLoad);
    tProblem.setExternalForce(tPointLoad);

    //SOLVE ELASTODYNAMICS PROBLEM
    auto tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Kokkos::deep_copy(tControl, static_cast<Plato::Scalar>(1));
    tProblem.setMaxNumIterationsAmgX(500);
    tProblem.solve(tControl);
}
*/

}
