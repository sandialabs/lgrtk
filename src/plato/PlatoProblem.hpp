#ifndef PLATO_PROBLEM_HPP
#define PLATO_PROBLEM_HPP

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"

#include "plato/VectorFunction.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PlatoAbstractProblem.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include "plato/AnalyzeMacros.hpp"

#ifdef HAVE_AMGX
#include "plato/alg/AmgXSparseLinearProblem.hpp"
#endif

namespace Plato
{

/******************************************************************************//**
 * @brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename SimplexPhysics>
class Problem: public Plato::AbstractProblem
{
private:

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::m_numSpatialDims; /*!< spatial dimensions */

    // required
    Plato::VectorFunction<SimplexPhysics> mEqualityConstraint; /*!< equality constraint interface */

    // optional
    std::shared_ptr<const Plato::ScalarFunctionBase> mConstraint; /*!< constraint constraint interface */
    std::shared_ptr<const Plato::ScalarFunctionBase> mObjective; /*!< objective constraint interface */

    Plato::ScalarMultiVector mAdjoint;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    bool mIsSelfAdjoint; /*!< indicates if problem is self-adjoint */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

public:
    /******************************************************************************//**
     * @brief PLATO problem constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    Problem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams) :
            mEqualityConstraint(aMesh, aMeshSets, mDataMap, aInputParams, aInputParams.get<std::string>("PDE Constraint")),
            mConstraint(nullptr),
            mObjective(nullptr),
            mResidual("MyResidual", mEqualityConstraint.size()),
            mStates("States", static_cast<Plato::OrdinalType>(1), mEqualityConstraint.size()),
            mJacobian(Teuchos::null),
            mIsSelfAdjoint(aInputParams.get<bool>("Self-Adjoint", false))
    {
        this->initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Set state variables
     * @param [in] aState 2D view of state variables
    **********************************************************************************/
    void setState(const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));
        Kokkos::deep_copy(mStates, aState);
    }

    /******************************************************************************//**
     * @brief Return 2D view of state variables
     * @return aState 2D view of state variables
    **********************************************************************************/
    Plato::ScalarMultiVector getState()
    {
        return mStates;
    }

    /******************************************************************************//**
     * @brief Return 2D view of adjoint variables
     * @return 2D view of adjoint variables
    **********************************************************************************/
    Plato::ScalarMultiVector getAdjoint()
    {
        return mAdjoint;
    }

    /******************************************************************************//**
     * @brief Apply Dirichlet constraints
     * @param [in] aMatrix Compressed Row Storage (CRS) matrix
     * @param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::m_numDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::m_numDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
    }

    void applyBoundaryLoads(const Plato::ScalarVector & aForce){}

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        mObjective->updateProblem(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Solve system of equations
     * @param [in] aControl 1D view of control variables
     * @return 2D view of state variables
    **********************************************************************************/
    Plato::ScalarMultiVector solution(const Plato::ScalarVector & aControl)
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        Plato::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mResidual = mEqualityConstraint.value(tStatesSubView, aControl);

        mJacobian = mEqualityConstraint.gradient_u(tStatesSubView, aControl);
        this->applyConstraints(mJacobian, mResidual);

#ifdef HAVE_AMGX
        using AmgXLinearProblem = Plato::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode>;
        auto tConfigString = AmgXLinearProblem::getConfigString();
        auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, tStatesSubView, mResidual, tConfigString));
        tSolver->solve();
        tSolver = Teuchos::null;
#endif

        mResidual = mEqualityConstraint.value(tStatesSubView, aControl);

        return mStates;
    }

    /******************************************************************************//**
     * @brief Evaluate objective function
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return objective function value
    **********************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tObjFuncValue = mObjective->value(tStatesSubView, aControl);
        return tObjFuncValue;
    }

    /******************************************************************************//**
     * @brief Evaluate constraint function
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return constraint function value
    **********************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->value(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective function
     * @param [in] aControl 1D view of control variables
     * @return objective function value
    **********************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        Plato::ScalarMultiVector tStates = solution(aControl);
        auto tStatesSubView = Kokkos::subview(tStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->value(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint function
     * @param [in] aControl 1D view of control variables
     * @return constraint function value
    **********************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->value(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective gradient wrt control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - objective gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        // compute dfdz: partial of objective wrt z
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tPartialObjectiveWRT_Control = mObjective->gradient_z(tStatesSubView, aControl);
        if(mIsSelfAdjoint)
        {
            Plato::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_Control);
        }
        else
        {
            // compute dfdu: partial of objective wrt u
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(tStatesSubView, aControl);
            Plato::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mEqualityConstraint.gradient_u_T(tStatesSubView, aControl);

            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
#ifdef HAVE_AMGX
            typedef Plato::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode> AmgXLinearProblem;
            auto tConfigString = AmgXLinearProblem::getConfigString();
            auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, tAdjointSubView, tPartialObjectiveWRT_State, tConfigString));
            tSolver->solve();
            tSolver = Teuchos::null;
#endif

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mEqualityConstraint.gradient_z(tStatesSubView, aControl);

            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tAdjointSubView, tPartialObjectiveWRT_Control);
        }
        return tPartialObjectiveWRT_Control;
    }

    /******************************************************************************//**
     * @brief Evaluate objective gradient wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - objective gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        // compute partial derivative wrt x
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        auto tPartialObjectiveWRT_Config  = mObjective->gradient_x(tStatesSubView, aControl);

        if(mIsSelfAdjoint)
        {
            Plato::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_Config);
        }
        else
        {
            // compute dfdu: partial of objective wrt u
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(tStatesSubView, aControl);
            Plato::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mEqualityConstraint.gradient_u(tStatesSubView, aControl);
            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector
              tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
#ifdef HAVE_AMGX
            typedef Plato::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode> AmgXLinearProblem;
            auto tConfigString = AmgXLinearProblem::getConfigString();
            auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, tAdjointSubView, tPartialObjectiveWRT_State, tConfigString));
            tSolver->solve();
            tSolver = Teuchos::null;
#endif

            // compute dgdx: partial of PDE wrt config.
            // dgdx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mEqualityConstraint.gradient_x(tStatesSubView, aControl);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, tAdjointSubView, tPartialObjectiveWRT_Config);
        }
        return tPartialObjectiveWRT_Config;
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - objective partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate objective partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - objective partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            const std::string tErrorMessage = "OBJECTIVE POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->gradient_x(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    {
        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_x(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            const std::string tErrorMessage = "CONSTRAINT POINTER WAS NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage)
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_x(tStatesSubView, aControl);
    }

private:
    /******************************************************************************//**
     * @brief Initialize member data
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        Plato::ScalarFunctionBaseFactory<SimplexPhysics> tFunctionBaseFactory;
        if(aInputParams.isType<std::string>("Constraint"))
        {
            std::string tName = aInputParams.get<std::string>("Constraint");
            mConstraint = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);
        }

        if(aInputParams.isType<std::string>("Objective"))
        {
            std::string tName = aInputParams.get<std::string>("Objective");
            mObjective = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);

            auto tLength = mEqualityConstraint.size();
            mAdjoint = Plato::ScalarMultiVector("MyAdjoint", 1, tLength);
        }

        // parse constraints
        //
        Plato::EssentialBCs<SimplexPhysics>
            tEssentialBoundaryConditions(aInputParams.sublist("Essential Boundary Conditions",false));
        tEssentialBoundaryConditions.get(aMeshSets, mBcDofs, mBcValues);
    }
};
// class Problem

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::Problem<::Plato::Thermal<1>>;
extern template class Plato::Problem<::Plato::Mechanics<1>>;
extern template class Plato::Problem<::Plato::Electromechanics<1>>;
extern template class Plato::Problem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
extern template class Plato::Problem<::Plato::Thermal<2>>;
extern template class Plato::Problem<::Plato::Mechanics<2>>;
extern template class Plato::Problem<::Plato::Electromechanics<2>>;
extern template class Plato::Problem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
extern template class Plato::Problem<::Plato::Thermal<3>>;
extern template class Plato::Problem<::Plato::Mechanics<3>>;
extern template class Plato::Problem<::Plato::Electromechanics<3>>;
extern template class Plato::Problem<::Plato::Thermomechanics<3>>;
#endif

#endif // PLATO_PROBLEM_HPP
