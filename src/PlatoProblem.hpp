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

#include "plato/Thermal.hpp"
#include "plato/Mechanics.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/ScalarFunction.hpp"
#include "plato/ScalarFunction.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PlatoAbstractProblem.hpp"

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

/**********************************************************************************/
template<typename SimplexPhysics>
class Problem: public Plato::AbstractProblem
{
/**********************************************************************************/
private:

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::m_numSpatialDims;

    // required
    VectorFunction<SimplexPhysics> mEqualityConstraint;

    // optional
    std::shared_ptr<const ScalarFunction<SimplexPhysics>> mConstraint;
    std::shared_ptr<const ScalarFunction<SimplexPhysics>> mObjective;

    Plato::ScalarVector mAdjoint;
    Plato::ScalarVector mResidual;
    Plato::ScalarVector mBoundaryLoads;

    Plato::ScalarMultiVector mStates;

    bool mIsSelfAdjoint;

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian;

    Plato::LocalOrdinalVector mBcDofs;
    Plato::ScalarVector mBcValues;

public:
    /******************************************************************************/
    Problem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList) :
            mEqualityConstraint(aMesh, aMeshSets, mDataMap, aParamList, aParamList.get < std::string > ("PDE Constraint")),
            mConstraint(nullptr),
            mObjective(nullptr),
            mResidual("MyResidual", mEqualityConstraint.size()),
            mBoundaryLoads("BoundaryLoads", mEqualityConstraint.size()),
            mStates("States", static_cast<Plato::OrdinalType>(1), mEqualityConstraint.size()),
            mJacobian(Teuchos::null),
            mIsSelfAdjoint(aParamList.get<bool>("Self-Adjoint", false))
    /******************************************************************************/
    {
        this->initialize(aMesh, aMeshSets, aParamList);
    }

    /******************************************************************************/
    void setState(const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));
        Kokkos::deep_copy(mStates, aState);
    }

    /******************************************************************************/
    Plato::ScalarMultiVector getState()
    /******************************************************************************/
    {
        return mStates;
    }

    /******************************************************************************/
    Plato::ScalarVector getAdjoint()
    /******************************************************************************/
    {
        return mAdjoint;
    }

    /******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    /******************************************************************************/
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SpatialDim>(aMatrix, aVector, mBcDofs, mBcValues);
        }
        else
        {
            Plato::applyConstraints<SpatialDim>(aMatrix, aVector, mBcDofs, mBcValues);
        }
    }

    /******************************************************************************/
    void applyBoundaryLoads(const Plato::ScalarVector & aForce)
    /******************************************************************************/
    {
        auto tBoundaryLoads = mBoundaryLoads;
        auto tNumDofs = aForce.size();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal){
            aForce(aDofOrdinal) += tBoundaryLoads(aDofOrdinal);
        }, "add boundary loads");
    }

    /******************************************************************************/
    Plato::ScalarMultiVector solution(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        Plato::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mResidual = mEqualityConstraint.value(tStatesSubView, aControl);

        mJacobian = mEqualityConstraint.gradient_u(tStatesSubView, aControl);
        this->applyConstraints(mJacobian, mResidual);

#ifdef HAVE_AMGX
        using AmgXLinearProblem = lgr::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode>;
        auto tConfigString = AmgXLinearProblem::getConfigString();
        auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, tStatesSubView, mResidual, tConfigString));
        tSolver->solve();
        tSolver = Teuchos::null;
#endif

        mResidual = mEqualityConstraint.value(tStatesSubView, aControl);
        return mStates;
    }

    /******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->value(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->value(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        Plato::ScalarMultiVector tStates = solution(aControl);
        auto tStatesSubView = Kokkos::subview(tStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->value(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->value(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
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
            mJacobian = mEqualityConstraint.gradient_u(tStatesSubView, aControl);

            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

#ifdef HAVE_AMGX
            typedef lgr::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode> AmgXLinearProblem;
            auto tConfigString = AmgXLinearProblem::getConfigString();
            auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, mAdjoint, tPartialObjectiveWRT_State, tConfigString));
            tSolver->solve();
            tSolver = Teuchos::null;
#endif

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mEqualityConstraint.gradient_z(tStatesSubView, aControl);

            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, mAdjoint, tPartialObjectiveWRT_Control);
        }
        return tPartialObjectiveWRT_Control;
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
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

#ifdef HAVE_AMGX
            typedef lgr::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode> AmgXLinearProblem;
            auto tConfigString = AmgXLinearProblem::getConfigString();
            auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, mAdjoint, tPartialObjectiveWRT_State, tConfigString));
            tSolver->solve();
            tSolver = Teuchos::null;
#endif

            // compute dgdx: partial of PDE wrt config.
            // dgdx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mEqualityConstraint.gradient_x(tStatesSubView, aControl);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, mAdjoint, tPartialObjectiveWRT_Config);
        }
        return tPartialObjectiveWRT_Config;
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->gradient_z(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mObjective->gradient_x(tStatesSubView, aControl);
    }


    /******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_x(tStatesSubView, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
    /******************************************************************************/
    {
        assert(aState.extent(0) == mStates.extent(0));
        assert(aState.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());
        return mConstraint->gradient_x(tStatesSubView, aControl);
    }

private:
    /******************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    /******************************************************************************/
    {
        if(aParamList.isType<std::string>("Linear Constraint"))
        {
            std::string tName = aParamList.get<std::string>("Linear Constraint");
            mConstraint =
                    std::make_shared<ScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);
        }

        if(aParamList.isType<std::string>("Objective"))
        {
            std::string tName = aParamList.get<std::string>("Objective");
            mObjective = std::make_shared<ScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);

            auto tLength = mEqualityConstraint.size();
            mAdjoint = Plato::ScalarVector("MyAdjoint", tLength);
        }

        // parse constraints
        //
        Plato::EssentialBCs<SimplexPhysics>
            tEssentialBoundaryConditions(aParamList.sublist("Essential Boundary Conditions",false));
        tEssentialBoundaryConditions.get(aMeshSets, mBcDofs, mBcValues);

        // parse loads
        //
        Plato::NaturalBCs<SimplexPhysics::SpaceDim, SimplexPhysics::m_numDofsPerNode>
            tNaturalBoundaryConditions(aParamList.sublist("Natural Boundary Conditions", false));
        tNaturalBoundaryConditions.get(&aMesh, aMeshSets, mBoundaryLoads);
    }
};

#endif // PLATO_PROBLEM_HPP
