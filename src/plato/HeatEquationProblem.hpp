#ifndef PLATO_HEAT_EQUATION_PROBLEM_HPP
#define PLATO_HEAT_EQUATION_PROBLEM_HPP

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
#include "plato/VectorFunctionInc.hpp"
#include "plato/ScalarFunctionInc.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PlatoAbstractProblem.hpp"
#include "plato/ComputedField.hpp"

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

/**********************************************************************************/
template<typename SimplexPhysics>
class HeatEquationProblem: public Plato::AbstractProblem
{
/**********************************************************************************/
private:

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::m_numSpatialDims;

    // required
    VectorFunctionInc<SimplexPhysics> mEqualityConstraint;

    Plato::OrdinalType mNumSteps;
    Plato::Scalar mTimeStep;

    // optional
    std::shared_ptr<const ScalarFunction<SimplexPhysics>> mConstraint;
    std::shared_ptr<const ScalarFunctionInc<SimplexPhysics>> mObjective;

    Plato::ScalarMultiVector mAdjoints;
    Plato::ScalarVector mResidual;
    Plato::ScalarVector mBoundaryLoads;

    Plato::ScalarMultiVector mStates;

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobianP;

    std::string mInitialTemperatureCF;
    Teuchos::RCP<Plato::ComputedFields<SpatialDim>> mComputedFields;

    bool mIsSelfAdjoint;

    Plato::LocalOrdinalVector mBcDofs;
    Plato::ScalarVector mBcValues;

public:
    /******************************************************************************/
    HeatEquationProblem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList) :
            mEqualityConstraint(aMesh, aMeshSets, mDataMap, aParamList, aParamList.get < std::string > ("PDE Constraint")),
            mNumSteps(aParamList.sublist("Time Integration").get<int>("Number Time Steps")),
            mTimeStep(aParamList.sublist("Time Integration").get<double>("Time Step")),
            mConstraint(nullptr),
            mObjective(nullptr),
            mResidual("MyResidual", mEqualityConstraint.size()),
            mBoundaryLoads("BoundaryLoads", mEqualityConstraint.size()),
            mStates("States", mNumSteps, mEqualityConstraint.size()),
            mJacobian(Teuchos::null),
            mJacobianP(Teuchos::null),
            mComputedFields(Teuchos::null),
            mIsSelfAdjoint(aParamList.get<bool>("Self-Adjoint", false))
    /******************************************************************************/
    {
        this->initialize(aMesh, aMeshSets, aParamList);
    }

    /******************************************************************************/
    void setState(const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));
        Kokkos::deep_copy(mStates, aStates);
    }

    /******************************************************************************/
    Plato::ScalarMultiVector getState()
    /******************************************************************************/
    {
        return mStates;
    }

    /******************************************************************************/
    Plato::ScalarMultiVector getAdjoint()
    /******************************************************************************/
    {
        return mAdjoints;
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
        // initialize first state
        //
        if (!mInitialTemperatureCF.empty()) {
          Plato::ScalarVector firstState = Kokkos::subview(mStates, 0, Kokkos::ALL());
          mComputedFields->get(mInitialTemperatureCF, firstState);
        }

        for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
          Plato::ScalarVector tState = Kokkos::subview(mStates, tStepIndex, Kokkos::ALL());
          Plato::ScalarVector tPrevState = Kokkos::subview(mStates, tStepIndex-1, Kokkos::ALL());
          Plato::fill(static_cast<Plato::Scalar>(0.0), tState);

          mResidual = mEqualityConstraint.value(tState, tPrevState, aControl, mTimeStep);

          mJacobian = mEqualityConstraint.gradient_u(tState, tPrevState, aControl, mTimeStep);
          this->applyConstraints(mJacobian, mResidual);

#ifdef HAVE_AMGX
          using AmgXLinearProblem = lgr::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode>;
          auto tConfigString = AmgXLinearProblem::getConfigString();
          Plato::ScalarVector deltaT("increment", tState.extent(0));
          Plato::fill(static_cast<Plato::Scalar>(0.0), deltaT);
          auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, deltaT, mResidual, tConfigString));
          tSolver->solve();
          tSolver = Teuchos::null;
          Plato::axpy(-1.0, deltaT, tState);
#endif

        }
        return mStates;
    }

    /******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->value(aStates, aControl);
    }

    /******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->value(tState, aControl);
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

        Plato::ScalarMultiVector tStates = solution(aControl);
        return mObjective->value(tStates, aControl);
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

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->value(tState, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        // compute dFd\phi: partial of objective wrt control
        auto tTotalObjectiveWRT_Control = mObjective->gradient_z(aStates, aControl, mTimeStep);

        // compute lagrange multiplier at the last time step, n
        

        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

            auto tState     = Kokkos::subview(aStates,   tStepIndex,   Kokkos::ALL());
            auto tPrevState = Kokkos::subview(aStates,   tStepIndex-1, Kokkos::ALL());
            Plato::ScalarVector tAdjoint   = Kokkos::subview(mAdjoints, tStepIndex,   Kokkos::ALL());

            // compute dFdT^k: partial of objective wrt T at step k = tStepIndex
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(aStates, aControl, mTimeStep, tStepIndex);

            if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1
                Plato::ScalarVector tNextState   = Kokkos::subview(aStates,   tStepIndex+1, Kokkos::ALL());
                Plato::ScalarVector tNextAdjoint = Kokkos::subview(mAdjoints, tStepIndex+1, Kokkos::ALL());
                // compute dQ^{k+1}/dT^k: partial of PDE at k+1 wrt current state, k.
                mJacobianP = mEqualityConstraint.gradient_p(tNextState, tState, aControl, mTimeStep);

                // multiply dQ^{k+1}/dT^k by lagrange multiplier from k+1 and add to dFdT^k
                Plato::MatrixTimesVectorPlusVector(mJacobianP, tNextAdjoint, tPartialObjectiveWRT_State);
            }
            Plato::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dQ^k/dT^k: partial of PDE at k wrt state current state, k.
            mJacobian = mEqualityConstraint.gradient_u(tState, tPrevState, aControl, mTimeStep);

            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

#ifdef HAVE_AMGX
            typedef lgr::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode> AmgXLinearProblem;
            auto tConfigString = AmgXLinearProblem::getConfigString();
            auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, tAdjoint, tPartialObjectiveWRT_State, tConfigString));
            tSolver->solve();
            tSolver = Teuchos::null;
#endif
            // compute dQ^k/d\phi: partial of PDE wrt control at step k.
            // dQ^k/d\phi is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mEqualityConstraint.gradient_z(tState, tPrevState, aControl, mTimeStep);
    
            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tAdjoint, tTotalObjectiveWRT_Control);

        }

        return tTotalObjectiveWRT_Control;
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));

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
        auto tPartialObjectiveWRT_Config  = mObjective->gradient_x(aStates, aControl, mTimeStep);

        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

            auto tState     = Kokkos::subview(aStates,   tStepIndex,   Kokkos::ALL());
            auto tPrevState = Kokkos::subview(aStates,   tStepIndex-1, Kokkos::ALL());
            Plato::ScalarVector tAdjoint   = Kokkos::subview(mAdjoints, tStepIndex,   Kokkos::ALL());

            // compute dFdT^k: partial of objective wrt T at step k = tStepIndex
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(aStates, aControl, mTimeStep, tStepIndex);
            Plato::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1
                auto tNextState   = Kokkos::subview(aStates,   tStepIndex+1, Kokkos::ALL());
                Plato::ScalarVector tNextAdjoint = Kokkos::subview(mAdjoints, tStepIndex+1, Kokkos::ALL());
                // compute dQ^{k+1}/dT^k: partial of PDE at k+1 wrt current state, k.
                mJacobianP = mEqualityConstraint.gradient_p(tNextState, tState, aControl, mTimeStep);

                // multiply dQ^{k+1}/dT^k by lagrange multiplier from k+1 and add to dFdT^k
                Plato::MatrixTimesVectorPlusVector(mJacobianP, tNextAdjoint, tPartialObjectiveWRT_State);
            }

            // compute dQ^k/dT^k: partial of PDE at k wrt state current state, k.
            mJacobian = mEqualityConstraint.gradient_u(tState, tPrevState, aControl, mTimeStep);

            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

#ifdef HAVE_AMGX
            typedef lgr::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::m_numDofsPerNode> AmgXLinearProblem;
            auto tConfigString = AmgXLinearProblem::getConfigString();
            auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*mJacobian, tAdjoint, tPartialObjectiveWRT_State, tConfigString));
            tSolver->solve();
            tSolver = Teuchos::null;
#endif

            // compute dQ^k/dx: partial of PDE wrt config.
            // dQ^k/dx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mEqualityConstraint.gradient_x(tState, tPrevState, aControl, mTimeStep);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, tAdjoint, tPartialObjectiveWRT_Config);
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
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_z(tState, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_z(tState, aControl);
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
        return mObjective->gradient_z(mStates, aControl, mTimeStep);
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
        return mObjective->gradient_x(mStates, aControl, mTimeStep);
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
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_x(tState, aControl, mTimeStep);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mStates.extent(0));
        assert(aStates.extent(1) == mStates.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_x(tState, aControl, mTimeStep);
    }

private:
    /******************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    /******************************************************************************/
    {
        if(aParamList.isSublist("Computed Fields"))
        {
          mComputedFields = Teuchos::rcp(new Plato::ComputedFields<SpatialDim>(aMesh, aParamList.sublist("Computed Fields")));
        }

        if(aParamList.isSublist("Initial Temperature"))
        {
          if(mComputedFields == Teuchos::null) {
            throw std::runtime_error("No 'Computed Fields' have been defined");
          }
          mInitialTemperatureCF = aParamList.sublist("Initial Temperature").get<std::string>("Computed Field");
          mComputedFields->find(mInitialTemperatureCF);
        }

        if(aParamList.isType<std::string>("Linear Constraint"))
        {
            std::string tName = aParamList.get<std::string>("Linear Constraint");
            mConstraint =
                    std::make_shared<ScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);
        }

        if(aParamList.isType<std::string>("Objective"))
        {
            std::string tName = aParamList.get<std::string>("Objective");
            mObjective = std::make_shared<ScalarFunctionInc<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tName);

            auto tLength = mEqualityConstraint.size();
            mAdjoints = Plato::ScalarMultiVector("MyAdjoint", mNumSteps, tLength);
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
