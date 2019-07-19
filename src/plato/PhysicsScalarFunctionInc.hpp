#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "plato/ScalarFunctionIncBase.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Physics scalar function inc class 
 **********************************************************************************/
template<typename PhysicsT>
class PhysicsScalarFunctionInc : public ScalarFunctionIncBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::mNumControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::mStateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using JacobianP = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianP;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::shared_ptr<Plato::AbstractScalarFunctionInc<Residual>> mScalarFunctionValue; /*!< scalar function value interface */
    std::shared_ptr<Plato::AbstractScalarFunctionInc<Jacobian>> mScalarFunctionGradientU; /*!< scalar function value partial wrt states */
    std::shared_ptr<Plato::AbstractScalarFunctionInc<JacobianP>> mScalarFunctionGradientP;
    std::shared_ptr<Plato::AbstractScalarFunctionInc<GradientX>> mScalarFunctionGradientX; /*!< scalar function value partial wrt configuration */
    std::shared_ptr<Plato::AbstractScalarFunctionInc<GradientZ>> mScalarFunctionGradientZ; /*!< scalar function value partial wrt controls */

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName;/*!< User defined function name */

	/******************************************************************************//**
     * @brief Initialization of Physics Scalar Function Inc
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", ""); // Must be a hardcoded type name (e.g. Volume)

        mScalarFunctionValue =
            tFactory.template createScalarFunctionInc<Residual>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientU =
            tFactory.template createScalarFunctionInc<Jacobian>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientP =
            tFactory.template createScalarFunctionInc<JacobianP>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientX =
            tFactory.template createScalarFunctionInc<GradientX>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientZ =
            tFactory.template createScalarFunctionInc<GradientZ>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
    }

public:
    /******************************************************************************//**
     * @brief Primary physics scalar function inc constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunctionInc(Omega_h::Mesh& aMesh,
            Omega_h::MeshSets& aMeshSets,
            Plato::DataMap & aDataMap,
            Teuchos::ParameterList& aInputParams,
            std::string& aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary physics scalar function inc constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    PhysicsScalarFunctionInc(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(),
            mScalarFunctionGradientU(),
            mScalarFunctionGradientP(),
            mScalarFunctionGradientX(),
            mScalarFunctionGradientZ(),
            mDataMap(aDataMap),
            mFunctionName("Undefined Name")
    {
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateValue(const std::shared_ptr<Plato::AbstractScalarFunctionInc<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the Jacobian automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientU(const std::shared_ptr<Plato::AbstractScalarFunctionInc<Jacobian>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the Jacobian P automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientP(const std::shared_ptr<Plato::AbstractScalarFunctionInc<JacobianP>>& aInput)
    {
        mScalarFunctionGradientP = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientZ automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientZ(const std::shared_ptr<Plato::AbstractScalarFunctionInc<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientX automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientX(const std::shared_ptr<Plato::AbstractScalarFunctionInc<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }

    // /******************************************************************************//**
    //  * @brief Update physics-based parameters within optimization iterations
    //  * @param [in] aState 1D view of state variables
    //  * @param [in] aControl 1D view of control variables
    //  **********************************************************************************/
    // void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    // {
    //     Plato::ScalarMultiVector tStateWS("state workset", mNumCells, mNumDofsPerCell);
    //     Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

    //     Plato::ScalarMultiVector tControlWS("control workset", mNumCells, mNumNodesPerCell);
    //     Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

    //     Plato::ScalarArray3D tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
    //     Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

    //     mScalarFunctionValue->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientU->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientP->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientZ->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientX->updateProblem(tStateWS, tControlWS, tConfigWS);
    // }

    /******************************************************************************//**
     * @brief Evaluate physics scalar function
     * @param [in] aStates 2D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarMultiVector & aStates,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar    = typename Residual::ConfigScalarType;
        using StateScalar     = typename Residual::StateScalarType;
        using PrevStateScalar = typename Residual::PrevStateScalarType;
        using ControlScalar   = typename Residual::ControlScalarType;
        using ResultScalar    = typename Residual::ResultScalarType;

        // workset control
        // 
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        // 
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);
        mDataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;


        Plato::ScalarMultiVectorT<StateScalar>     tStateWS("state workset",mNumCells,mNumDofsPerCell);
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("prev state workset",mNumCells,mNumDofsPerCell);

        ResultScalar tReturnVal(0.0);

        auto tNumSteps = aStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(aStates, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        //
        auto tPrevState = Kokkos::subview(aStates, tStepIndex-1, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionValue->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // sum across elements
        //
        tReturnVal += Plato::local_result_sum<Plato::Scalar>(mNumCells, tResult);
        }

        mScalarFunctionValue->postEvaluate( tReturnVal );

        return tReturnVal;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarMultiVector & aStates,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using PrevStateScalar = typename GradientX::PrevStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("previous state workset",mNumCells,mNumDofsPerCell);

        // create return view
        //
        Plato::Scalar tObjectiveValue(0.0);
        Plato::ScalarVector tObjGradientX("objective gradient configuration",mNumSpatialDims*mNumNodes);

        // workset control
        // 
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        // 
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);


        auto tNumSteps = aStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(aStates, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        //
        auto tPrevState = Kokkos::subview(aStates, tStepIndex-1, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientX->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        Plato::assemble_vector_gradient<mNumNodesPerCell, mNumSpatialDims>(mNumCells, mConfigEntryOrdinal, tResult, tObjGradientX);
        tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        }

        mScalarFunctionGradientX->postEvaluate( tObjGradientX, tObjectiveValue );

        return tObjGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step
     * @param [in] aStepIndex step index
     * @return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarMultiVector & aStates,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep,
                                   Plato::OrdinalType aStepIndex) const
    {
        using ConfigScalar    = typename Jacobian::ConfigScalarType;
        using StateScalar     = typename Jacobian::StateScalarType;
        using PrevStateScalar = typename Jacobian::PrevStateScalarType;
        using ControlScalar   = typename Jacobian::ControlScalarType;
        using ResultScalar    = typename Jacobian::ResultScalarType;

        assert(aStepIndex < aStates.extent(0));
        assert(aStates.extent(0) > 1);
        assert(aStepIndex > 0);

        auto tNumSteps = aStates.extent(0);

        // workset control
        // 
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        // 
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // workset state
        // 
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        auto tState = Kokkos::subview(aStates, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        // 
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("previous state workset",mNumCells,mNumDofsPerCell);
        auto tPrevState = Kokkos::subview(aStates, aStepIndex-1, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        mScalarFunctionGradientU->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state",mNumDofsPerNode*mNumNodes);
        Plato::assemble_vector_gradient<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mStateEntryOrdinal, tResult, tObjGradientU);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        if( aStepIndex+1 < tNumSteps ) {

        // workset next state
        // 
        Plato::ScalarMultiVectorT<PrevStateScalar> tNextStateWS("next state workset",mNumCells,mNumDofsPerCell);
        auto tNextState = Kokkos::subview(aStates, aStepIndex+1, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tNextState, tNextStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientP->evaluate( tNextStateWS, tStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::assemble_vector_gradient<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mStateEntryOrdinal, tResult, tObjGradientU);
        tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        }

        mScalarFunctionGradientU->postEvaluate( tObjGradientU, tObjectiveValue );

        return tObjGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarMultiVector & aStates,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {        
        using ConfigScalar    = typename GradientZ::ConfigScalarType;
        using StateScalar     = typename GradientZ::StateScalarType;
        using PrevStateScalar = typename GradientZ::PrevStateScalarType;
        using ControlScalar   = typename GradientZ::ControlScalarType;
        using ResultScalar    = typename GradientZ::ResultScalarType;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("previous state workset",mNumCells,mNumDofsPerCell);

        // initialize objective value to zero
        //
        Plato::Scalar tObjectiveValue(0.0);

        // workset control
        // 
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        // 
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control",mNumNodes);

        auto tNumSteps = aStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(aStates, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        //
        auto tPrevState = Kokkos::subview(aStates, tStepIndex-1, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientZ->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        Plato::assemble_scalar_gradient<mNumNodesPerCell>(mNumCells, mControlEntryOrdinal, tResult, tObjGradientZ);

        tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        }

        mScalarFunctionGradientZ->postEvaluate( tObjGradientZ, tObjectiveValue );

        return tObjGradientZ;
    }

    /******************************************************************************//**
     * @brief Set user defined function name
     * @param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * @brief Return user defined function name
     * @return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
//class PhysicsScalarFunctionInc

}
//namespace Plato

#include "Thermal.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::PhysicsScalarFunctionInc<::Plato::Thermal<1>>;
extern template class Plato::PhysicsScalarFunctionInc<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::PhysicsScalarFunctionInc<::Plato::Thermal<2>>;
extern template class Plato::PhysicsScalarFunctionInc<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::PhysicsScalarFunctionInc<::Plato::Thermal<3>>;
extern template class Plato::PhysicsScalarFunctionInc<::Plato::Thermomechanics<3>>;
#endif