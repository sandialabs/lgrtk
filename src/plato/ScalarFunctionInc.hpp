#ifndef SCALAR_FUNCTION_INC_HPP
#define SCALAR_FUNCTION_INC_HPP

#include <memory>
#include <cassert>

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"

/******************************************************************************/
/*! objective class

   This class takes as a template argument a scalar function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control. status
  
*/
/******************************************************************************/
template<typename PhysicsT>
class ScalarFunctionInc : public WorksetBase<PhysicsT>
{
  private:
    using WorksetBase<PhysicsT>::m_numDofsPerCell;
    using WorksetBase<PhysicsT>::m_numNodesPerCell;
    using WorksetBase<PhysicsT>::m_numDofsPerNode;
    using WorksetBase<PhysicsT>::m_numSpatialDims;
    using WorksetBase<PhysicsT>::m_numControl;
    using WorksetBase<PhysicsT>::m_numNodes;
    using WorksetBase<PhysicsT>::m_numCells;

    using WorksetBase<PhysicsT>::m_stateEntryOrdinal;
    using WorksetBase<PhysicsT>::m_controlEntryOrdinal;
    using WorksetBase<PhysicsT>::m_configEntryOrdinal;

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using JacobianP = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianP;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    static constexpr Plato::OrdinalType m_numConfigDofsPerCell = m_numSpatialDims*m_numNodesPerCell;

    std::shared_ptr<AbstractScalarFunctionInc<Residual>>  mScalarFunctionValue;
    std::shared_ptr<AbstractScalarFunctionInc<Jacobian>>  mScalarFunctionGradientU;
    std::shared_ptr<AbstractScalarFunctionInc<JacobianP>> mScalarFunctionGradientP;
    std::shared_ptr<AbstractScalarFunctionInc<GradientX>> mScalarFunctionGradientX;
    std::shared_ptr<AbstractScalarFunctionInc<GradientZ>> mScalarFunctionGradientZ;

     Plato::DataMap& m_dataMap;

  public:

    /**************************************************************************/
    ScalarFunctionInc(Omega_h::Mesh& aMesh, 
                      Omega_h::MeshSets& aMeshSets,
                      Plato::DataMap & aDataMap,
                      Teuchos::ParameterList& aParamList,
                      const std::string & aScalarFunctionType ) :
            WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap)
    /**************************************************************************/
    {
      typename PhysicsT::FunctionFactory tFactory;

      mScalarFunctionValue
        = tFactory.template createScalarFunctionInc<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientU
        = tFactory.template createScalarFunctionInc<Jacobian>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientP
        = tFactory.template createScalarFunctionInc<JacobianP>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientZ
        = tFactory.template createScalarFunctionInc<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientX
        = tFactory.template createScalarFunctionInc<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);
    }

    /**************************************************************************/
    ScalarFunctionInc(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(nullptr),
            mScalarFunctionGradientU(nullptr),
            mScalarFunctionGradientP(nullptr),
            mScalarFunctionGradientX(nullptr),
            mScalarFunctionGradientZ(nullptr),
            m_dataMap(aDataMap)
    {
    }

    void allocateValue(const std::shared_ptr<AbstractScalarFunctionInc<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }

    void allocateGradientU(const std::shared_ptr<AbstractScalarFunctionInc<Jacobian>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }

    void allocateGradientP(const std::shared_ptr<AbstractScalarFunctionInc<JacobianP>>& aInput)
    {
        mScalarFunctionGradientP = aInput;
    }

    void allocateGradientZ(const std::shared_ptr<AbstractScalarFunctionInc<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }

    void allocateGradientX(const std::shared_ptr<AbstractScalarFunctionInc<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }

    /**************************************************************************/
    Plato::Scalar value(const Plato::ScalarMultiVector & aStates,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Residual::ConfigScalarType;
      using StateScalar     = typename Residual::StateScalarType;
      using PrevStateScalar = typename Residual::PrevStateScalarType;
      using ControlScalar   = typename Residual::ControlScalarType;
      using ResultScalar    = typename Residual::ResultScalarType;

      // workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",m_numCells,m_numNodesPerCell);

      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);

      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result view
      //
      Plato::ScalarVectorT<ResultScalar> tResult("result",m_numCells);

      m_dataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;


      Plato::ScalarMultiVectorT<StateScalar>     tStateWS("state workset",m_numCells,m_numDofsPerCell);
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("prev state workset",m_numCells,m_numDofsPerCell);

      ResultScalar tReturnVal(0.0);

      auto tNumSteps = aStates.extent(0);
      auto tLastStepIndex = tNumSteps - 1;
      for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(aStates, tStepIndex, Kokkos::ALL());

        WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        //
        auto tPrevState = Kokkos::subview(aStates, tStepIndex-1, Kokkos::ALL());

        WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionValue->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // sum across elements
        //
        tReturnVal += Plato::local_result_sum<Plato::Scalar>(m_numCells, tResult);
      }

      mScalarFunctionValue->postEvaluate( tReturnVal );

      return tReturnVal;
    }

    /**************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarMultiVector & aStates,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename GradientX::ConfigScalarType;
      using StateScalar     = typename GradientX::StateScalarType;
      using PrevStateScalar = typename GradientX::PrevStateScalarType;
      using ControlScalar   = typename GradientX::ControlScalarType;
      using ResultScalar    = typename GradientX::ResultScalarType;

      Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",m_numCells,m_numDofsPerCell);
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("previous state workset",m_numCells,m_numDofsPerCell);

      // create return view
      //
      Plato::Scalar tObjectiveValue(0.0);
      Plato::ScalarVector tObjGradientX("objective gradient configuration",m_numSpatialDims*m_numNodes);

      // workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",m_numCells,m_numNodesPerCell);

      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);

      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      //
      Plato::ScalarVectorT<ResultScalar> tResult("result",m_numCells);


      auto tNumSteps = aStates.extent(0);
      auto tLastStepIndex = tNumSteps - 1;
      for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(aStates, tStepIndex, Kokkos::ALL());

        WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        //
        auto tPrevState = Kokkos::subview(aStates, tStepIndex-1, Kokkos::ALL());

        WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientX->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        Plato::assemble_vector_gradient<m_numNodesPerCell, m_numSpatialDims>(m_numCells, m_configEntryOrdinal, tResult, tObjGradientX);
        tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      }

      mScalarFunctionGradientX->postEvaluate( tObjGradientX, tObjectiveValue );

      return tObjGradientX;
    }
    
    /**************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarMultiVector & aStates,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep,
                                   Plato::OrdinalType aStepIndex) const
    /**************************************************************************/
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
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      //
      Plato::ScalarVectorT<ResultScalar> tResult("result",m_numCells);

      // workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",m_numCells,m_numDofsPerCell);
      auto tState = Kokkos::subview(aStates, aStepIndex, Kokkos::ALL());
      WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

      // workset prev state
      // 
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("previous state workset",m_numCells,m_numDofsPerCell);
      auto tPrevState = Kokkos::subview(aStates, aStepIndex-1, Kokkos::ALL());
      WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

      // evaluate function
      //
      mScalarFunctionGradientU->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

      // create and assemble to return view
      //
      Plato::ScalarVector tObjGradientU("objective gradient state",m_numDofsPerNode*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells, m_stateEntryOrdinal, tResult, tObjGradientU);
      Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      if( aStepIndex+1 < tNumSteps ) {

        // workset next state
        // 
        Plato::ScalarMultiVectorT<PrevStateScalar> tNextStateWS("next state workset",m_numCells,m_numDofsPerCell);
        auto tNextState = Kokkos::subview(aStates, aStepIndex+1, Kokkos::ALL());
        WorksetBase<PhysicsT>::worksetState(tNextState, tNextStateWS);
  
        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientP->evaluate( tNextStateWS, tStateWS, tControlWS, tConfigWS, tResult, aTimeStep );
  
        // create and assemble to return view
        //
        Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells, m_stateEntryOrdinal, tResult, tObjGradientU);
        tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);
      }

      mScalarFunctionGradientU->postEvaluate( tObjGradientU, tObjectiveValue );

      return tObjGradientU;
    }
    
    /**************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarMultiVector & aStates,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename GradientZ::ConfigScalarType;
      using StateScalar     = typename GradientZ::StateScalarType;
      using PrevStateScalar = typename GradientZ::PrevStateScalarType;
      using ControlScalar   = typename GradientZ::ControlScalarType;
      using ResultScalar    = typename GradientZ::ResultScalarType;

      Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",m_numCells,m_numDofsPerCell);
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("previous state workset",m_numCells,m_numDofsPerCell);

      // initialize objective value to zero
      //
      Plato::Scalar tObjectiveValue(0.0);

      // workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",m_numCells,m_numNodesPerCell);

      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);

      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result view
      //
      Plato::ScalarVectorT<ResultScalar> tResult("result",m_numCells);

      // create return vector
      //
      Plato::ScalarVector tObjGradientZ("objective gradient control",m_numNodes);

      auto tNumSteps = aStates.extent(0);
      auto tLastStepIndex = tNumSteps - 1;
      for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(aStates, tStepIndex, Kokkos::ALL());

        WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset prev state
        //
        auto tPrevState = Kokkos::subview(aStates, tStepIndex-1, Kokkos::ALL());

        WorksetBase<PhysicsT>::worksetState(tPrevState, tPrevStateWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientZ->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

        Plato::assemble_scalar_gradient<m_numNodesPerCell>(m_numCells, m_controlEntryOrdinal, tResult, tObjGradientZ);

        tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      }

      mScalarFunctionGradientZ->postEvaluate( tObjGradientZ, tObjectiveValue );

      return tObjGradientZ;
    }
};

#endif
