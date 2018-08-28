#ifndef LGR_PLATO_SCALAR_FUNCTION_HPP
#define LGR_PLATO_SCALAR_FUNCTION_HPP

#include <memory>
#include <cassert>

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"

/******************************************************************************/
/*! objective class

   This class takes as a template argument a scalar function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control. status
  
*/
/******************************************************************************/
template<typename PhysicsT>
class ScalarFunction : public WorksetBase<PhysicsT>
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

    using Residual  = typename Plato::Evaluation<PhysicsT>::Residual;
    using Jacobian  = typename Plato::Evaluation<PhysicsT>::Jacobian;
    using GradientX = typename Plato::Evaluation<PhysicsT>::GradientX;
    using GradientZ = typename Plato::Evaluation<PhysicsT>::GradientZ;

    static constexpr Plato::OrdinalType m_numConfigDofsPerCell = m_numSpatialDims*m_numNodesPerCell;

    std::shared_ptr<AbstractScalarFunction<Residual>>  mScalarFunctionValue;
    std::shared_ptr<AbstractScalarFunction<Jacobian>>  mScalarFunctionGradientU;
    std::shared_ptr<AbstractScalarFunction<GradientX>> mScalarFunctionGradientX;
    std::shared_ptr<AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ;

     Plato::DataMap& m_dataMap;

  public:

    /**************************************************************************/
    ScalarFunction(Omega_h::Mesh& aMesh, 
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
        = tFactory.template createScalarFunction<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientU
        = tFactory.template createScalarFunction<Jacobian>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientZ
        = tFactory.template createScalarFunction<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);

      mScalarFunctionGradientX
        = tFactory.template createScalarFunction<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aScalarFunctionType);
    }

    /**************************************************************************/
    ScalarFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(nullptr),
            mScalarFunctionGradientU(nullptr),
            mScalarFunctionGradientX(nullptr),
            mScalarFunctionGradientZ(nullptr),
            m_dataMap(aDataMap)
    {
    }

    void allocateValue(const std::shared_ptr<AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }

    void allocateGradientU(const std::shared_ptr<AbstractScalarFunction<Jacobian>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }

    void allocateGradientZ(const std::shared_ptr<AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }

    void allocateGradientX(const std::shared_ptr<AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }

    /**************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename Residual::ConfigScalarType;
      using StateScalar   = typename Residual::StateScalarType;
      using ControlScalar = typename Residual::ControlScalarType;
      using ResultScalar  = typename Residual::ResultScalarType;

      // workset state
      //
      Plato::ScalarMultiVectorT<StateScalar>
        tStateWS("state workset",m_numCells,m_numDofsPerCell);

      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",m_numCells,m_numNodesPerCell);

      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result view
      //
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      m_dataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;

      // evaluate function
      //
      mScalarFunctionValue->evaluate( tStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

      // sum across elements
      //
      auto tReturnVal = Plato::local_result_sum<Plato::Scalar>(m_numCells, tResult);

      mScalarFunctionValue->postEvaluate( tReturnVal );

      return tReturnVal;
    }

    /**************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename GradientX::ConfigScalarType;
      using StateScalar   = typename GradientX::StateScalarType;
      using ControlScalar = typename GradientX::ControlScalarType;
      using ResultScalar  = typename GradientX::ResultScalarType;

      // workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar>
        tStateWS("state workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar>
        tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      //
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      // evaluate function
      //
      mScalarFunctionGradientX->evaluate( tStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

      // create and assemble to return view
      //
      Plato::ScalarVector tObjGradientX("objective gradient configuration",m_numSpatialDims*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numSpatialDims>(m_numCells, m_configEntryOrdinal, tResult, tObjGradientX);
      Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      mScalarFunctionGradientX->postEvaluate( tObjGradientX, tObjectiveValue );

      return tObjGradientX;
    }
    
    /**************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename Jacobian::ConfigScalarType;
      using StateScalar   = typename Jacobian::StateScalarType;
      using ControlScalar = typename Jacobian::ControlScalarType;
      using ResultScalar  = typename Jacobian::ResultScalarType;

      // workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar>
        tStateWS("sacado-ized state",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar>
        tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create return view
      //
      Plato::ScalarVectorT<ResultScalar>
        tResult("result",m_numCells);

      // evaluate function
      //
      mScalarFunctionGradientU->evaluate( tStateWS, tControlWS, tConfigWS, tResult, aTimeStep );

      // create and assemble to return view
      //
      Plato::ScalarVector tObjGradientU("objective gradient state",m_numDofsPerNode*m_numNodes);
      Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells, m_stateEntryOrdinal, tResult, tObjGradientU);
      Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      mScalarFunctionGradientU->postEvaluate( tObjGradientU, tObjectiveValue );

      return tObjGradientU;
    }
    
    /**************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename GradientZ::ConfigScalarType;
      using StateScalar   = typename GradientZ::StateScalarType;
      using ControlScalar = typename GradientZ::ControlScalarType;
      using ResultScalar  = typename GradientZ::ResultScalarType;

      // workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar>
       tControlWS("control workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // workset state
      //
      Plato::ScalarMultiVectorT<StateScalar>
        tStateWS("state workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
        tConfigWS("config workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result 
      //
      Plato::ScalarVectorT<ResultScalar>
        tResult("elastic energy",m_numCells);

      // evaluate function 
      //
      mScalarFunctionGradientZ->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

      // create and assemble to return view
      //
      Plato::ScalarVector tObjGradientZ("objective gradient control",m_numNodes);
      Plato::assemble_scalar_gradient<m_numNodesPerCell>(m_numCells, m_controlEntryOrdinal, tResult, tObjGradientZ);
      Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

      mScalarFunctionGradientZ->postEvaluate( tObjGradientZ, tObjectiveValue );

      return tObjGradientZ;
    }
};

#endif
