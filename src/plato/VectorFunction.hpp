#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/SimplexFadTypes.hpp"

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction : public WorksetBase<PhysicsT>
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

    using Residual  = typename Plato::Evaluation<PhysicsT>::Residual;
    using Jacobian  = typename Plato::Evaluation<PhysicsT>::Jacobian;
    using GradientX = typename Plato::Evaluation<PhysicsT>::GradientX;
    using GradientZ = typename Plato::Evaluation<PhysicsT>::GradientZ;

    static constexpr Plato::OrdinalType m_numConfigDofsPerCell = m_numSpatialDims*m_numNodesPerCell;

    std::shared_ptr<AbstractVectorFunction<Residual>>  mVectorFunctionResidual;
    std::shared_ptr<AbstractVectorFunction<Jacobian>>  mVectorFunctionJacobianU;
    std::shared_ptr<AbstractVectorFunction<GradientX>> mVectorFunctionJacobianX;
    std::shared_ptr<AbstractVectorFunction<GradientZ>> mVectorFunctionJacobianZ;

    Plato::DataMap& m_dataMap;

  public:

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aMeshSets mesh sets data base
    * @param [in] aDataMap problem-specific data map 
    * @param [in] aParamList Teuchos parameter list with input data
    * @param [in] aProblemType problem type 
    *
    ******************************************************************************/
    VectorFunction(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap& aDataMap,
                   Teuchos::ParameterList& aParamList,
                   std::string& aProblemType) :
            WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap)
    {
      typename PhysicsT::FunctionFactory tFunctionFactory;

      mVectorFunctionResidual = tFunctionFactory.template createVectorFunction<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianU = tFunctionFactory.template createVectorFunction<Jacobian>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianZ = tFunctionFactory.template createVectorFunction<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianX = tFunctionFactory.template createVectorFunction<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
    }

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    VectorFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            WorksetBase<PhysicsT>(aMesh),
            mVectorFunctionResidual(),
            mVectorFunctionJacobianU(),
            mVectorFunctionJacobianX(),
            mVectorFunctionJacobianZ(),
            m_dataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual residual evaluator
    * @param [in] aJacobian Jacobian evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<AbstractVectorFunction<Residual>>& aResidual,
                          const std::shared_ptr<AbstractVectorFunction<Jacobian>>& aJacobian)
    {
        mVectorFunctionResidual = aResidual;
        mVectorFunctionJacobianU = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateJacobianZ(const std::shared_ptr<AbstractVectorFunction<GradientZ>>& aGradientZ)
    {
        mVectorFunctionJacobianZ = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateJacobianX(const std::shared_ptr<AbstractVectorFunction<GradientX>>& aGradientX)
    {
        mVectorFunctionJacobianX = aGradientX; 
    }

    /**************************************************************************//**
    *
    * @brief Return local number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return m_numNodes*m_numDofsPerNode;
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(const Plato::ScalarVector & aState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename Residual::ConfigScalarType;
      using StateScalar   = typename Residual::StateScalarType;
      using ControlScalar = typename Residual::ControlScalarType;
      using ResultScalar  = typename Residual::ResultScalarType;

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result
      //
      Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual",m_numCells, m_numDofsPerCell);

      // evaluate function
      //
      mVectorFunctionResidual->evaluate( tStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

      // create and assemble to return view
      //
      Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tReturnValue("Assembled Residual",m_numDofsPerNode*m_numNodes);
      WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue );

      return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using ConfigScalar = typename GradientX::ConfigScalarType;
        using StateScalar = typename GradientX::StateScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", m_numCells, m_numDofsPerCell);
        WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", m_numCells, m_numNodesPerCell);
        WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", m_numCells, m_numDofsPerCell);

        // evaluate function
        //
        mVectorFunctionJacobianX->evaluate(tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        // create return matrix
        //
        auto tMesh = mVectorFunctionJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numSpatialDims, m_numDofsPerNode>(&tMesh);

        // assembly to return matrix
        Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numSpatialDims, m_numDofsPerNode>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tJacobianMatEntries = tJacobianMat->entries();
        WorksetBase<PhysicsT>::assembleTransposeJacobian(m_numDofsPerCell, m_numConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename Jacobian::ConfigScalarType;
      using StateScalar   = typename Jacobian::StateScalarType;
      using ControlScalar = typename Jacobian::ControlScalarType;
      using ResultScalar  = typename Jacobian::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",m_numCells,m_numDofsPerCell);

      // evaluate function
      //
      mVectorFunctionJacobianU->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numDofsPerNode, m_numDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, m_numDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVectorT<Plato::Scalar> & aState,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar  = typename GradientZ::ConfigScalarType;
      using StateScalar   = typename GradientZ::StateScalarType;
      using ControlScalar = typename GradientZ::ControlScalarType;
      using ResultScalar  = typename GradientZ::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",m_numCells,m_numDofsPerCell);

      // evaluate function 
      //
      mVectorFunctionJacobianZ->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numControl, m_numDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numControl, m_numDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      WorksetBase<PhysicsT>::assembleTransposeJacobian(m_numDofsPerCell, m_numNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }
};

#endif
