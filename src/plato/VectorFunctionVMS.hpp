#ifndef VECTOR_FUNCTION_VMS_HPP
#define VECTOR_FUNCTION_VMS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/AbstractVectorFunctionVMS.hpp"
#include "plato/SimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunctionVMS : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::m_numNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::m_numSpatialDims;
    using Plato::WorksetBase<PhysicsT>::m_numControl;
    using Plato::WorksetBase<PhysicsT>::m_numNodes;
    using Plato::WorksetBase<PhysicsT>::m_numCells;

    using Plato::WorksetBase<PhysicsT>::m_numNSPerNode;
    using Plato::WorksetBase<PhysicsT>::m_numNSPerCell;

    using Plato::WorksetBase<PhysicsT>::m_stateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::m_controlEntryOrdinal;

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using JacobianN = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianN;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    static constexpr Plato::OrdinalType m_numConfigDofsPerCell = m_numSpatialDims*m_numNodesPerCell;

    std::shared_ptr<Plato::AbstractVectorFunctionVMS<Residual>>  mVectorFunctionVMSResidual;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<Jacobian>>  mVectorFunctionVMSJacobianU;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<JacobianN>> mVectorFunctionVMSJacobianN;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientX>> mVectorFunctionVMSJacobianX;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientZ>> mVectorFunctionVMSJacobianZ;

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
    VectorFunctionVMS(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap& aDataMap,
                   Teuchos::ParameterList& aParamList,
                   std::string aProblemType) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap)
    {
      typename PhysicsT::FunctionFactory tFunctionFactory;

      mVectorFunctionVMSResidual = tFunctionFactory.template createVectorFunctionVMS<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianU = tFunctionFactory.template createVectorFunctionVMS<Jacobian>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianN = tFunctionFactory.template createVectorFunctionVMS<JacobianN>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianZ = tFunctionFactory.template createVectorFunctionVMS<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianX = tFunctionFactory.template createVectorFunctionVMS<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
    }

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    VectorFunctionVMS(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mVectorFunctionVMSResidual(),
            mVectorFunctionVMSJacobianU(),
            mVectorFunctionVMSJacobianN(),
            mVectorFunctionVMSJacobianX(),
            mVectorFunctionVMSJacobianZ(),
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
    void allocateResidual(const std::shared_ptr<Plato::AbstractVectorFunctionVMS<Residual>>& aResidual,
                          const std::shared_ptr<Plato::AbstractVectorFunctionVMS<Jacobian>>& aJacobian)
    {
        mVectorFunctionVMSResidual = aResidual;
        mVectorFunctionVMSJacobianU = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateJacobianZ(const std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientZ>>& aGradientZ)
    {
        mVectorFunctionVMSJacobianZ = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateJacobianX(const std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientX>>& aGradientX)
    {
        mVectorFunctionVMSJacobianX = aGradientX; 
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
          const Plato::ScalarVector & aNodeState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Residual::ConfigScalarType;
      using StateScalar     = typename Residual::StateScalarType;
      using NodeStateScalar = typename Residual::NodeStateScalarType;
      using ControlScalar   = typename Residual::ControlScalarType;
      using ResultScalar    = typename Residual::ResultScalarType;

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", m_numCells, m_numNSPerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result
      //
      Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual",m_numCells, m_numDofsPerCell);

      // evaluate function
      //
      mVectorFunctionVMSResidual->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

      // create and assemble to return view
      //
      Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tReturnValue("Assembled Residual",m_numDofsPerNode*m_numNodes);
      Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue );

      return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", m_numCells, m_numNSPerCell);
        Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", m_numCells, m_numSpatialDims*m_numNodesPerCell);

        // evaluate function
        //
        mVectorFunctionVMSJacobianX->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        // create return matrix
        //
        auto tMesh = mVectorFunctionVMSJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numSpatialDims, m_numDofsPerNode>(&tMesh);

        // assembly to return matrix
        Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numSpatialDims, m_numDofsPerNode>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tJacobianMatEntries = tJacobianMat->entries();
        Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(m_numDofsPerCell, m_numConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(const Plato::ScalarVector & aState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Jacobian::ConfigScalarType;
      using StateScalar     = typename Jacobian::StateScalarType;
      using NodeStateScalar = typename Jacobian::NodeStateScalarType;
      using ControlScalar   = typename Jacobian::ControlScalarType;
      using ResultScalar    = typename Jacobian::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",m_numCells, m_numNSPerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",m_numCells,m_numDofsPerCell);

      // evaluate function
      //
      mVectorFunctionVMSJacobianU->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionVMSJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixTransposeEntryOrdinal<m_numSpatialDims, m_numDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, m_numDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }
    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Jacobian::ConfigScalarType;
      using StateScalar     = typename Jacobian::StateScalarType;
      using NodeStateScalar = typename Jacobian::NodeStateScalarType;
      using ControlScalar   = typename Jacobian::ControlScalarType;
      using ResultScalar    = typename Jacobian::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",m_numCells, m_numNSPerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",m_numCells,m_numDofsPerCell);

      // evaluate function
      //
      mVectorFunctionVMSJacobianU->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionVMSJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numDofsPerNode, m_numDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numDofsPerNode, m_numDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(m_numDofsPerCell, m_numDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }
    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename JacobianN::ConfigScalarType;
      using StateScalar     = typename JacobianN::StateScalarType;
      using NodeStateScalar = typename JacobianN::NodeStateScalarType;
      using ControlScalar   = typename JacobianN::ControlScalarType;
      using ResultScalar    = typename JacobianN::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", m_numCells, m_numDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", m_numCells, m_numNSPerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", m_numCells, m_numNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianNodeState",m_numCells, m_numNSPerCell);

      // evaluate function
      //
      mVectorFunctionVMSJacobianN->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionVMSJacobianN->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numSpatialDims, m_numNSPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numSpatialDims, m_numNSPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(m_numDofsPerCell, m_numNSPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVectorT<Plato::Scalar> & aState,
               const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename GradientZ::ConfigScalarType;
      using StateScalar     = typename GradientZ::StateScalarType;
      using NodeStateScalar = typename GradientZ::NodeStateScalarType;
      using ControlScalar   = typename GradientZ::ControlScalarType;
      using ResultScalar    = typename GradientZ::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",m_numCells, m_numNodesPerCell, m_numSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",m_numCells,m_numNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",m_numCells,m_numDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",m_numCells, m_numNSPerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",m_numCells,m_numDofsPerCell);

      // evaluate function 
      //
      mVectorFunctionVMSJacobianZ->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionVMSJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, m_numControl, m_numDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<m_numSpatialDims, m_numControl, m_numDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(m_numDofsPerCell, m_numNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }
};
// class VectorFunctionVMS

} // namespace Plato

#endif
