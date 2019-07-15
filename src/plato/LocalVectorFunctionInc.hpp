#pragma once

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/AbstractLocalVectorFunctionInc.hpp"
#include "plato/SimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! local vector function class

   This class takes as a template argument a vector function in the form:

   H = H(U^k, U^{k-1}, C^k, C^{k-1}, X)

   and manages the evaluation of the function and derivatives wrt global state, U^k; 
   previous global state, U^{k-1}; local state, C^k; 
   previous local state, C^{k-1}; and control, X.
*/
/******************************************************************************/
template<typename PhysicsT>
class LocalVectorFunctionInc : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumLocalDofsPerCell; // Added
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual        = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GlobalJacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GlobalJacobian;  // Changed
    using GlobalJacobianP = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GlobalJacobianP;
    using LocalJacobian   = typename Plato::Evaluation<typename PhysicsT::SimplexT>::LocalJacobian;   // Added
    using LocalJacobianP  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::LocalJacobianP;
    using GradientX       = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ       = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;

    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<Residual>>        mLocalVectorFunctionResidual;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobian>>  mLocalVectorFunctionGlobalJacobianU;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobianP>> mLocalVectorFunctionGlobalJacobianUP;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobian>>   mLocalVectorFunctionLocalJacobianC;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobianP>>  mLocalVectorFunctionLocalJacobianCP;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GradientX>>       mLocalVectorFunctionJacobianX;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GradientZ>>       mLocalVectorFunctionJacobianZ;

    Plato::DataMap& mDataMap;

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
    LocalVectorFunctionInc(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap& aDataMap,
                           Teuchos::ParameterList& aParamList,
                           std::string& aProblemType) :
                           Plato::WorksetBase<PhysicsT>(aMesh),
                           mDataMap(aDataMap)
    {
      typename PhysicsT::FunctionFactory tFunctionFactory;

      mLocalVectorFunctionResidual  = tFunctionFactory.template createLocalVectorFunctionInc<Residual>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianU = tFunctionFactory.template createLocalVectorFunctionInc<Jacobian>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianP = tFunctionFactory.template createLocalVectorFunctionInc<JacobianP>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianZ = tFunctionFactory.template createLocalVectorFunctionInc<GradientZ>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianX = tFunctionFactory.template createLocalVectorFunctionInc<GradientX>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
    }

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    LocalVectorFunctionInc(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
                           Plato::WorksetBase<PhysicsT>(aMesh),
                           mLocalVectorFunctionResidual(),
                           mLocalVectorFunctionJacobianU(),
                           mLocalVectorFunctionJacobianP(),
                           mLocalVectorFunctionJacobianX(),
                           mLocalVectorFunctionJacobianZ(),
                           mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual residual evaluator
    * @param [in] aJacobian jacobian evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<Residual>>& aResidual,
                          const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<Jacobian>>& aJacobian)
    {
        mLocalVectorFunctionResidual  = aResidual;
        mLocalVectorFunctionJacobianU = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateJacobianZ(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GradientZ>>& aGradientZ)
    {
        mLocalVectorFunctionJacobianZ = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateJacobianX(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GradientX>>& aGradientX)
    {
        mLocalVectorFunctionJacobianX = aGradientX; 
    }

    /**************************************************************************//**
    *
    * @brief Return local number of local degrees of freedom per cell
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumCells * mNumLocalDofsPerCell;
    }

    /**************************************************************************//**
    *
    * @brief Return state names
    *
    ******************************************************************************/
    std::vector<std::string> getDofNames() const
    {
      return mLocalVectorFunctionResidual->getDofNames();
    }

    /**************************************************************************/
    void
    value(const Plato::ScalarVector & aGlobalState,
          const Plato::ScalarVector & aPrevGlobalState,
          const Plato::ScalarVector & aLocalState,
          const Plato::ScalarVector & aPrevLocalState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar          = typename Residual::ConfigScalarType;

      using GlobalStateScalar     = typename Residual::GlobalStateScalarType;
      using PrevGlobalStateScalar = typename Residual::PrevGlobalStateScalarType;

      using LocalStateScalar      = typename Residual::LocalStateScalarType;
      using PrevLocalStateScalar  = typename Residual::PrevLocalStateScalarType;

      using ControlScalar         = typename Residual::ControlScalarType;

      using ResultScalar          = typename Residual::ResultScalarType;

      // Workset global state
      //
      Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumLocalDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevGlobalStateScalar> tPrevGlobalStateWS("Prev Global State Workset", mNumCells, mNumLocalDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result
      //
      Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", mNumCells, mNumLocalDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionResidual->updateLocalState( tGlobalStateWS, tPrevGlobalStateWS, 
                                                      tLocalStateWS , tPrevLocalStateWS,
                                                      tControlWS    , tConfigWS, 
                                                      aTimeStep );
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using PrevStateScalar = typename GradientX::PrevStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset prev state
        //
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

        // evaluate function
        //
        mLocalVectorFunctionJacobianX->evaluate(tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        // create return matrix
        //
        auto tMesh = mLocalVectorFunctionJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(&tMesh);

        // assembly to return matrix
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tJacobianMatEntries = tJacobianMat->entries();
        Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Jacobian::ConfigScalarType;
      using StateScalar     = typename Jacobian::StateScalarType;
      using PrevStateScalar = typename Jacobian::PrevStateScalarType;
      using ControlScalar   = typename Jacobian::ControlScalarType;
      using ResultScalar    = typename Jacobian::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      // 
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionJacobianU->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mLocalVectorFunctionJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_p(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename JacobianP::ConfigScalarType;
      using StateScalar     = typename JacobianP::StateScalarType;
      using PrevStateScalar = typename JacobianP::PrevStateScalarType;
      using ControlScalar   = typename JacobianP::ControlScalarType;
      using ResultScalar    = typename JacobianP::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      // 
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionJacobianP->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mLocalVectorFunctionJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename GradientZ::ConfigScalarType;
      using StateScalar     = typename GradientZ::StateScalarType;
      using PrevStateScalar = typename GradientZ::PrevStateScalarType;
      using ControlScalar   = typename GradientZ::ControlScalarType;
      using ResultScalar    = typename GradientZ::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",mNumCells,mNumDofsPerCell);

      // evaluate function 
      //
      mLocalVectorFunctionJacobianZ->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mLocalVectorFunctionJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return (tJacobianMat);
    }
};
// class LocalVectorFunctionInc

} // namespace Plato
