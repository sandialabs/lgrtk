#pragma once

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/AbstractLocalVectorFunctionInc.hpp"
#include "plato/SimplexFadTypes.hpp"

#include "plato/J2PlasticityLocalResidual.hpp"
#include "plato/SimplexPlasticity.hpp"
#include "plato/Mechanics.hpp"

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
class LocalVectorFunctionInc
{
  private:
    static constexpr Plato::OrdinalType mNumDofsPerCell = PhysicsT::mNumDofsPerCell;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;
    static constexpr Plato::OrdinalType mNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    static constexpr Plato::OrdinalType mNumSpatialDims = PhysicsT::mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumControl = PhysicsT::mNumControl;

    const Plato::OrdinalType mNumNodes;
    const Plato::OrdinalType mNumCells;

    using Residual        = typename Plato::Evaluation<PhysicsT>::Residual;
    using GlobalJacobian  = typename Plato::Evaluation<PhysicsT>::Jacobian;
    using GlobalJacobianP = typename Plato::Evaluation<PhysicsT>::JacobianP;
    using LocalJacobian   = typename Plato::Evaluation<PhysicsT>::LocalJacobian;
    using LocalJacobianP  = typename Plato::Evaluation<PhysicsT>::LocalJacobianP;
    using GradientX       = typename Plato::Evaluation<PhysicsT>::GradientX;
    using GradientZ       = typename Plato::Evaluation<PhysicsT>::GradientZ;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;

    Plato::WorksetBase<PhysicsT> mWorksetBase;

    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<Residual>>        mLocalVectorFunctionResidual;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobian>>  mLocalVectorFunctionJacobianU;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobianP>> mLocalVectorFunctionJacobianUP;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobian>>   mLocalVectorFunctionJacobianC;
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobianP>>  mLocalVectorFunctionJacobianCP;
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
                           mWorksetBase(aMesh),
                           mNumCells(aMesh.nelems()),
                           mNumNodes(aMesh.nverts()),
                           mDataMap(aDataMap)
    {
      PRINTERR("Warning: Plato::Mechanics should be changed to PhysicsT or something similar eventually.")
      typename Plato::Mechanics<mNumSpatialDims>::FunctionFactory tFunctionFactory;

      mLocalVectorFunctionResidual  = tFunctionFactory.template createLocalVectorFunctionInc<Residual>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianU = tFunctionFactory.template createLocalVectorFunctionInc<GlobalJacobian>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianUP = tFunctionFactory.template createLocalVectorFunctionInc<GlobalJacobianP>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianC = tFunctionFactory.template createLocalVectorFunctionInc<LocalJacobian>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianCP = tFunctionFactory.template createLocalVectorFunctionInc<LocalJacobianP>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianZ = tFunctionFactory.template createLocalVectorFunctionInc<GradientZ>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mLocalVectorFunctionJacobianX = tFunctionFactory.template createLocalVectorFunctionInc<GradientX>
                                                                (aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      //  TESTING FOR COMPILATION BEFORE MAKING FUNCTION FACTORIES
      // mLocalVectorFunctionResidual  = 
      //         std::make_shared<J2PlasticityLocalResidual<Residual, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);

      // mLocalVectorFunctionJacobianU = 
      //         std::make_shared<J2PlasticityLocalResidual<GlobalJacobian, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);

      // mLocalVectorFunctionJacobianUP = 
      //         std::make_shared<J2PlasticityLocalResidual<GlobalJacobianP, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);

      // mLocalVectorFunctionJacobianC = 
      //         std::make_shared<J2PlasticityLocalResidual<LocalJacobian, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);

      // mLocalVectorFunctionJacobianCP = 
      //         std::make_shared<J2PlasticityLocalResidual<LocalJacobianP, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);

      // mLocalVectorFunctionJacobianZ = 
      //         std::make_shared<J2PlasticityLocalResidual<GradientZ, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);

      // mLocalVectorFunctionJacobianX = 
      //         std::make_shared<J2PlasticityLocalResidual<GradientX, Plato::SimplexPlasticity<mNumSpatialDims>>>
      //                                                 (aMesh, aMeshSets, aDataMap, aParamList);
    }

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    LocalVectorFunctionInc(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
                           mWorksetBase(aMesh),
                           mNumCells(aMesh.nelems()),
                           mNumNodes(aMesh.nverts()),
                           mLocalVectorFunctionResidual(),
                           mLocalVectorFunctionJacobianU(),
                           mLocalVectorFunctionJacobianUP(),
                           mLocalVectorFunctionJacobianC(),
                           mLocalVectorFunctionJacobianCP(),
                           mLocalVectorFunctionJacobianX(),
                           mLocalVectorFunctionJacobianZ(),
                           mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual residual evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<Residual>>& aResidual)
    {
        mLocalVectorFunctionResidual = aResidual;
    }

    /**************************************************************************//**
    *
    * @brief Allocate global jacobian evaluator
    * @param [in] aJacobian global jacobian evaluator
    *
    ******************************************************************************/
    void allocateJacobianU(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobian>>& aJacobian)
    {
        mLocalVectorFunctionJacobianU = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate previous global jacobian evaluator
    * @param [in] aJacobian previous global jacobian evaluator
    *
    ******************************************************************************/
    void allocateJacobianUP(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobianP>>& aJacobian)
    {
        mLocalVectorFunctionJacobianUP = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate local jacobian evaluator
    * @param [in] aJacobian local jacobian evaluator
    *
    ******************************************************************************/
    void allocateJacobianC(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobian>>& aJacobian)
    {
        mLocalVectorFunctionJacobianC = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate previous local jacobian evaluator
    * @param [in] aJacobian previous local jacobian evaluator
    *
    ******************************************************************************/
    void allocateJacobianCP(const std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobianP>>& aJacobian)
    {
        mLocalVectorFunctionJacobianCP = aJacobian;
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
    * @brief Return total number of local degrees of freedom
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

    /**************************************************************************//**
    * @brief Update the local state variables
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [out] aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    ******************************************************************************/
    void
    updateLocalState(const Plato::ScalarVector & aGlobalState,
                     const Plato::ScalarVector & aPrevGlobalState,
                     const Plato::ScalarVector & aLocalState,
                     const Plato::ScalarVector & aPrevLocalState,
                     const Plato::ScalarVector & aControl,
                     Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar         = typename Residual::ConfigScalarType;
      using StateScalar          = typename Residual::StateScalarType;
      using PrevStateScalar      = typename Residual::PrevStateScalarType;
      using LocalStateScalar     = typename Residual::LocalStateScalarType;
      using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
      using ControlScalar        = typename Residual::ControlScalarType;
      using ResultScalar         = typename Residual::ResultScalarType;

      // Workset global state
      //
      Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev Global State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      mWorksetBase.worksetControl(aControl, tControlWS);

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      mWorksetBase.worksetConfig(tConfigWS);

      // update the local state variables
      //
      mLocalVectorFunctionResidual->updateLocalState( tGlobalStateWS, tPrevGlobalStateWS, 
                                                      tLocalStateWS , tPrevLocalStateWS,
                                                      tControlWS    , tConfigWS, 
                                                      aTimeStep );

      auto tNumCells = mNumCells;
      auto tNumLocalDofsPerCell = mNumLocalDofsPerCell;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells),LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            Plato::OrdinalType tDofOrdinal = aCellOrdinal * tNumLocalDofsPerCell;
            for (Plato::OrdinalType tColumn = 0; tColumn < tNumLocalDofsPerCell; ++tColumn)
            {
              aLocalState(tDofOrdinal + tColumn) = tLocalStateWS(aCellOrdinal, tColumn);
            }
        }, "fill local state with updated version");
    }


    /**************************************************************************//**
    * @brief Compute the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return local residual vector
    ******************************************************************************/
    Plato::ScalarVectorT<typename Residual::ResultScalarType>
    value(const Plato::ScalarVector & aGlobalState,
          const Plato::ScalarVector & aPrevGlobalState,
          const Plato::ScalarVector & aLocalState,
          const Plato::ScalarVector & aPrevLocalState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar         = typename Residual::ConfigScalarType;
        using StateScalar          = typename Residual::StateScalarType;
        using PrevStateScalar      = typename Residual::PrevStateScalarType;
        using LocalStateScalar     = typename Residual::LocalStateScalarType;
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        using ControlScalar        = typename Residual::ControlScalarType;
        using ResultScalar         = typename Residual::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset global state
        //
        Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset prev global state
        //
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        //
        Plato::ScalarMultiVectorT<LocalStateScalar> 
                                 tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset prev local state
        //
        Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                                 tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tResidual("Residual", mNumCells, mNumLocalDofsPerCell);

        // evaluate function
        //
        mLocalVectorFunctionResidual->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                               tLocalStateWS,  tPrevLocalStateWS, 
                                               tControlWS, tConfigWS, tResidual, aTimeStep);
        Plato::OrdinalType tTotalNumLocalDofs = mNumCells * mNumLocalDofsPerCell;

        auto tNumLocalDofsPerCell = mNumLocalDofsPerCell;
        auto tNumCells = mNumCells;

        Plato::ScalarVector tResidualVector("Residual Vector", tTotalNumLocalDofs);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells),LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            Plato::OrdinalType tDofOrdinal = aCellOrdinal * tNumLocalDofsPerCell;
            for (Plato::OrdinalType tColumn = 0; tColumn < tNumLocalDofsPerCell; ++tColumn)
            {
              tResidualVector(tDofOrdinal + tColumn) = tResidual(aCellOrdinal, tColumn);
            }
        }, "flatten residual vector");

        return tResidualVector;
    }


    /**************************************************************************//**
    * @brief Compute the gradient wrt configuration of the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return gradient wrt configuration of the local residual vector
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename GradientX::ResultScalarType>
    gradient_x(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        using ConfigScalar         = typename GradientX::ConfigScalarType;
        using StateScalar          = typename GradientX::StateScalarType;
        using PrevStateScalar      = typename GradientX::PrevStateScalarType;
        using LocalStateScalar     = typename GradientX::LocalStateScalarType;
        using PrevLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
        using ControlScalar        = typename GradientX::ControlScalarType;
        using ResultScalar         = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset global state
        //
        Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
        mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

        // Workset prev global state
        //
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        //
        Plato::ScalarMultiVectorT<LocalStateScalar> 
                                 tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

        // Workset prev local state
        //
        Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                                 tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumLocalDofsPerCell);

        // evaluate function
        //
        mLocalVectorFunctionJacobianX->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                tLocalStateWS,  tPrevLocalStateWS, 
                                                tControlWS, tConfigWS, tJacobian, aTimeStep);

        return tJacobian;
    }

    /**************************************************************************//**
    * @brief Compute the gradient wrt global state of the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return gradient wrt global state of the local residual vector
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename GlobalJacobian::ResultScalarType>
    gradient_u(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar         = typename GlobalJacobian::ConfigScalarType;
      using StateScalar          = typename GlobalJacobian::StateScalarType;
      using PrevStateScalar      = typename GlobalJacobian::PrevStateScalarType;
      using LocalStateScalar     = typename GlobalJacobian::LocalStateScalarType;
      using PrevLocalStateScalar = typename GlobalJacobian::PrevLocalStateScalarType;
      using ControlScalar        = typename GlobalJacobian::ControlScalarType;
      using ResultScalar         = typename GlobalJacobian::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      mWorksetBase.worksetConfig(tConfigWS);

      // Workset global state
      //
      Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> 
                               tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                               tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      mWorksetBase.worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumLocalDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionJacobianU->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                              tLocalStateWS,  tPrevLocalStateWS, 
                                              tControlWS, tConfigWS, tJacobian, aTimeStep);

      return tJacobian;
    }

    /**************************************************************************//**
    * @brief Compute the gradient wrt previous global state of the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return gradient wrt previous global state of the local residual vector
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename GlobalJacobianP::ResultScalarType>
    gradient_up(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar         = typename GlobalJacobianP::ConfigScalarType;
      using StateScalar          = typename GlobalJacobianP::StateScalarType;
      using PrevStateScalar      = typename GlobalJacobianP::PrevStateScalarType;
      using LocalStateScalar     = typename GlobalJacobianP::LocalStateScalarType;
      using PrevLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
      using ControlScalar        = typename GlobalJacobianP::ControlScalarType;
      using ResultScalar         = typename GlobalJacobianP::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      mWorksetBase.worksetConfig(tConfigWS);

      // Workset global state
      //
      Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> 
                               tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                               tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      mWorksetBase.worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumLocalDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionJacobianUP->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                              tLocalStateWS,  tPrevLocalStateWS, 
                                              tControlWS, tConfigWS, tJacobian, aTimeStep);

      return tJacobian;
    }

    /**************************************************************************//**
    * @brief Compute the gradient wrt local state of the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return gradient wrt local state of the local residual vector
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename LocalJacobian::ResultScalarType>
    gradient_c(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar         = typename LocalJacobian::ConfigScalarType;
      using StateScalar          = typename LocalJacobian::StateScalarType;
      using PrevStateScalar      = typename LocalJacobian::PrevStateScalarType;
      using LocalStateScalar     = typename LocalJacobian::LocalStateScalarType;
      using PrevLocalStateScalar = typename LocalJacobian::PrevLocalStateScalarType;
      using ControlScalar        = typename LocalJacobian::ControlScalarType;
      using ResultScalar         = typename LocalJacobian::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      mWorksetBase.worksetConfig(tConfigWS);

      // Workset global state
      //
      Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> 
                               tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                               tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      mWorksetBase.worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianLocalState",mNumCells,mNumLocalDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionJacobianC->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                              tLocalStateWS,  tPrevLocalStateWS, 
                                              tControlWS, tConfigWS, tJacobian, aTimeStep);

      return tJacobian;
    }

    /**************************************************************************//**
    * @brief Compute the gradient wrt previous local state of the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return gradient wrt previous local state of the local residual vector
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename LocalJacobianP::ResultScalarType>
    gradient_cp(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar         = typename LocalJacobianP::ConfigScalarType;
      using StateScalar          = typename LocalJacobianP::StateScalarType;
      using PrevStateScalar      = typename LocalJacobianP::PrevStateScalarType;
      using LocalStateScalar     = typename LocalJacobianP::LocalStateScalarType;
      using PrevLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
      using ControlScalar        = typename LocalJacobianP::ControlScalarType;
      using ResultScalar         = typename LocalJacobianP::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      mWorksetBase.worksetConfig(tConfigWS);

      // Workset global state
      //
      Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> 
                               tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                               tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      mWorksetBase.worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianLocalState",mNumCells,mNumLocalDofsPerCell);

      // evaluate function
      //
      mLocalVectorFunctionJacobianCP->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                              tLocalStateWS,  tPrevLocalStateWS, 
                                              tControlWS, tConfigWS, tJacobian, aTimeStep);

      return tJacobian;
    }

    /**************************************************************************//**
    * @brief Compute the gradient wrt control of the local residual vector
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [in]  aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aTimeStep time step
    * @return gradient wrt control of the local residual vector
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename GradientZ::ResultScalarType>
    gradient_z(const Plato::ScalarVector & aGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      using ConfigScalar         = typename GradientZ::ConfigScalarType;
      using StateScalar          = typename GradientZ::StateScalarType;
      using PrevStateScalar      = typename GradientZ::PrevStateScalarType;
      using LocalStateScalar     = typename GradientZ::LocalStateScalarType;
      using PrevLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
      using ControlScalar        = typename GradientZ::ControlScalarType;
      using ResultScalar         = typename GradientZ::ResultScalarType;

      // Workset config
      //
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
      mWorksetBase.worksetConfig(tConfigWS);

      // Workset global state
      //
      Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

      // Workset prev global state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
      mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

      // Workset local state
      //
      Plato::ScalarMultiVectorT<LocalStateScalar> 
                               tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS);

      // Workset prev local state
      //
      Plato::ScalarMultiVectorT<PrevLocalStateScalar> 
                               tPrevLocalStateWS("Prev Local State Workset", mNumCells, mNumLocalDofsPerCell);
      mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      mWorksetBase.worksetControl(aControl, tControlWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",mNumCells,mNumLocalDofsPerCell);

      // evaluate function 
      //
      mLocalVectorFunctionJacobianZ->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                              tLocalStateWS,  tPrevLocalStateWS, 
                                              tControlWS, tConfigWS, tJacobian, aTimeStep);

      return tJacobian;
    }
};
// class LocalVectorFunctionInc

} // namespace Plato

#ifdef PLATO_2D
extern template class Plato::LocalVectorFunctionInc<Plato::SimplexPlasticity<2>>;
extern template class Plato::LocalVectorFunctionInc<Plato::SimplexThermoPlasticity<2>>;
#endif
#ifdef PLATO_3D
extern template class Plato::LocalVectorFunctionInc<Plato::SimplexPlasticity<3>>;
extern template class Plato::LocalVectorFunctionInc<Plato::SimplexThermoPlasticity<3>>;
#endif