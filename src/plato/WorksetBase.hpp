#ifndef WORKSET_BASE_HPP
#define WORKSET_BASE_HPP

#include <cassert>

#include <Omega_h_mesh.hpp>

#include "ImplicitFunctors.hpp"
#include "plato/SimplexFadTypes.hpp"

namespace Plato
{

template <class Scalar, class Result>
inline Scalar local_result_sum(const Plato::OrdinalType& aNumCells, const Result & aResult)
{
  Scalar tReturnVal(0.0);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal, Scalar& aLocalResult)
  {
    aLocalResult += aResult(aCellOrdinal);
  }, tReturnVal);
  return tReturnVal;
}
// function local_result_sum

/*************************************************************************//**
*
* @brief Assemble scalar function global value
*
* Assemble scalar function global value from local values.
*
* @fn Scalar assemble_scalar_func_value(const Plato::OrdinalType& aNumCells, const Result& aResult)
* @tparam Scalar typename of return value
* @tparam Result result vector view typename
* @param aNumCells number of cells (i.e. elements)
* @param aResult scalar vector
* @return global function value
*
*****************************************************************************/
template <class Scalar, class Result>
inline Scalar assemble_scalar_func_value(const Plato::OrdinalType& aNumCells, const Result& aResult)
{
  Scalar tReturnValue(0.0);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType& aCellOrdinal, Scalar & aLocalValue)
  {
    aLocalValue += aResult(aCellOrdinal).val();
  }, tReturnValue);
  return tReturnValue;
}
// function assemble_scalar_func_value

/*************************************************************************//**
*
* @brief Assemble vector gradient of a scalar function
*
* @fn void assemble_vector_gradient(const Plato::OrdinalType& aNumCells,const EntryOrdinal& aEntryOrdinal,const Gradient& aGradient,ReturnVal& aOutput)
* @tparam NumNodesPerCell number of nodes per cells (i.e. elements)
* @tparam NumDofsPerNode number of degrees of freedom per node
* @tparam EntryOrdinal entry ordinal view type
* @tparam Gradient gradient workset view type
* @tparam ReturnVal output (i.e. assembled gradient) view type
* @param aNumCells number of cells
* @param aEntryOrdinal global indices to output vector
* @param aGradien gradient workset - gradient values for each cell
* @param aOutput assembled global gradient
*
* *****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, Plato::OrdinalType NumDofsPerNode, class EntryOrdinal, class Gradient, class ReturnVal>
inline void assemble_vector_gradient(const Plato::OrdinalType& aNumCells,
                                     const EntryOrdinal& aEntryOrdinal,
                                     const Gradient& aGradient,
                                     ReturnVal& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex < NumDofsPerNode; tDimIndex++)
            {
                Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(aCellOrdinal, tNodeIndex, tDimIndex);
                Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal).dx(tNodeIndex * NumDofsPerNode + tDimIndex));
            }
        }
    }, "Assemble - Vector Gradient Calculation");
}
// function assemble_vector_gradient

/*************************************************************************//**
*
* @brief Assemble scalar gradient of a scalar function
*
* @fn void assemble_scalar_gradient(const Plato::OrdinalType& aNumCells,const EntryOrdinal& aEntryOrdinal,const Gradient& aGradient,ReturnVal& aOutput)
* @tparam NumNodesPerCell number of nodes per cell
* @tparam EntryOrdinal entry ordinal view type
* @tparam Gradient gradient workset view type
* @tparam ReturnVal output (i.e. assembled gradient) view type
* @param aNumCells number of cells (i.e. elements)
* @param aEntryOrdinal global indices to output vector
* @param aGradien gradient workset - gradient values for each cell
* @param aOutput assembled global gradient
*
*****************************************************************************/
template<Plato::OrdinalType NumNodesPerCell, class EntryOrdinal, class Gradient, class ReturnVal>
inline void assemble_scalar_gradient(const Plato::OrdinalType& aNumCells,
                                     const EntryOrdinal& aEntryOrdinal,
                                     const Gradient& aGradient,
                                     ReturnVal& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      for(Plato::OrdinalType tNodeIndex=0; tNodeIndex < NumNodesPerCell; tNodeIndex++)
      {
          Plato::OrdinalType tEntryOrdinal = aEntryOrdinal(aCellOrdinal, tNodeIndex);
          Kokkos::atomic_add(&aOutput(tEntryOrdinal), aGradient(aCellOrdinal).dx(tNodeIndex));
      }
    }, "Assemble - Scalar Gradient Calculation");
}
// function assemble_scalar_gradient

/******************************************************************************/
template<int numNodesPerCell, class ControlEntryOrdinal, class Control, class ControlWS>
inline void workset_control_scalar_scalar(int aNumCells,
                                          ControlEntryOrdinal aControlEntryOrdinal,
                                          Control aControl,
                                          ControlWS aControlWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++)
        {
            int tEntryOrdinal = aControlEntryOrdinal(aCellOrdinal, tNodeIndex);
            aControlWS(aCellOrdinal,tNodeIndex) = aControl(tEntryOrdinal);
        }
    }, "workset_control_scalar_scalar");
}
// function workset_control_scalar_scalar

/******************************************************************************/
template<int numNodesPerCell, class ControlFad, class ControlEntryOrdinal, class Control, class FadControlWS>
inline void workset_control_scalar_fad(int aNumCells,
                                       ControlEntryOrdinal aControlEntryOrdinal,
                                       Control aControl,
                                       FadControlWS aFadControlWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++)
        {
            int tEntryOrdinal = aControlEntryOrdinal(aCellOrdinal, tNodeIndex);
            aFadControlWS(aCellOrdinal,tNodeIndex) = ControlFad( numNodesPerCell, tNodeIndex, aControl(tEntryOrdinal));
        }
    }, "workset_control_scalar_fad");
}
// function workset_control_scalar_fad

/******************************************************************************/
template<int numDofsPerNode, int numNodesPerCell, class StateEntryOrdinal, class State, class StateWS>
inline void workset_state_scalar_scalar(int aNumCells, StateEntryOrdinal aStateEntryOrdinal, State aState, StateWS aStateWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for(int tDofIndex = 0; tDofIndex < numDofsPerNode; tDofIndex++)
        {
            for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++)
            {
                int tEntryOrdinal = aStateEntryOrdinal(aCellOrdinal, tNodeIndex, tDofIndex);
                int tLocalDof = (tNodeIndex * numDofsPerNode) + tDofIndex;
                aStateWS(aCellOrdinal, tLocalDof) = aState(tEntryOrdinal);
            }
        }
    }, "workset_state_scalar_scalar");
}
// function workset_state_scalar_scalar

/******************************************************************************/
template<int numDofsPerNode, int numNodesPerCell, class StateFad, class StateEntryOrdinal, class State, class FadStateWS>
inline void workset_state_scalar_fad(int aNumCells,
                                     StateEntryOrdinal aStateEntryOrdinal,
                                     State aState,
                                     FadStateWS aFadStateWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for(int tDofIndex = 0; tDofIndex < numDofsPerNode; tDofIndex++)
        {
            for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++)
            {
                int tEntryOrdinal = aStateEntryOrdinal(aCellOrdinal, tNodeIndex, tDofIndex);
                int tLocalDof = tNodeIndex * numDofsPerNode + tDofIndex;
                aFadStateWS(aCellOrdinal,tLocalDof) = StateFad(numDofsPerNode*numNodesPerCell, tLocalDof, aState(tEntryOrdinal));
            }
        }
    }, "workset_state_scalar_fad");
}
// function workset_state_scalar_fad

/******************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class State, class StateWS>
inline void workset_local_state_scalar_scalar(Plato::OrdinalType aNumCells, State & aState, StateWS & aStateWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      printf("%d %d %d %d %d\n", aNumCells, aCellOrdinal, aState.size(), aStateWS.size(), aStateWS.extent(1));
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          Plato::OrdinalType tGlobalDof = (aCellOrdinal * NumLocalDofsPerCell) + tDofIndex;
          aStateWS(aCellOrdinal, tDofIndex) = aState(tGlobalDof);
        }
    }, "workset_local_state_scalar_scalar");
}
// function workset_local_state_scalar_scalar

/******************************************************************************/
template<Plato::OrdinalType NumLocalDofsPerCell, class StateFad, class State, class FadStateWS>
inline void workset_local_state_scalar_fad(Plato::OrdinalType aNumCells,
                                           State aState,
                                           FadStateWS aFadStateWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
      printf("%d %d %d %d %d\n", aNumCells, aCellOrdinal, aState.size(), aFadStateWS.extent(0), aFadStateWS.extent(1));
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumLocalDofsPerCell; tDofIndex++)
        {
          Plato::OrdinalType tGlobalDof = (aCellOrdinal * NumLocalDofsPerCell) + tDofIndex;
          aFadStateWS(aCellOrdinal,tDofIndex) = StateFad(NumLocalDofsPerCell, tDofIndex, aState(tGlobalDof));
        }
    }, "workset_local_state_scalar_fad");
}
// function workset_local_state_scalar_fad

/******************************************************************************/
template<int spaceDim, int numNodesPerCell, class ConfigWS, class NodeCoordinates>
inline void workset_config_scalar(int aNumCells, NodeCoordinates aNodeCoordinate, ConfigWS aConfigWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for(int tDimIndex = 0; tDimIndex < spaceDim; tDimIndex++)
        {
            for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++)
            {
                aConfigWS(aCellOrdinal,tNodeIndex,tDimIndex) = aNodeCoordinate(aCellOrdinal,tNodeIndex,tDimIndex);
            }
        }
    }, "workset_config_scalar");
}
// function workset_config_scalar

/******************************************************************************/
template<int spaceDim, int numNodesPerCell, int numConfigDofsPerCell, class ConfigFad, class FadConfigWS, class NodeCoordinates>
inline void workset_config_fad(int aNumCells, NodeCoordinates aNodeCoordinate, FadConfigWS aFadConfigWS)
/******************************************************************************/
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for(int tDimIndex = 0; tDimIndex < spaceDim; tDimIndex++)
        {
            for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++)
            {
                int tLocalDim = tNodeIndex * spaceDim + tDimIndex;
                aFadConfigWS(aCellOrdinal,tNodeIndex,tDimIndex) =
                        ConfigFad(numConfigDofsPerCell, tLocalDim, aNodeCoordinate(aCellOrdinal,tNodeIndex,tDimIndex));
            }
        }
    }, "workset_config_fad");
}
// function workset_config_fad

/******************************************************************************/
template<int numNodesPerCell, int numDofsPerNode, class StateEntryOrdinal, class Residual, class ReturnVal>
inline void assemble_residual(int aNumCells, 
                              const StateEntryOrdinal & aStateEntryOrdinal, 
                              const Residual & aResidual, 
                              ReturnVal & aReturnValue)
/******************************************************************************/
{
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
  {
    for(int tNodeIndex = 0; tNodeIndex < numNodesPerCell; tNodeIndex++){
      for(int tDofIndex = 0; tDofIndex < numDofsPerNode; tDofIndex++){
        int tEntryOrdinal = aStateEntryOrdinal(aCellOrdinal, tNodeIndex, tDofIndex);
        Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aResidual(aCellOrdinal,tNodeIndex*numDofsPerNode+tDofIndex));
      }
    }
  }, "assemble_residual");
}
// function assemble_residual

/******************************************************************************/
template<class MatrixEntriesOrdinal, class Jacobian, class ReturnVal>
inline void assemble_jacobian(int aNumCells,
                              int aNumRowsPerCell,
                              int aNumColumnsPerCell,
                              const MatrixEntriesOrdinal & aMatrixEntryOrdinal,
                              const Jacobian & aJacobianWorkset,
                              ReturnVal & aReturnValue)
/******************************************************************************/
{
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
  {
    for(int tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++){
      for(int tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++){
        int tEntryOrdinal = aMatrixEntryOrdinal(aCellOrdinal, tRowIndex, tColumnIndex);
        Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal,tRowIndex).dx(tColumnIndex));
      }
    }
  }, "assemble_jacobian");
}
// function assemble_jacobian

/******************************************************************************/
template<class MatrixEntriesOrdinal, class Jacobian, class ReturnVal>
inline void assemble_transpose_jacobian(int aNumCells,
                                        int aNumRowsPerCell,
                                        int aNumColumnsPerCell,
                                        const MatrixEntriesOrdinal & aMatrixEntryOrdinal,
                                        const Jacobian & aJacobianWorkset,
                                        ReturnVal & aReturnValue)
/******************************************************************************/
{
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
  {
    for(int tRowIndex = 0; tRowIndex < aNumRowsPerCell; tRowIndex++){
      for(int tColumnIndex = 0; tColumnIndex < aNumColumnsPerCell; tColumnIndex++){
        int tEntryOrdinal = aMatrixEntryOrdinal(aCellOrdinal, tColumnIndex, tRowIndex);
        Kokkos::atomic_add(&aReturnValue(tEntryOrdinal), aJacobianWorkset(aCellOrdinal,tRowIndex).dx(tColumnIndex));
      }
    }
  }, "assemble_transpose_jacobian");
}
// function assemble_transpose_jacobian

/******************************************************************************/
/*! Base class for workset functionality.
*/
/******************************************************************************/
template<typename SimplexPhysics>
class WorksetBase : public SimplexPhysics
{
  protected:
    Plato::OrdinalType mNumCells;
    Plato::OrdinalType mNumNodes;

    using SimplexPhysics::mNumDofsPerNode;
    using SimplexPhysics::mNumControl;
    using SimplexPhysics::mNumNodesPerCell;
    using SimplexPhysics::mNumDofsPerCell;
    using SimplexPhysics::mNumLocalDofsPerCell;
    using SimplexPhysics::mNumNSPerNode;

    using StateFad      = typename Plato::SimplexFadTypes<SimplexPhysics>::StateFad;
    using LocalStateFad = typename Plato::SimplexFadTypes<SimplexPhysics>::LocalStateFad;
    using NodeStateFad  = typename Plato::SimplexFadTypes<SimplexPhysics>::NodeStateFad;
    using ControlFad    = typename Plato::SimplexFadTypes<SimplexPhysics>::ControlFad;
    using ConfigFad     = typename Plato::SimplexFadTypes<SimplexPhysics>::ConfigFad;

    static constexpr int SpaceDim = SimplexPhysics::mNumSpatialDims;
    static constexpr int mNumConfigDofsPerCell = SpaceDim*mNumNodesPerCell;

    Plato::VectorEntryOrdinal<SpaceDim,mNumDofsPerNode> mStateEntryOrdinal;
    Plato::VectorEntryOrdinal<SpaceDim,mNumNSPerNode>   mNodeStateEntryOrdinal;
    Plato::VectorEntryOrdinal<SpaceDim,mNumControl>     mControlEntryOrdinal;
    Plato::VectorEntryOrdinal<SpaceDim,SpaceDim>        mConfigEntryOrdinal;

    Plato::NodeCoordinate<SpaceDim>     mNodeCoordinate;

  public:
    /**************************************************************************/
    WorksetBase(Omega_h::Mesh& aMesh) :
            mNumCells(aMesh.nelems()),
            mNumNodes(aMesh.nverts()),
            mStateEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, mNumDofsPerNode>(&aMesh)),
            mNodeStateEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, mNumNSPerNode>(&aMesh)),
            mControlEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, mNumControl>(&aMesh)),
            mConfigEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, SpaceDim>(&aMesh)),
            mNodeCoordinate(Plato::NodeCoordinate<SpaceDim>(&aMesh))
    {
    }
    /**************************************************************************/

    /**************************************************************************/
    void worksetControl( const Plato::ScalarVectorT<Plato::Scalar> & aControl,
                         Plato::ScalarMultiVectorT<Plato::Scalar> & aControlWS ) const
    /**************************************************************************/
    {
      Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
              mNumCells, mControlEntryOrdinal, aControl, aControlWS);
    }

    /**************************************************************************/
    void worksetControl( const Plato::ScalarVectorT<Plato::Scalar> & aControl,
                         Plato::ScalarMultiVectorT<ControlFad> & aFadControlWS ) const
    /**************************************************************************/
    {
      Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
              mNumCells, mControlEntryOrdinal, aControl, aFadControlWS);
    }

    /**************************************************************************/
    void worksetState( const Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace> & aState,
                       Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace> & aStateWS ) const
    /**************************************************************************/
    {
      Plato::workset_state_scalar_scalar<mNumDofsPerNode, mNumNodesPerCell>(
              mNumCells, mStateEntryOrdinal, aState, aStateWS);
    }

    /**************************************************************************/
    void worksetState( const Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace> & aState,
                       Kokkos::View<StateFad**, Kokkos::LayoutRight, Plato::MemSpace> & aFadStateWS ) const
    /**************************************************************************/
    {
      Plato::workset_state_scalar_fad<mNumDofsPerNode, mNumNodesPerCell, StateFad>(
              mNumCells, mStateEntryOrdinal, aState, aFadStateWS);
    }

    /**************************************************************************/
    void worksetLocalState( const Plato::ScalarVectorT<Plato::Scalar> & aLocalState,
                                  Plato::ScalarMultiVectorT<Plato::Scalar> & aLocalStateWS ) const
    /**************************************************************************/
    {
      printf("%d %d %d %d\n", aLocalState.size(), aLocalStateWS.extent(0), aLocalStateWS.extent(1), mNumLocalDofsPerCell);
      Plato::workset_local_state_scalar_scalar<mNumLocalDofsPerCell>(
              mNumCells, aLocalState, aLocalStateWS);
    }

    /**************************************************************************/
    void worksetLocalState( const Plato::ScalarVectorT<Plato::Scalar> & aLocalState,
                            Plato::ScalarMultiVectorT<LocalStateFad>  & aFadLocalStateWS ) const
    /**************************************************************************/
    {
      printf("%d %d %d %d\n", aLocalState.size(), aFadLocalStateWS.extent(0), aFadLocalStateWS.extent(1), mNumLocalDofsPerCell);
      Plato::workset_local_state_scalar_fad<mNumLocalDofsPerCell, LocalStateFad>(
              mNumCells, aLocalState, aFadLocalStateWS);
    }

    /**************************************************************************/
    void worksetNodeState( const Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace> & aState,
                           Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace> & aNodeStateWS ) const
    /**************************************************************************/
    {
      Plato::workset_state_scalar_scalar<mNumNSPerNode, mNumNodesPerCell>(
              mNumCells, mNodeStateEntryOrdinal, aState, aNodeStateWS);
    }

    /**************************************************************************/
    void worksetNodeState( const Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace> & aState,
                           Kokkos::View<NodeStateFad**, Kokkos::LayoutRight, Plato::MemSpace> & aFadStateWS ) const
    /**************************************************************************/
    {
      Plato::workset_state_scalar_fad<mNumNSPerNode, mNumNodesPerCell, NodeStateFad>(
              mNumCells, mNodeStateEntryOrdinal, aState, aFadStateWS);
    }
    
    /**************************************************************************/
    void worksetConfig(Plato::ScalarArray3DT<Plato::Scalar> & aConfigWS) const
    /**************************************************************************/
    {
      Plato::workset_config_scalar<SpaceDim, mNumNodesPerCell>(
              mNumCells, mNodeCoordinate, aConfigWS);
    }

    /**************************************************************************/
    void worksetConfig(Plato::ScalarArray3DT<ConfigFad> & aFadConfigWS) const
    /**************************************************************************/
    {
      Plato::workset_config_fad<SpaceDim, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigFad>(
              mNumCells, mNodeCoordinate, aFadConfigWS);
    }
    
    /**************************************************************************/
    template<class ResidualWorksetType, class AssembledResidualType>
    void assembleResidual(const ResidualWorksetType & aResidualWorkset, AssembledResidualType & aReturnValue) const
    /**************************************************************************/
    {
        Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>(
                mNumCells, WorksetBase<SimplexPhysics>::mStateEntryOrdinal, aResidualWorkset, aReturnValue);
    }

    /**************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void assembleJacobian(int aNumRows, 
                          int aNumColumns,
                          const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
                          const JacobianWorksetType & aJacobianWorkset,
                          AssembledJacobianType & aReturnValue) const
    /**************************************************************************/
    {
        Plato::assemble_jacobian(mNumCells, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /**************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void assembleTransposeJacobian(int aNumRows,
                                   int aNumColumns,
                                   const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
                                   const JacobianWorksetType & aJacobianWorkset,
                                   AssembledJacobianType & aReturnValue) const
    /**************************************************************************/
    {
        Plato::assemble_transpose_jacobian(mNumCells, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

};
// class WorksetBase

}//namespace Plato

#endif
