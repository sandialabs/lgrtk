#ifndef WORKSET_BASE_HPP
#define WORKSET_BASE_HPP

#include <cassert>

#include <Omega_h_mesh.hpp>

#include "ImplicitFunctors.hpp"
#include "plato/SimplexFadTypes.hpp"

#ifdef NDEBUG
#error "We need debug on"
#endif

namespace Plato
{

template <class Scalar, class Result>
inline Scalar local_result_sum(const Plato::OrdinalType& aNumCells, Result aResult) {
  Scalar tReturnVal(0.0);
  Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal, Scalar& aLocalResult)
  {
    aLocalResult += aResult(aCellOrdinal);
  }, tReturnVal);
  return tReturnVal;
}

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

/******************************************************************************/
template<int numDofsPerNode, int numNodesPerCell, int numDofsPerCell, class StateFad, class StateEntryOrdinal, class State, class FadStateWS>
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
                aFadStateWS(aCellOrdinal,tLocalDof) = StateFad(numDofsPerCell, tLocalDof, aState(tEntryOrdinal));
            }
        }
    }, "workset_state_scalar_fad");
}

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

}//namespace Plato

/******************************************************************************/
/*! Base class for workset functionality.
*/
/******************************************************************************/
template<typename SimplexPhysics>
class WorksetBase : public SimplexPhysics
{
  protected:
    Plato::OrdinalType m_numCells;
    Plato::OrdinalType m_numNodes;

    using SimplexPhysics::m_numDofsPerNode;
    using SimplexPhysics::m_numControl;
    using SimplexPhysics::m_numNodesPerCell;
    using SimplexPhysics::m_numDofsPerCell;

    using StateFad   = typename Plato::SimplexFadTypes<SimplexPhysics>::StateFad;
    using ControlFad = typename Plato::SimplexFadTypes<SimplexPhysics>::ControlFad;
    using ConfigFad  = typename Plato::SimplexFadTypes<SimplexPhysics>::ConfigFad;

    static constexpr int SpaceDim = SimplexPhysics::m_numSpatialDims;
    static constexpr int m_numConfigDofsPerCell = SpaceDim*m_numNodesPerCell;

    Plato::VectorEntryOrdinal<SpaceDim,m_numDofsPerNode> m_stateEntryOrdinal;
    Plato::VectorEntryOrdinal<SpaceDim,m_numControl>     m_controlEntryOrdinal;
    Plato::VectorEntryOrdinal<SpaceDim,SpaceDim>         m_configEntryOrdinal;

    Plato::NodeCoordinate<SpaceDim>     m_nodeCoordinate;

  public:
    /**************************************************************************/
    WorksetBase(Omega_h::Mesh& aMesh) :
            m_numCells(aMesh.nelems()),
            m_numNodes(aMesh.nverts()),
            m_stateEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, m_numDofsPerNode>(&aMesh)),
            m_controlEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, m_numControl>(&aMesh)),
            m_configEntryOrdinal(Plato::VectorEntryOrdinal<SpaceDim, SpaceDim>(&aMesh)),
            m_nodeCoordinate(Plato::NodeCoordinate<SpaceDim>(&aMesh))
    {
    }
    /**************************************************************************/

    /**************************************************************************/
    void worksetControl( const Plato::ScalarVectorT<Plato::Scalar> & aControl,
                         Plato::ScalarMultiVectorT<Plato::Scalar> & aControlWS ) const
    /**************************************************************************/
    {
      Plato::workset_control_scalar_scalar<m_numNodesPerCell>(
              m_numCells, m_controlEntryOrdinal, aControl, aControlWS);
    }

    /**************************************************************************/
    void worksetControl( const Plato::ScalarVectorT<Plato::Scalar> & aControl,
                         Plato::ScalarMultiVectorT<ControlFad> & aFadControlWS ) const
    /**************************************************************************/
    {
      Plato::workset_control_scalar_fad<m_numNodesPerCell, ControlFad>(
              m_numCells, m_controlEntryOrdinal, aControl, aFadControlWS);
    }

    /**************************************************************************/
    void worksetState( const Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace> & aState,
                       Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace> & aStateWS ) const
    /**************************************************************************/
    {
      Plato::workset_state_scalar_scalar<m_numDofsPerNode, m_numNodesPerCell>(
              m_numCells, m_stateEntryOrdinal, aState, aStateWS);
    }

    /**************************************************************************/
    void worksetState( const Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace> & aState,
                       Kokkos::View<StateFad**, Kokkos::LayoutRight, Plato::MemSpace> & aFadStateWS ) const
    /**************************************************************************/
    {
      Plato::workset_state_scalar_fad<m_numDofsPerNode, m_numNodesPerCell, m_numDofsPerCell, StateFad>(
              m_numCells, m_stateEntryOrdinal, aState, aFadStateWS);
    }
    
    /**************************************************************************/
    void worksetConfig(Plato::ScalarArray3DT<Plato::Scalar> & aConfigWS) const
    /**************************************************************************/
    {
      Plato::workset_config_scalar<SpaceDim, m_numNodesPerCell>(
              m_numCells, m_nodeCoordinate, aConfigWS);
    }

    /**************************************************************************/
    void worksetConfig(Plato::ScalarArray3DT<ConfigFad> & aFadConfigWS) const
    /**************************************************************************/
    {
      Plato::workset_config_fad<SpaceDim, m_numNodesPerCell, m_numConfigDofsPerCell, ConfigFad>(
              m_numCells, m_nodeCoordinate, aFadConfigWS);
    }
    
    /**************************************************************************/
    template<class ResidualWorksetType, class AssembledResidualType>
    void assembleResidual(const ResidualWorksetType & aResidualWorkset, AssembledResidualType & aReturnValue) const
    /**************************************************************************/
    {
        Plato::assemble_residual<m_numNodesPerCell, m_numDofsPerNode>(
                m_numCells, WorksetBase<SimplexPhysics>::m_stateEntryOrdinal, aResidualWorkset, aReturnValue);
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
        Plato::assemble_jacobian(m_numCells, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
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
        Plato::assemble_transpose_jacobian(m_numCells, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

};

#endif
