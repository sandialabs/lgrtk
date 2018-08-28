#ifndef LGR_PLATO_IMPLICIT_FUNCTORS_HPP
#define LGR_PLATO_IMPLICIT_FUNCTORS_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_vector.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <CrsLinearProblem.hpp>
#include <CrsMatrix.hpp>
#include <Fields.hpp>
#include <ParallelComm.hpp>

#include "ErrorHandling.hpp"

#include "plato/SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************/
  /*! Create a 1D Kokkos::View
  
    @param name Arbitrary descriptive name.
    @param entryCount Number of elements in the returned Kokkos::View.
  */
template<class T>
Omega_h::Write<T> getArray_Omega_h(std::string name, Plato::OrdinalType entryCount)
/******************************************************************************/
{
  Kokkos::View<T*, Kokkos::LayoutRight, Plato::MemSpace> view(name, entryCount);
  return Omega_h::Write<T>(view);
} 

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode=1>
class VectorEntryOrdinal
{
  public:
    const Omega_h::LOs m_cells2nodes;

  public:
    VectorEntryOrdinal(
      Omega_h::Mesh* mesh ) : 
      m_cells2nodes(mesh->ask_elem_verts()) {}

    DEVICE_TYPE inline Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal=0) const
    {
        Plato::OrdinalType vertexNumber = m_cells2nodes[cellOrdinal*(SpaceDim+1) + nodeOrdinal];
        return vertexNumber * DofsPerNode + dofOrdinal;
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class NodeCoordinate
{
  private:
    const Omega_h::LOs m_cells2nodes;
    const Omega_h::Reals m_coords;

  public:
    NodeCoordinate(
      Omega_h::Mesh* mesh ) : 
      m_cells2nodes(mesh->ask_elem_verts()),
      m_coords(mesh->coords()) { }

    DEVICE_TYPE
    inline
    Plato::Scalar
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal) const
    {
        Plato::OrdinalType vertexNumber = m_cells2nodes[cellOrdinal*(SpaceDim+1) + nodeOrdinal];
        Plato::Scalar coord = m_coords[vertexNumber * SpaceDim + dofOrdinal];
        return coord;
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class JacobianDet
{
  private:
    const NodeCoordinate<SpaceDim> m_nodeCoordinate;

  public:
    JacobianDet( Omega_h::Mesh* mesh ) : 
      m_nodeCoordinate(mesh) {}

    DEVICE_TYPE
    inline
    Plato::Scalar
    operator()(Plato::OrdinalType cellOrdinal) const {
      Omega_h::Matrix<SpaceDim, SpaceDim> jacobian;

      for (Plato::OrdinalType d1=0; d1<SpaceDim; d1++) {
        for (Plato::OrdinalType d2=0; d2<SpaceDim; d2++) {
          jacobian[d1][d2] = m_nodeCoordinate(cellOrdinal,d2,d1) - m_nodeCoordinate(cellOrdinal,SpaceDim,d1);
        }
      }
      return Omega_h::determinant(jacobian);
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class SideNodeCoordinate
{
  private:
    const Omega_h::LOs m_sides2nodes;
    const Omega_h::Reals m_coords;

  public:
    SideNodeCoordinate(
      Omega_h::Mesh* mesh ) : 
      m_sides2nodes(mesh->ask_verts_of(SpaceDim-1)),
      m_coords(mesh->coords()) { }

    DEVICE_TYPE
    inline
    Plato::Scalar
    operator()(Plato::OrdinalType sideOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal) const
    {
        Plato::OrdinalType vertexNumber = m_sides2nodes[sideOrdinal*SpaceDim + nodeOrdinal];
        Plato::Scalar coord = m_coords[vertexNumber*SpaceDim + dofOrdinal];
        return coord;
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeGradientWorkset
{
  private:
    Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;

  public:
    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarArray3DT<ScalarType> aGradients,
               Plato::ScalarArray3DT<ScalarType> aConfig,
               Plato::ScalarVectorT<ScalarType> aCellVolume) const;
};

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeGradientWorkset<3>::operator()(Plato::OrdinalType aCellOrdinal,
                                        Plato::ScalarArray3DT<ScalarType> aGradients,
                                        Plato::ScalarArray3DT<ScalarType> aConfig,
                                        Plato::ScalarVectorT<ScalarType> aCellVolume) const
  {
    ScalarType j11=aConfig(aCellOrdinal,0,0)-aConfig(aCellOrdinal,3,0);
    ScalarType j12=aConfig(aCellOrdinal,1,0)-aConfig(aCellOrdinal,3,0);
    ScalarType j13=aConfig(aCellOrdinal,2,0)-aConfig(aCellOrdinal,3,0);

    ScalarType j21=aConfig(aCellOrdinal,0,1)-aConfig(aCellOrdinal,3,1);
    ScalarType j22=aConfig(aCellOrdinal,1,1)-aConfig(aCellOrdinal,3,1);
    ScalarType j23=aConfig(aCellOrdinal,2,1)-aConfig(aCellOrdinal,3,1);

    ScalarType j31=aConfig(aCellOrdinal,0,2)-aConfig(aCellOrdinal,3,2);
    ScalarType j32=aConfig(aCellOrdinal,1,2)-aConfig(aCellOrdinal,3,2);
    ScalarType j33=aConfig(aCellOrdinal,2,2)-aConfig(aCellOrdinal,3,2);

    ScalarType detj = j11*j22*j33+j12*j23*j31+j13*j21*j32
                     -j11*j23*j32-j12*j21*j33-j13*j22*j31;

    ScalarType i11 = (j22*j33-j23*j32)/detj;
    ScalarType i12 = (j13*j32-j12*j33)/detj;
    ScalarType i13 = (j12*j23-j13*j22)/detj;

    ScalarType i21 = (j23*j31-j21*j33)/detj;
    ScalarType i22 = (j11*j33-j13*j31)/detj;
    ScalarType i23 = (j13*j21-j11*j23)/detj;

    ScalarType i31 = (j21*j32-j22*j31)/detj;
    ScalarType i32 = (j12*j31-j11*j32)/detj;
    ScalarType i33 = (j11*j22-j12*j21)/detj;

    aCellVolume(aCellOrdinal) = fabs(detj);

    aGradients(aCellOrdinal,0,0) = i11;
    aGradients(aCellOrdinal,0,1) = i12;
    aGradients(aCellOrdinal,0,2) = i13;

    aGradients(aCellOrdinal,1,0) = i21;
    aGradients(aCellOrdinal,1,1) = i22;
    aGradients(aCellOrdinal,1,2) = i23;

    aGradients(aCellOrdinal,2,0) = i31;
    aGradients(aCellOrdinal,2,1) = i32;
    aGradients(aCellOrdinal,2,2) = i33;

    aGradients(aCellOrdinal,3,0) = -(i11+i21+i31);
    aGradients(aCellOrdinal,3,1) = -(i12+i22+i32);
    aGradients(aCellOrdinal,3,2) = -(i13+i23+i33);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeGradientWorkset<2>::operator()(Plato::OrdinalType cellOrdinal,
                                        Plato::ScalarArray3DT<ScalarType> gradients,
                                        Plato::ScalarArray3DT<ScalarType> config,
                                        Plato::ScalarVectorT<ScalarType> cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,2,0);
    ScalarType j12=config(cellOrdinal,1,0)-config(cellOrdinal,2,0);

    ScalarType j21=config(cellOrdinal,0,1)-config(cellOrdinal,2,1);
    ScalarType j22=config(cellOrdinal,1,1)-config(cellOrdinal,2,1);

    ScalarType detj = j11*j22-j12*j21;

    ScalarType i11 = j22/detj;
    ScalarType i12 =-j12/detj;

    ScalarType i21 =-j21/detj;
    ScalarType i22 = j11/detj;

    cellVolume(cellOrdinal) = fabs(detj);

    gradients(cellOrdinal,0,0) = i11;
    gradients(cellOrdinal,0,1) = i12;

    gradients(cellOrdinal,1,0) = i21;
    gradients(cellOrdinal,1,1) = i22;

    gradients(cellOrdinal,2,0) = -(i11+i21);
    gradients(cellOrdinal,2,1) = -(i12+i22);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeGradientWorkset<1>::operator()(Plato::OrdinalType cellOrdinal,
                                        Plato::ScalarArray3DT<ScalarType> gradients,
                                        Plato::ScalarArray3DT<ScalarType> config,
                                        Plato::ScalarVectorT<ScalarType> cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,1,0);

    ScalarType detj = j11;

    ScalarType i11 = 1.0/detj;

    cellVolume(cellOrdinal) = fabs(detj);

    gradients(cellOrdinal,0,0) = i11;
    gradients(cellOrdinal,1,0) =-i11;
  }

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeCellVolume
{
  private:
    Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;

  public:
    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType cellOrdinal,
               Plato::ScalarArray3DT<ScalarType> config,
               ScalarType& cellVolume) const;
};

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeCellVolume<3>::operator()(Plato::OrdinalType cellOrdinal,
                                   Plato::ScalarArray3DT<ScalarType> config,
                                   ScalarType& cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,3,0);
    ScalarType j12=config(cellOrdinal,1,0)-config(cellOrdinal,3,0);
    ScalarType j13=config(cellOrdinal,2,0)-config(cellOrdinal,3,0);

    ScalarType j21=config(cellOrdinal,0,1)-config(cellOrdinal,3,1);
    ScalarType j22=config(cellOrdinal,1,1)-config(cellOrdinal,3,1);
    ScalarType j23=config(cellOrdinal,2,1)-config(cellOrdinal,3,1);

    ScalarType j31=config(cellOrdinal,0,2)-config(cellOrdinal,3,2);
    ScalarType j32=config(cellOrdinal,1,2)-config(cellOrdinal,3,2);
    ScalarType j33=config(cellOrdinal,2,2)-config(cellOrdinal,3,2);

    ScalarType detj = j11*j22*j33+j12*j23*j31+j13*j21*j32
                     -j11*j23*j32-j12*j21*j33-j13*j22*j31;

    cellVolume = fabs(detj);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeCellVolume<2>::operator()(Plato::OrdinalType cellOrdinal,
                                   Plato::ScalarArray3DT<ScalarType> config,
                                   ScalarType& cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,2,0);
    ScalarType j12=config(cellOrdinal,1,0)-config(cellOrdinal,2,0);

    ScalarType j21=config(cellOrdinal,0,1)-config(cellOrdinal,2,1);
    ScalarType j22=config(cellOrdinal,1,1)-config(cellOrdinal,2,1);

    ScalarType detj = j11*j22-j12*j21;

    cellVolume = fabs(detj);
  }

  template<>
  template<typename ScalarType>
  DEVICE_TYPE inline void
  ComputeCellVolume<1>::operator()(Plato::OrdinalType cellOrdinal,
                                   Plato::ScalarArray3DT<ScalarType> config,
                                   ScalarType& cellVolume) const
  {
    ScalarType j11=config(cellOrdinal,0,0)-config(cellOrdinal,1,0);

    ScalarType detj = j11;

    cellVolume = fabs(detj);
  }


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeGradient
{
  private:
    const NodeCoordinate<SpaceDim> m_nodeCoordinate;

  public:
    ComputeGradient( 
      NodeCoordinate<SpaceDim> nodeCoordinate) :
      m_nodeCoordinate(nodeCoordinate) {}

    DEVICE_TYPE
    inline
    void
    operator()(Plato::OrdinalType cellOrdinal,
               Omega_h::Vector<SpaceDim>* gradients,
               Scalar& cellVolume) const
    {
      // compute jacobian/Det/inverse for cell:
      //
      Omega_h::Matrix<SpaceDim, SpaceDim> jacobian, jacobianInverse;
      for (Plato::OrdinalType d1=0; d1<SpaceDim; d1++)
      {
        for (Plato::OrdinalType d2=0; d2<SpaceDim; d2++)
        {
          jacobian[d1][d2] = m_nodeCoordinate(cellOrdinal,d2,d1) - m_nodeCoordinate(cellOrdinal,SpaceDim,d1);
        } 
      } 
      Plato::Scalar jacobianDet = Omega_h::determinant(jacobian);
      jacobianInverse = Omega_h::invert(jacobian);
      cellVolume = fabs(jacobianDet);

      // ref gradients in 3D are:
      //    field 0 = ( 1 ,0, 0)
      //    field 1 = ( 0, 1, 0)
      //    field 2 = ( 0, 0, 1)
      //    field 3 = (-1,-1,-1)
      
      // Therefore, when we multiply by the transpose jacobian inverse (which is what we do to compute physical gradients),
      // we have the following values:
      //    field 0 = row 0 of jacobianInv
      //    field 1 = row 1 of jacobianInv
      //    field 2 = row 2 of jacobianInv
      //    field 3 = negative sum of the three rows
      
      for (Plato::OrdinalType d=0; d<SpaceDim; d++)
      {
        gradients[SpaceDim][d] = 0.0;
      } 
      
      for (Plato::OrdinalType nodeOrdinal=0; nodeOrdinal<SpaceDim; nodeOrdinal++)  // "d1" for jacobian
      {
        for (Plato::OrdinalType d=0; d<SpaceDim; d++) // "d2" for jacobian
        {
          gradients[nodeOrdinal][d] = jacobianInverse[nodeOrdinal][d];
          gradients[SpaceDim][d]   -= jacobianInverse[nodeOrdinal][d];
        } 
      } 
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeGradientMatrix : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;
    static constexpr auto m_numSpaceDim = SpaceDim;

  public:

    DEVICE_TYPE
    void
    operator()( const Omega_h::Vector<m_numSpaceDim>* gradients,
                      Omega_h::Vector<m_numVoigtTerms>* gradientMatrix) const
    {
      for (Plato::OrdinalType iDof=0; iDof<m_numDofsPerCell; iDof++){
        for (Plato::OrdinalType iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
          gradientMatrix[iDof][iVoigt] = 0.0;
        }
      }

      for (Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++)
      {
        Plato::OrdinalType voigtTerm=0;
        for (Plato::OrdinalType iDof=0; iDof<m_numSpaceDim; iDof++){
          gradientMatrix[m_numSpaceDim*iNode+iDof][voigtTerm] = gradients[iNode][iDof];
          voigtTerm++;
        }
    
        for (Plato::OrdinalType jDof=m_numSpaceDim-1; jDof>=1; jDof--){
          for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
            gradientMatrix[m_numSpaceDim*iNode+iDof][voigtTerm] = gradients[iNode][jDof];
            gradientMatrix[m_numSpaceDim*iNode+jDof][voigtTerm] = gradients[iNode][iDof];
            voigtTerm++;
          }
        }
      }
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim, typename OrdinalLookupType>
class Assemble
{
  private:
    static constexpr auto m_numVoigtTerms   = (SpaceDim == 3) ? 6 : 
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr auto m_numNodesPerCell = SpaceDim+1;
    static constexpr auto m_numDofsPerCell  = SpaceDim*m_numNodesPerCell;
    
    const typename CrsMatrixType::ScalarVector m_matrixEntries;
    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    const OrdinalLookupType m_entryOrdinalLookup;
    const Plato::OrdinalType m_entriesLength;

  public:
    Assemble(const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> aCellStiffness,
             Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
             OrdinalLookupType aEntryOrdinalLookup ) :
        m_matrixEntries(aMatrix->entries()),
        m_cellStiffness(aCellStiffness),
        m_entryOrdinalLookup(aEntryOrdinalLookup),
        m_entriesLength(m_matrixEntries.size()) {}

    DEVICE_TYPE
    inline
    void
    operator()(Plato::OrdinalType cellOrdinal,
               const Omega_h::Vector<m_numVoigtTerms>* gradientMatrix,
               const Plato::Scalar& cellVolume) const
    {
      for (Plato::OrdinalType iDof=0; iDof<m_numDofsPerCell; iDof++)
      {
        for (Plato::OrdinalType jDof=0; jDof<m_numDofsPerCell; jDof++)
        {
            Plato::Scalar integral = (gradientMatrix[iDof] * (m_cellStiffness * gradientMatrix[jDof])) * cellVolume;
          
            auto entryOrdinal = m_entryOrdinalLookup(cellOrdinal,iDof,jDof);
            if (entryOrdinal < m_entriesLength)
            {
                Kokkos::atomic_add(&m_matrixEntries(entryOrdinal), integral);
            }
        }
      }
    }
};
/******************************************************************************/


/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
class BlockMatrixEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVector m_rowMap;
    const typename CrsMatrixType::OrdinalVector m_columnIndices;
    const Omega_h::LOs m_cells2nodes;
    
  public:
    BlockMatrixEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Omega_h::Mesh* mesh ) :
      m_rowMap(matrix->rowMap()), 
      m_columnIndices(matrix->columnIndices()), 
      m_cells2nodes(mesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = m_cells2nodes[cellOrdinal * (SpaceDim+1) + iNode];
        Plato::OrdinalType jLocalOrdinal = m_cells2nodes[cellOrdinal * (SpaceDim+1) + jNode];
        Plato::RowMapEntryType rowStart = m_rowMap(iLocalOrdinal);
        Plato::RowMapEntryType rowEnd   = m_rowMap(iLocalOrdinal+1);
        for (Plato::RowMapEntryType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (m_columnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode_I*DofsPerNode_J+iDof*DofsPerNode_J+jDof;
          }
        }
        return Plato::RowMapEntryType(-1);
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofsPerNode_J=DofsPerNode>
class MatrixEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVector m_rowMap;
    const typename CrsMatrixType::OrdinalVector m_columnIndices;
    const Omega_h::LOs m_cells2nodes;
    
  public:
    MatrixEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                       Omega_h::Mesh* aMesh ) :
      m_rowMap(aMatrix->rowMap()),
      m_columnIndices(aMatrix->columnIndices()),
      m_cells2nodes(aMesh->ask_elem_verts()) { }

    DEVICE_TYPE
    inline
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
      auto iNode = icellDof / DofsPerNode;
      auto iDof  = icellDof % DofsPerNode;
      auto jNode = jcellDof / DofsPerNode;
      auto jDof  = jcellDof % DofsPerNode;
      Plato::OrdinalType iLocalOrdinal = m_cells2nodes[cellOrdinal * (SpaceDim+1) + iNode];
      Plato::OrdinalType jLocalOrdinal = m_cells2nodes[cellOrdinal * (SpaceDim+1) + jNode];
      Plato::RowMapEntryType rowStart = m_rowMap(DofsPerNode*iLocalOrdinal+iDof  );
      Plato::RowMapEntryType rowEnd   = m_rowMap(DofsPerNode*iLocalOrdinal+iDof+1);
      for (Plato::RowMapEntryType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
      {
        if (m_columnIndices(entryOrdinal) == DofsPerNode*jLocalOrdinal+jDof)
        {
          return entryOrdinal;
        }
      }
      return Plato::RowMapEntryType(-1);
    }
};
/******************************************************************************/


/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param mesh Omega_h mesh on which the matrix is based.  

  Create a block matrix from connectivity in mesh with block size 
  DofsPerNode_I X DofsPerNode_J.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
Teuchos::RCP<MatrixType> 
CreateBlockMatrix( Omega_h::Mesh* mesh )
/******************************************************************************/
{
    const Plato::OrdinalType vertexDim = 0;
    Omega_h::Graph nodeNodeGraph = mesh->ask_star(vertexDim);
    
    auto rowMapOmega_h        = nodeNodeGraph.a2ab;
    auto columnIndicesOmega_h = nodeNodeGraph.ab2b;
    
    auto numRows = rowMapOmega_h.size() - 1;
    // Omega_h does not include the diagonals: add numRows, and then 
    // add 1 to each rowMap entry after the first
    auto nnz = columnIndicesOmega_h.size() + numRows; 

    // account for num dofs per node
    constexpr Plato::OrdinalType numBlockDofs = DofsPerNode_I*DofsPerNode_J;
  
    typename MatrixType::RowMapVector  rowMap("row map", numRows+1);
    typename MatrixType::ScalarVector  entries("matrix entries", nnz*numBlockDofs);
    typename MatrixType::OrdinalVector columnIndices("column indices", nnz);

    // The compressed row storage format in Omega_h doesn't include diagonals.  This 
    // function creates a CRSMatrix with diagonal entries included.
  
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numRows), LAMBDA_EXPRESSION(Plato::OrdinalType rowNumber)
    {
      auto entryOffset_oh = rowMapOmega_h[rowNumber];
      auto R0 = rowMapOmega_h[rowNumber] + rowNumber;
      auto R1 = rowMapOmega_h[rowNumber+1] + rowNumber+1;
      auto numNodesThisRow = R1-R0;
      rowMap(rowNumber) = R0;
      rowMap(rowNumber+1) = R1;
      
      Plato::OrdinalType i_oh = 0; // will track i until we insert the diagonal entry
      for (Plato::OrdinalType i=0; i<numNodesThisRow; i_oh++, i++)
      {
        bool insertDiagonal = false;
        if ((i_oh == i) && (i_oh + entryOffset_oh >= rowMapOmega_h[rowNumber+1]))
        {
          // i_oh == i                    --> have not inserted diagonal
          // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
          insertDiagonal = true;
        }
        else if (i_oh == i)
        {
          // i_oh + entryOffset_oh in bounds
          auto columnIndex = columnIndicesOmega_h[i_oh + entryOffset_oh];
          if (columnIndex > rowNumber)
          {
            insertDiagonal = true;
          }
        }
        if (insertDiagonal)
        {
          // store the diagonal entry
          columnIndices(R0+i) = rowNumber;
          i_oh--; // i_oh lags i by 1 after we hit the diagonal
        }
        else
        {
          columnIndices(R0+i) = columnIndicesOmega_h[i_oh + entryOffset_oh];
        }
      }
    });

    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries, DofsPerNode_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param mesh Omega_h mesh on which the matrix is based.  

  Create a matrix from connectivity in mesh with DofsPerNode.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode>
Teuchos::RCP<MatrixType> 
CreateMatrix( Omega_h::Mesh* mesh )
/******************************************************************************/
{
    const Plato::OrdinalType vertexDim = 0;
    Omega_h::Graph nodeNodeGraph = mesh->ask_star(vertexDim);
    
    auto rowMapOmega_h        = nodeNodeGraph.a2ab;
    auto columnIndicesOmega_h = nodeNodeGraph.ab2b;
    
    auto numRows = rowMapOmega_h.size() - 1;
    // Omega_h does not include the diagonals: add numRows, and then 
    // add 1 to each rowMap entry after the first
    auto nnz = columnIndicesOmega_h.size() + numRows; 

    // account for num dofs per node
    constexpr Plato::OrdinalType numDofsSquared = DofsPerNode*DofsPerNode;
  
    typename MatrixType::RowMapVector  rowMap("row map", numRows*DofsPerNode+1);
    typename MatrixType::ScalarVector  entries("matrix entries", nnz*numDofsSquared);
    typename MatrixType::OrdinalVector columnIndices("column indices", nnz*numDofsSquared);

    // The compressed row storage format in Omega_h doesn't include diagonals.  This 
    // function creates a CRSMatrix with diagonal entries included and expands the
    // graph to DofsPerNode.
  
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numRows), LAMBDA_EXPRESSION(Plato::OrdinalType rowNumber)
    {
      auto entryOffset_oh = rowMapOmega_h[rowNumber];
      auto R0 = rowMapOmega_h[rowNumber] + rowNumber;
      auto R1 = rowMapOmega_h[rowNumber+1] + rowNumber+1;
      auto numNodesThisRow = R1-R0;
      auto numDofsThisRow = numNodesThisRow*DofsPerNode;
      auto dofRowOffset = DofsPerNode*rowNumber;
      auto dofColOffset = numDofsSquared*R0;
      for (Plato::OrdinalType iDof=0; iDof<=DofsPerNode; iDof++){
        rowMap(dofRowOffset+iDof) = dofColOffset+iDof*numDofsThisRow;
      }
      
      Plato::OrdinalType i_oh = 0; // will track i until we insert the diagonal entry
      for (Plato::OrdinalType i=0; i<numNodesThisRow; i_oh++, i++)
      {
        bool insertDiagonal = false;
        if ((i_oh == i) && (i_oh + entryOffset_oh >= rowMapOmega_h[rowNumber+1]))
        {
          // i_oh == i                    --> have not inserted diagonal
          // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
          insertDiagonal = true;
        }
        else if (i_oh == i)
        {
          // i_oh + entryOffset_oh in bounds
          auto columnIndex = columnIndicesOmega_h[i_oh + entryOffset_oh];
          if (columnIndex > rowNumber)
          {
            insertDiagonal = true;
          }
        }
        if (insertDiagonal)
        {
          // store the diagonal entry
          for (Plato::OrdinalType iDof=0; iDof<DofsPerNode; iDof++){
            columnIndices(numDofsSquared*R0+DofsPerNode*i+iDof) = DofsPerNode*rowNumber+iDof;
          }
          i_oh--; // i_oh lags i by 1 after we hit the diagonal
        }
        else
        {
          for (Plato::OrdinalType iDof=0; iDof<DofsPerNode; iDof++){
            columnIndices(dofColOffset+DofsPerNode*i+iDof) = DofsPerNode*columnIndicesOmega_h[i_oh + entryOffset_oh]+iDof;
          }
        }
      }
      for (Plato::OrdinalType iDof=0; iDof<numDofsThisRow; iDof++)
      {
        for (Plato::OrdinalType jDof=1; jDof<DofsPerNode; jDof++){
          columnIndices(dofColOffset+jDof*numDofsThisRow+iDof) = columnIndices(dofColOffset+iDof);
        }
      }
    });

    auto retMatrix = Teuchos::rcp(new MatrixType( rowMap, columnIndices, entries ));
    return retMatrix;
}


} // end namespace lgr



#endif
