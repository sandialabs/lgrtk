#ifndef APPLY_CONSTRAINTS_HPP
#define APPLY_CONSTRAINTS_HPP

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<int NumDofPerNode>
 void applyBlockConstraints(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                            Plato::ScalarVector aRhs,
                            Plato::LocalOrdinalVector aDirichletDofs,
                            Plato::ScalarVector aDirichletValues
  )
/******************************************************************************/
{
  
  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
    Plato::OrdinalType tNumBCs = aDirichletDofs.size();
    auto tRowMap        = aMatrix->rowMap();
    auto tColumnIndices = aMatrix->columnIndices();
    ScalarVector tMatrixEntries = aMatrix->entries();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBCs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aBcOrdinal)
    {
        OrdinalType tRowDofOrdinal = aDirichletDofs[aBcOrdinal];
        Scalar tValue = aDirichletValues[aBcOrdinal];
        auto tRowNodeOrdinal = tRowDofOrdinal / NumDofPerNode;
        auto tLocalRowDofOrdinal  = tRowDofOrdinal % NumDofPerNode;
        RowMapEntryType tRowStart = tRowMap(tRowNodeOrdinal  );
        RowMapEntryType tRowEnd   = tRowMap(tRowNodeOrdinal+1);
        for (RowMapEntryType tColumnNodeOffset=tRowStart; tColumnNodeOffset<tRowEnd; tColumnNodeOffset++)
        {
            for (RowMapEntryType tLocalColumnDofOrdinal=0; tLocalColumnDofOrdinal<NumDofPerNode; tLocalColumnDofOrdinal++)
            {
                OrdinalType tColumnNodeOrdinal = tColumnIndices(tColumnNodeOffset);
                auto tEntryOrdinal = NumDofPerNode*NumDofPerNode*tColumnNodeOffset
                        + NumDofPerNode*tLocalRowDofOrdinal + tLocalColumnDofOrdinal;
                auto tColumnDofOrdinal = NumDofPerNode*tColumnNodeOrdinal+tLocalColumnDofOrdinal;
                if (tColumnDofOrdinal == tRowDofOrdinal) // diagonal
                {
                    tMatrixEntries(tEntryOrdinal) = 1.0;
                }
                else
                {
                    // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
                    // to maintain symmetry
                    Kokkos::atomic_add(&aRhs(tColumnDofOrdinal), -tMatrixEntries(tEntryOrdinal)*tValue);
                    tMatrixEntries(tEntryOrdinal) = 0.0;
                    RowMapEntryType tColRowStart = tRowMap(tColumnNodeOrdinal  );
                    RowMapEntryType tColRowEnd   = tRowMap(tColumnNodeOrdinal+1);
                    for (RowMapEntryType tColRowNodeOffset=tColRowStart; tColRowNodeOffset<tColRowEnd; tColRowNodeOffset++)
                    {
                        OrdinalType tColRowNodeOrdinal = tColumnIndices(tColRowNodeOffset);
                        auto tColRowEntryOrdinal = NumDofPerNode*NumDofPerNode*tColRowNodeOffset
                                +NumDofPerNode*tLocalColumnDofOrdinal+tLocalRowDofOrdinal;
                        auto tColRowDofOrdinal = NumDofPerNode*tColRowNodeOrdinal+tLocalRowDofOrdinal;
                        if (tColRowDofOrdinal == tRowDofOrdinal)
                        {
                            // this is the (col, row) entry -- clear it, too
                            tMatrixEntries(tColRowEntryOrdinal) = 0.0;
                        }
                    }
                }
            }
        }
    },"Dirichlet BC imposition - First loop");
  
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumBCs), LAMBDA_EXPRESSION(int bcOrdinal){
        OrdinalType tDofOrdinal = aDirichletDofs[bcOrdinal];
        Scalar tValue = aDirichletValues[bcOrdinal];
        aRhs(tDofOrdinal) = tValue;
    },"Dirichlet BC imposition - Second loop");

}

/******************************************************************************/
template<int NumDofPerNode> void
applyConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> matrix,
    Plato::ScalarVector                rhs,
    Plato::LocalOrdinalVector          bcDofs,
    Plato::ScalarVector                bcValues
  )
/******************************************************************************/
{
  
  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
  int numBCs = bcDofs.size();
  auto rowMap        = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();
  ScalarVector matrixEntries = matrix->entries();
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numBCs), LAMBDA_EXPRESSION(int bcOrdinal)
  {
    OrdinalType nodeNumber = bcDofs[bcOrdinal];
    Scalar value = bcValues[bcOrdinal];
    RowMapEntryType rowStart = rowMap(nodeNumber  );
    RowMapEntryType rowEnd   = rowMap(nodeNumber+1);
    for (RowMapEntryType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
    {
      OrdinalType column = columnIndices(entryOrdinal);
      if (column == nodeNumber) // diagonal
      {
        matrixEntries(entryOrdinal) = 1.0;
      }
      else
      {
        // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
        // to maintain symmetry
        Kokkos::atomic_add(&rhs(column), -matrixEntries(entryOrdinal)*value);
        matrixEntries(entryOrdinal) = 0.0;
        RowMapEntryType colRowStart = rowMap(column  );
        RowMapEntryType colRowEnd   = rowMap(column+1);
        for (RowMapEntryType colRowEntryOrdinal=colRowStart; colRowEntryOrdinal<colRowEnd; colRowEntryOrdinal++)
        {
          OrdinalType colRowColumn = columnIndices(colRowEntryOrdinal);
          if (colRowColumn == nodeNumber)
          {
            // this is the (col, row) entry -- clear it, too
            matrixEntries(colRowEntryOrdinal) = 0.0;
          }
        }
      }
    }
  },"BC imposition");
  
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numBCs), LAMBDA_EXPRESSION(int bcOrdinal)
  {
    OrdinalType nodeNumber = bcDofs[bcOrdinal];
    Scalar value = bcValues[bcOrdinal];
    rhs(nodeNumber) = value;
  },"BC imposition");

}

} // namespace Plato


#endif
