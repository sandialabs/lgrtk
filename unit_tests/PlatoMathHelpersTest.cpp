/*
 * PlatoMathHelpersTest.cpp
 *
 *  Created on: July 11, 2018
 */

#include <vector>

//#define COMPUTE_GOLD_
#ifdef COMPUTE_GOLD_
  #include <iostream>
  #include <fstream>
#endif

#include <assert.h>

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoMathFunctors.hpp"
#include "plato/Mechanics.hpp"
#include "plato/StabilizedMechanics.hpp"
#include "plato/PhysicsScalarFunction.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/VectorFunctionVMS.hpp"
#include "plato/ApplyProjection.hpp"
#include "plato/AnalyzeMacros.hpp"
#include "plato/HyperbolicTangentProjection.hpp"
#include "plato/alg/CrsMatrix.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include "KokkosKernels_SparseUtils.hpp"
#include <Kokkos_Concepts.hpp>
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include<KokkosKernels_IOUtils.hpp>

#include <Omega_h_mesh.hpp>

namespace PlatoUnitTests
{


using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

namespace PlatoDevel {

void getDataAsNonBlock(
      const Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
            Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
            Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
            Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues)
{
    const auto& tRowMap = aMatrix->rowMap();
    const auto& tColMap = aMatrix->columnIndices();
    const auto& tValues = aMatrix->entries();

    auto tNumMatrixRows = aMatrix->numRows();
    auto tNumBlockRows = aMatrix->blockSizeRow();
    auto tNumBlockCols = aMatrix->blockSizeCol();
    auto tBlockSize = tNumBlockRows*tNumBlockCols;

    // generate non block row map
    //
    aMatrixRowMap = Plato::ScalarVectorT<Plato::OrdinalType>("non block row map", tNumMatrixRows+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows+1), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumBlockRows;
        auto tLocalRowIndex = tMatrixRowIndex % tNumBlockRows;
        auto tFrom = tRowMap(tBlockRowIndex);
        auto tTo   = tRowMap(tBlockRowIndex+1);
        auto tBlockRowSize = tTo - tFrom;
        aMatrixRowMap(tMatrixRowIndex) = tFrom * tBlockSize + tLocalRowIndex * tBlockRowSize * tNumBlockRows;
    });

    // generate non block col map and non block values
    //
    auto tNumMatrixColEntries = tColMap.extent(0)*tBlockSize;
    aMatrixColMap = Plato::ScalarVectorT<Plato::OrdinalType>("non block col map", tNumMatrixColEntries);
    aMatrixValues = Plato::ScalarVectorT<Plato::Scalar>     ("non block values",  tNumMatrixColEntries);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumBlockRows;
        auto tLocalRowIndex = tMatrixRowIndex % tNumBlockRows;
        auto tFrom = tRowMap(tBlockRowIndex);
        auto tTo   = tRowMap(tBlockRowIndex+1);
        Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
        for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
        {
            for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumBlockCols; ++tBlockColOffset )
            {
                auto tMapIndex = tColMap(tColMapIndex)*tNumBlockCols+tBlockColOffset;
                auto tValIndex = tColMapIndex*tBlockSize+tLocalRowIndex*tNumBlockCols+tBlockColOffset;
                aMatrixColMap(tMatrixRowFrom) = tMapIndex;
                aMatrixValues(tMatrixRowFrom) = tValues(tValIndex);
                tMatrixRowFrom++;
            }
        }
    });
}

void setDataFromNonBlock(
            Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
      const Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues)
{
    auto tNumMatrixRows = aMatrix->numRows();
    auto tNumBlockRows = aMatrix->blockSizeRow();
    auto tNumBlockCols = aMatrix->blockSizeCol();
    auto tBlockSize = tNumBlockRows*tNumBlockCols;

    // generate block row map
    //
    auto tNumNodeRows = tNumMatrixRows/tNumBlockRows;
    Plato::ScalarVectorT<Plato::OrdinalType> tRowMap("block row map", tNumNodeRows+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows+1), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumBlockRows;
        auto tLocalRowIndex = tMatrixRowIndex % tNumBlockRows;
        if(tLocalRowIndex == 0)
          tRowMap(tBlockRowIndex) = aMatrixRowMap(tMatrixRowIndex)/tBlockSize;
    });
    aMatrix->setRowMap(tRowMap);

    // generate block col map and block values
    //
    auto tNumBlockMatEntries = aMatrixValues.extent(0);
    auto tNumBlockColEntries = tNumBlockMatEntries / tBlockSize;
    Plato::ScalarVectorT<Plato::OrdinalType> tColMap("block col map", tNumBlockColEntries);
    Plato::ScalarVectorT<Plato::Scalar>      tValues("block values",  tNumBlockMatEntries);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumBlockRows;
        auto tLocalRowIndex = tMatrixRowIndex % tNumBlockRows;
        auto tFrom = tRowMap(tBlockRowIndex);
        auto tTo   = tRowMap(tBlockRowIndex+1);
        Plato::OrdinalType tBlockRowFrom = tRowMap(tBlockRowIndex);
        Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
        for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
        {
            if(tLocalRowIndex == 0)
            {
                tColMap(tColMapIndex) = aMatrixColMap(tMatrixRowFrom)/tNumBlockCols;
            }
            for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumBlockCols; ++tBlockColOffset )
            {
                auto tValIndex = tColMapIndex*tBlockSize + tLocalRowIndex*tNumBlockCols + tBlockColOffset;
                tValues(tValIndex) = aMatrixValues(tMatrixRowFrom++);
            }
        }
    });
    aMatrix->setColumnIndices(tColMap);
    aMatrix->setEntries(tValues);
}

void RowSummedInverseMultiply(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
            Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo)
{
    auto tNumBlockRows = aInMatrixOne->rowMap().size()-1;
    auto tNumRows = aInMatrixOne->numRows();
    Plato::RowSum tRowSumFunctor(aInMatrixOne);
    Plato::DiagonalInverseMultiply tDiagInverseMultiplyFunctor(aInMatrixTwo);
    Plato::ScalarVector tRowSum("row sum", tNumRows);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tBlockRowOrdinal) {
      tRowSumFunctor(tBlockRowOrdinal, tRowSum);
      tDiagInverseMultiplyFunctor(tBlockRowOrdinal, tRowSum);
    });
}

void MatrixMinusEqualsMatrix(
            Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo)
{
    auto tEntriesOne = aInMatrixOne->entries();
    auto tEntriesTwo = aInMatrixTwo->entries();
    auto tNumEntries = tEntriesOne.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & tEntryOrdinal) {
      tEntriesOne(tEntryOrdinal) -= tEntriesTwo(tEntryOrdinal);
    });
}

void MatrixMinusEqualsMatrix(
            Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo, Plato::OrdinalType aOffset)
{
    auto tRowMap = aInMatrixOne->rowMap();
    auto tNumBlockRows = tRowMap.size()-1;

    auto tFromBlockSizeRow = aInMatrixOne->blockSizeRow();
    auto tFromBlockSizeCol = aInMatrixOne->blockSizeCol();
    auto tToBlockSizeRow   = aInMatrixTwo->blockSizeRow();
    auto tToBlockSizeCol   = aInMatrixTwo->blockSizeCol();

    assert(tToBlockSizeCol == tFromBlockSizeCol);

    auto tEntriesOne = aInMatrixOne->entries();
    auto tEntriesTwo = aInMatrixTwo->entries();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tBlockRowOrdinal) {
      
        auto tFrom = tRowMap(tBlockRowOrdinal);
        auto tTo   = tRowMap(tBlockRowOrdinal+1);
        for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
        {
            auto tFromEntryOffset = tColMapIndex * tFromBlockSizeRow * tFromBlockSizeCol;
            auto tToEntryOffset   = tColMapIndex * tToBlockSizeRow   * tToBlockSizeCol + aOffset * tToBlockSizeCol;;
            for( Plato::OrdinalType tBlockColOrdinal=0; tBlockColOrdinal<tToBlockSizeCol; ++tBlockColOrdinal )
            {
                auto tToEntryIndex   = tToEntryOffset   + tBlockColOrdinal;
                auto tFromEntryIndex = tFromEntryOffset + tBlockColOrdinal;
                tEntriesOne[tToEntryIndex] -= tEntriesTwo[tFromEntryIndex];
            }
        }
    });
}

void MatrixMinusMatrix( 
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix,
            Plato::OrdinalType aOffset)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;
    typedef Kokkos::DefaultExecutionSpace device;

    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <OrdinalType, OrdinalType, Scalar,
        typename device::execution_space, 
        typename device::memory_space,
        typename device::memory_space > KernelHandle;

    const auto& tMatOne = *aInMatrixOne;
    const auto& tMatTwo = *aInMatrixTwo;
    auto& tOutMat = *aOutMatrix;
    const OrdinalType tNumRowsOne = tMatOne.numRows();
    const OrdinalType tNumColsOne = tMatOne.numCols();
    const OrdinalType tNumRowsTwo = tMatTwo.numRows();
    const OrdinalType tNumColsTwo = tMatTwo.numCols();
    const OrdinalType tNumRowsOut = tOutMat.numRows();
    const OrdinalType tNumColsOut = tOutMat.numCols();

    if (tNumRowsTwo != tNumColsOne) { THROWERR("input matrices have incompatible shapes"); }
    if (tNumRowsOut != tNumRowsOne) { THROWERR("output matrix has incorrect shape"); }
    if (tNumColsOut != tNumColsTwo) { THROWERR("output matrix has incorrect shape"); }

    ScalarView tMatOneValues;
    OrdinalView tMatOneRowMap, tMatOneColMap;
    PlatoDevel::getDataAsNonBlock(aInMatrixOne, tMatOneRowMap, tMatOneColMap, tMatOneValues);

    ScalarView tMatTwoValues;
    OrdinalView tMatTwoRowMap, tMatTwoColMap;
    PlatoDevel::getDataAsNonBlock(aInMatrixTwo, tMatTwoRowMap, tMatTwoColMap, tMatTwoValues);

    OrdinalView tOutRowMap ("output row map", tNumRowsOne + 1);

    KernelHandle tKernel;
    tKernel.create_spadd_handle(/*sort rows=*/ false);
    KokkosSparse::Experimental::spadd_symbolic< KernelHandle,
      OrdinalView, OrdinalView,
      OrdinalView, OrdinalView,
      OrdinalView, OrdinalView
    >
    ( &tKernel,
      tMatOneRowMap, tMatOneColMap,
      tMatTwoRowMap, tMatTwoColMap,
      tOutRowMap
    );

    auto tAddHandle = tKernel.get_spadd_handle();

    size_t tNumOutValues = tAddHandle->get_max_result_nnz();
    OrdinalView tOutColMap(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValues);
    ScalarView  tOutValues(Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValues);
    KokkosSparse::Experimental::spadd_numeric< KernelHandle,
      OrdinalView, OrdinalView, Scalar, ScalarView,
      OrdinalView, OrdinalView, Scalar, ScalarView,
      OrdinalView, OrdinalView, ScalarView
    >
    ( &tKernel,
      tMatOneRowMap, tMatOneColMap, tMatOneValues, 1.0,
      tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, -1.0,
      tOutRowMap,    tOutColMap,    tOutValues
    );

    PlatoDevel::setDataFromNonBlock(aOutMatrix, tOutRowMap, tOutColMap, tOutValues);
    tKernel.destroy_spadd_handle();
}
void MatrixMatrixMultiply( 
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix,
      SPGEMMAlgorithm aAlgorithm = SPGEMM_KK_SPEED)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;
    typedef Kokkos::DefaultExecutionSpace device;

    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <OrdinalType, OrdinalType, Scalar,
        typename device::execution_space, 
        typename device::memory_space,
        typename device::memory_space > KernelHandle;

    KernelHandle tKernel;
    tKernel.set_team_work_size(1);
    tKernel.set_dynamic_scheduling(false);

    tKernel.create_spgemm_handle(aAlgorithm);

    const auto& tMatOne = *aInMatrixOne;
    const auto& tMatTwo = *aInMatrixTwo;
    auto& tOutMat = *aOutMatrix;
    const OrdinalType tNumRowsOne = tMatOne.numRows();
    const OrdinalType tNumColsOne = tMatOne.numCols();
    const OrdinalType tNumRowsTwo = tMatTwo.numRows();
    const OrdinalType tNumColsTwo = tMatTwo.numCols();
    const OrdinalType tNumRowsOut = tOutMat.numRows();
    const OrdinalType tNumColsOut = tOutMat.numCols();

    if (tNumRowsTwo != tNumColsOne) { THROWERR("input matrices have incompatible shapes"); }
    if (tNumRowsOut != tNumRowsOne) { THROWERR("output matrix has incorrect shape"); }
    if (tNumColsOut != tNumColsTwo) { THROWERR("output matrix has incorrect shape"); }

    ScalarView tMatOneValues;
    OrdinalView tMatOneRowMap, tMatOneColMap;
    PlatoDevel::getDataAsNonBlock(aInMatrixOne, tMatOneRowMap, tMatOneColMap, tMatOneValues);

    ScalarView tMatTwoValues;
    OrdinalView tMatTwoRowMap, tMatTwoColMap;
    PlatoDevel::getDataAsNonBlock(aInMatrixTwo, tMatTwoRowMap, tMatTwoColMap, tMatTwoValues);

    OrdinalView tOutRowMap ("output row map", tNumRowsOne + 1);
    spgemm_symbolic ( &tKernel, tNumRowsOne, tNumRowsTwo, tNumColsTwo,
        tMatOneRowMap, tMatOneColMap, /*transpose=*/false,
        tMatTwoRowMap, tMatTwoColMap, /*transpose=*/false,
        tOutRowMap
    );

    OrdinalView tOutColMap;
    ScalarView  tOutValues;
    size_t tNumOutValues = tKernel.get_spgemm_handle()->get_c_nnz();
    if (tNumOutValues){
      tOutColMap = OrdinalView(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValues);
      tOutValues = ScalarView (Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValues);
    }
    spgemm_numeric( &tKernel, tNumRowsOne, tNumRowsTwo, tNumColsTwo,
        tMatOneRowMap, tMatOneColMap, tMatOneValues, /*transpose=*/false,
        tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, /*transpose=*/false,
        tOutRowMap, tOutColMap, tOutValues
    );

    PlatoDevel::setDataFromNonBlock(aOutMatrix, tOutRowMap, tOutColMap, tOutValues);
    tKernel.destroy_spgemm_handle();
}

} // end namespace PlatoDevel

Teuchos::RCP<Teuchos::ParameterList> gElastostaticsParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>        \n"
    "  <ParameterList name='Material Model'>                             \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                 \n"
    "      <Parameter name='Poissons Ratio' type='double' value='0.3'/>  \n"
    "      <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>\n"
    "    </ParameterList>                                                \n"
    "  </ParameterList>                                                  \n"
    "  <ParameterList name='Elliptic'>                                   \n"
    "    <ParameterList name='Penalty Function'>                         \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>        \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>           \n"
    "    </ParameterList>                                                \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                    \n"
  );


template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aView,
      const std::vector<DataType>          & aVec)
 {
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    for (unsigned int i = 0; i < aVec.size(); ++i)
    {
        if(tView_host(i) != aVec[i])
        {
            return false;
        }
    }
    return true;
 }

template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aViewA,
      const Plato::ScalarVectorT<DataType> & aViewB)
 {
    if( aViewA.extent(0) != aViewB.extent(0) ) return false;

    auto tViewA_host = Kokkos::create_mirror(aViewA);
    Kokkos::deep_copy(tViewA_host, aViewA);
    auto tViewB_host = Kokkos::create_mirror(aViewB);
    Kokkos::deep_copy(tViewB_host, aViewB);
    for (unsigned int i = 0; i < aViewA.extent(0); ++i)
    {
        if(tViewA_host(i) != tViewB_host(i)) return false;
    }
    return true;
 }

bool is_zero(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrix)
 {
    auto tEntries = aInMatrix->entries();
    auto tEntries_host = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tEntries_host, tEntries);
    for (unsigned int i = 0; i < tEntries_host.extent(0); ++i)
    {
        if(tEntries_host(i) != 0.0) return false;
    }
    return true;
 }

bool is_same(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixA,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixB)
 {
    if( !is_same(aInMatrixA->rowMap(), aInMatrixB->rowMap()) )
    {
        return false;
    }
    if( !is_same(aInMatrixA->columnIndices(), aInMatrixB->columnIndices()) )
    {
        return false;
    }
    if( !is_same(aInMatrixA->entries(), aInMatrixB->entries()) )
    {
        return false;
    }
    return true;
 }

Teuchos::RCP<Plato::CrsMatrixType> createSquareMatrix()
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);
  auto nverts = mesh->nverts();

  // create vector data
  //
  Plato::ScalarVector u("state", spaceDim*nverts);
  Plato::ScalarVector z("control", nverts);
  Plato::fill(1.0, z);

  // create residual function
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::VectorFunction<::Plato::Mechanics<spaceDim>>
    tVectorFunction(*mesh, tMeshSets, tDataMap, *gElastostaticsParams, gElastostaticsParams->get<std::string>("PDE Constraint"));

  // compute and test objective value
  //
  return tVectorFunction.gradient_u(u,z);
}

template <typename PhysicsT>
Teuchos::RCP<Plato::VectorFunctionVMS<PhysicsT>>
createStabilizedResidual(Teuchos::RCP<Omega_h::Mesh> aMesh)
{
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;

  return Teuchos::rcp( new Plato::VectorFunctionVMS<PhysicsT>
         (*aMesh, tMeshSets, tDataMap, *gElastostaticsParams, gElastostaticsParams->get<std::string>("PDE Constraint")));
}

template <typename PhysicsT>
Teuchos::RCP<Plato::VectorFunctionVMS<typename PhysicsT::ProjectorT>>
createStabilizedProjector(Teuchos::RCP<Omega_h::Mesh> aMesh)
{
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  return Teuchos::rcp( new Plato::VectorFunctionVMS<typename PhysicsT::ProjectorT>
         (*aMesh, tMeshSets, tDataMap, *gElastostaticsParams, std::string("State Gradient Projection")));
}

/******************************************************************************/
/*! 
  \brief Transform a block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_FromToBlockMatrix)
{
  auto tMatrixA = createSquareMatrix();

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixA->numCols();
  auto tNumBlockSizeRows = tMatrixA->blockSizeRow();
  auto tNumBlockSizeCols = tMatrixA->blockSizeCol();
  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumBlockSizeRows, tNumBlockSizeCols) );

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  PlatoDevel::getDataAsNonBlock  (tMatrixA, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  std::vector<Plato::OrdinalType> tGoldMatrixRowMap = {
    0, 24, 48, 72, 99, 126, 153, 168, 183, 198, 231, 264, 297, 318, 339, 360,
    375, 390, 405, 426, 447, 468, 483, 498, 513, 540, 567, 594, 627, 660, 693,
    714, 735, 756, 771, 786, 801, 822, 843, 864, 909, 954, 999, 1032, 1065,
    1098, 1125, 1152, 1179, 1212, 1245, 1278, 1305, 1332, 1359, 1383, 1407,
    1431, 1458, 1485, 1512, 1545, 1578, 1611, 1644, 1677, 1710, 1731, 1752,
    1773, 1788, 1803, 1818, 1839, 1860, 1881, 1908, 1935, 1962, 1977, 1992, 2007
  };
  TEST_ASSERT(is_same(tMatrixRowMap, tGoldMatrixRowMap));
 
  std::vector<Plato::OrdinalType> tGoldMatrixColMap = {
    0, 1, 2, 24, 25, 26, 3, 4, 5, 9, 10, 11, 27, 28, 29, 39, 40, 41, 63, 64, 65,
    75, 76, 77, 0, 1, 2, 24, 25, 26, 3, 4, 5, 9, 10, 11, 27, 28, 29, 39, 40, 41,
    63, 64, 65, 75, 76, 77, 0, 1, 2, 24, 25, 26, 3, 4, 5, 9, 10, 11, 27, 28, 29,
    39, 40, 41, 63, 64, 65, 75, 76, 77, 0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8,
    12, 13, 14, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 67, 68, 0, 1, 2
  };
  TEST_ASSERT(is_same(tMatrixColMap, tGoldMatrixColMap));

  std::vector<Plato::Scalar> tGoldMatrixEntries = {
    352564.1025641025, 0, 0, -64102.564102564102, 32051.282051282051, 0,
    -64102.564102564102, 0, 32051.282051282051, 0, 32051.282051282051, 32051.282051282051,
    0, -80128.205128205125, 48076.923076923071, 0, -80128.205128205125, -80128.205128205125,
    0, 48076.923076923071, -80128.205128205125, -224358.97435897432, 48076.923076923071, 48076.923076923071,
    0, 352564.10256410256, 0, 48076.923076923071, -224358.97435897432, 48076.923076923071,
    0, -64102.564102564102, 32051.282051282051, 48076.923076923071, 0, -80128.205128205125
  };
  TEST_ASSERT(is_same(tMatrixEntries, tGoldMatrixEntries));

  PlatoDevel::setDataFromNonBlock(tMatrixB, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  TEST_ASSERT(is_same(tMatrixA, tMatrixB));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then compute A.A and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_MatrixMatrixMultiply)
{
  auto tMatrixA = createSquareMatrix();

  auto tNumOutRows = tMatrixA->numRows();
  auto tNumOutCols = tMatrixA->numCols();
  auto tNumOutBlockSizeRows = tMatrixA->blockSizeRow();
  auto tNumOutBlockSizeCols = tMatrixA->blockSizeCol();
  auto tMatrixAA = Teuchos::rcp( new Plato::CrsMatrixType( tNumOutRows, tNumOutCols, tNumOutBlockSizeRows, tNumOutBlockSizeCols) );

  PlatoDevel::MatrixMatrixMultiply( tMatrixA, tMatrixA, tMatrixAA);

  std::vector<Plato::OrdinalType> tGoldMatrixRowMap = {
    0, 27, 49, 63, 85, 101, 115, 131, 145, 167, 189, 205, 219, 235, 262, 284, 306, 328, 350, 377, 399, 421, 443, 459, 473, 489, 511, 525
  };
  TEST_ASSERT(is_same(tMatrixAA->rowMap(), tGoldMatrixRowMap));

  std::vector<Plato::OrdinalType> tGoldMatrixColMap = {
    0, 8, 1, 3, 9, 13, 21, 25, 7, 6, 12, 16, 2, 4, 14, 22, 5, 15, 10, 11, 17, 20, 18, 19, 23, 24, 26, 0, 8, 1, 3,
    9, 13, 21, 25, 2, 4, 14, 22, 6, 5, 15, 16, 17, 18, 19, 20, 23, 24, 0, 1, 3, 2, 4, 13, 14, 21, 22, 5, 15, 18,
    19, 23, 0, 8, 1, 3, 9, 13, 21, 25, 2, 4, 14, 22, 6, 5, 15, 16, 7, 12, 17, 18, 19, 20, 0, 1, 3, 2, 4, 13, 14
  };
  TEST_ASSERT(is_same(tMatrixAA->columnIndices(), tGoldMatrixColMap));

  std::vector<Plato::Scalar> tGoldMatrixEntries = {
    221893491124.26025, -15922912557.527939, -15922912557.527939, -15922912557.527939,  221893491124.26028, -15922912557.527941,
   -15922912557.527939, -15922912557.527941,  221893491124.26025, -55473372781.065079,  35441321499.013802, -19261587771.203156,
    96564760026.298477, -260930309007.23196,  96564760026.298462, -19261587771.203156,  35441321499.013802, -55473372781.065071,
   -55473372781.065079, -19261587771.203152,  35441321499.013802, -19261587771.203156, -55473372781.065071,  35441321499.013802,
    96564760026.298477,  96564760026.298462, -260930309007.23199, -26709401709.401711,  59325690335.30571,   59325690335.30571,
    124815088757.39642,  46741452991.452988, -157431377383.30042
  };
  TEST_ASSERT(is_same(tMatrixAA->entries(), tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then verify that A - A = 0.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_MatrixMinusEqualsMatrix)
{
  auto tMatrixA = createSquareMatrix();
  PlatoDevel::MatrixMinusEqualsMatrix( tMatrixA, tMatrixA );
  TEST_ASSERT(is_zero(tMatrixA));
}

/******************************************************************************/
/*! 
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_RowSummedInverseMultiply)
{
  auto tMatrix = createSquareMatrix();
  auto tNumOutRows = tMatrix->numRows();
  auto tNumOutCols = tMatrix->numCols();
  auto tNumOutBlockSizeRows = tMatrix->blockSizeRow();
  auto tNumOutBlockSizeCols = tMatrix->blockSizeCol();
  auto tMatrixSquared = Teuchos::rcp( new Plato::CrsMatrixType( tNumOutRows, tNumOutCols, tNumOutBlockSizeRows, tNumOutBlockSizeCols) );

  PlatoDevel::MatrixMatrixMultiply( tMatrix, tMatrix, tMatrixSquared);
  PlatoDevel::RowSummedInverseMultiply( tMatrixSquared, tMatrixSquared );
  std::vector<Plato::Scalar> tGoldMatrixEntries = {
    352564.1025641025, 0, 0, -64102.564102564102, 32051.282051282051, 0,
    -64102.564102564102, 0, 32051.282051282051, 0, 32051.282051282051, 32051.282051282051,
    0, -80128.205128205125, 48076.923076923071, 0, -80128.205128205125, -80128.205128205125,
    0, 48076.923076923071, -80128.205128205125, -224358.97435897432, 48076.923076923071, 48076.923076923071,
    0, 352564.10256410256, 0, 48076.923076923071, -224358.97435897432, 48076.923076923071,
    0, -64102.564102564102, 32051.282051282051, 48076.923076923071, 0, -80128.205128205125
  };
  auto tMatrixEntries = tMatrixSquared->entries();
  TEST_ASSERT(is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create a stabilized residual, g, and a projector, P, then compute
 derivatives dg/du^T, dg/dn^T, dP/du^T, and dP/dn and the condensed matrix:

 A = dg/du^T - dP/du^T . RowSum(dP/dn)^{-1} . dg/dn^T

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_CondenseMatrix)
{
  constexpr int cSpaceDim  = 3;
  constexpr int cMeshWidth = 2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);
  using PhysicsT = Plato::StabilizedMechanics<cSpaceDim>;
  auto tResidual = createStabilizedResidual<PhysicsT>(tMesh);
  auto tProjector = createStabilizedProjector<PhysicsT>(tMesh);

  auto tNverts = tMesh->nverts();
  Plato::ScalarVector U("state",          tResidual->size());
  Plato::ScalarVector N("project p grad", tProjector->size());
  Plato::ScalarVector z("control",        tNverts);
  Plato::ScalarVector p("nodal pressure", tNverts);
  Plato::fill(1.0, z);

  //                                        u, n, z
  auto t_dg_du_T = tResidual->gradient_u_T (U, N, z);
  auto t_dg_dn_T = tResidual->gradient_n_T (U, N, z);
  auto t_dP_dn_T = tProjector->gradient_n_T(N, p, z);
  auto t_dP_du   = tProjector->gradient_u  (N, p, z);
  
  auto tNumOutRows = t_dP_dn_T->numRows();
  auto tNumOutCols = t_dg_dn_T->numCols();
  auto tNumOutBlockSizeRows = t_dP_dn_T->blockSizeRow();
  auto tNumOutBlockSizeCols = t_dg_dn_T->blockSizeCol();
  auto tMatrixProduct = Teuchos::rcp( new Plato::CrsMatrixType( tNumOutRows, tNumOutCols, tNumOutBlockSizeRows, tNumOutBlockSizeCols) );
  auto tCondensedMatrix = Teuchos::rcp( new Plato::CrsMatrixType( tNumOutRows, tNumOutCols, tNumOutBlockSizeRows, tNumOutBlockSizeCols) );

  PlatoDevel::RowSummedInverseMultiply( t_dP_du, t_dg_dn_T );

  PlatoDevel::MatrixMatrixMultiply( t_dP_dn_T, t_dg_dn_T, tMatrixProduct );

  auto tOffset = PhysicsT::ProjectorT::SimplexT::mProjectionDof;
  PlatoDevel::MatrixMinusMatrix( t_dg_du_T, tMatrixProduct, tCondensedMatrix, tOffset );

  std::cout << "wait here shitass" << std::endl;

  //TODO compare with gold
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_InvertLocalMatrices)
{
    const int N = 3; // Number of matrices to invert
    Plato::ScalarArray3D tMatrix("Matrix A", N, 2, 2);
    auto tHostMatrix = Kokkos::create_mirror(tMatrix);
    for (unsigned int i = 0; i < N; ++i)
    {
      const Plato::Scalar tScaleFactor = 1.0 / (1.0 + i);
      tHostMatrix(i,0,0) = -2.0 * tScaleFactor;
      tHostMatrix(i,1,0) =  1.0 * tScaleFactor;
      tHostMatrix(i,0,1) =  1.5 * tScaleFactor;
      tHostMatrix(i,1,1) = -0.5 * tScaleFactor;
    }
    Kokkos::deep_copy(tMatrix, tHostMatrix);

    Plato::ScalarArray3D tAInverse("A Inverse", N, 2, 2);
    auto tHostAInverse = Kokkos::create_mirror(tAInverse);
    for (unsigned int i = 0; i < N; ++i)
    {
      tHostAInverse(i,0,0) = 1.0;
      tHostAInverse(i,1,0) = 0.0;
      tHostAInverse(i,0,1) = 0.0;
      tHostAInverse(i,1,1) = 1.0;
    }
    Kokkos::deep_copy(tAInverse, tHostAInverse);

    using namespace KokkosBatched::Experimental;

    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in/out]A: 2d view
    /// [in]tiny: a magnitude scalar value compatible to the value type of A
    /// int SerialLU<Algo::LU::Unblocked>::invoke(const AViewType &A, const ScalarType tiny = 0)

    /// [template]SideType: Side::Left or Side::Right
    /// [template]UploType: Uplo::Upper or Uplo::Lower
    /// [template]TransType: Trans::NoTranspose or Trans::Transpose
    /// [template]DiagType: Diag::Unit or Diag::NonUnit
    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in]alpha: a scalar value
    /// [in]A: 2d view
    /// [in/out]B: 2d view
    /// int SerialTrsm<SideType,UploType,TransType,DiagType,AlgoType>
    ///    ::invoke(const ScalarType alpha, const AViewType &A, const BViewType &B);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,N), LAMBDA_EXPRESSION(const Plato::OrdinalType & n) {
      auto A    = Kokkos::subview(tMatrix  , n, Kokkos::ALL(), Kokkos::ALL());
      auto Ainv = Kokkos::subview(tAInverse, n, Kokkos::ALL(), Kokkos::ALL());

      SerialLU<Algo::LU::Blocked>::invoke(A);
      SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
      SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
    });

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar> > tGoldMatrixInverse = { {1.0, 3.0}, {2.0, 4.0} };

    Kokkos::deep_copy(tHostAInverse, tAInverse);
    for (unsigned int n = 0; n < N; ++n)
      for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          {
            //printf("Matrix %d Inverse (%d,%d) = %f\n", n, i, j, tHostAInverse(n, i, j));
            const Plato::Scalar tScaleFactor = (1.0 + n);
            TEST_FLOATING_EQUALITY(tHostAInverse(n, i, j), tScaleFactor * tGoldMatrixInverse[i][j], tTolerance);
          }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, HyperbolicTangentProjection)
{
    const Plato::OrdinalType tNumNodesPerCell = 2;
    typedef Sacado::Fad::SFad<Plato::Scalar, tNumNodesPerCell> FadType;

    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tOutputVal("OutputVal", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tOutputGrad("OutputGrad", tNumNodesPerCell);
    Plato::ScalarMultiVectorT<FadType> tControl("Control", tNumCells, tNumNodesPerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tControl(aCellOrdinal, 0) = FadType(tNumNodesPerCell, 0, 1.0);
        tControl(aCellOrdinal, 1) = FadType(tNumNodesPerCell, 1, 1.0);
    }, "Set Controls");

    // SET EVALUATION TYPES FOR UNIT TEST
    Plato::HyperbolicTangentProjection tProjection;
    Plato::ApplyProjection<Plato::HyperbolicTangentProjection> tApplyProjection(tProjection);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        FadType tValue = tApplyProjection(aCellOrdinal, tControl);
        tOutputVal(aCellOrdinal) = tValue.val();
        tOutputGrad(0) = tValue.dx(0);
        tOutputGrad(1) = tValue.dx(1);
    }, "UnitTest: HyperbolicTangentProjection_GradZ");

    // TEST OUTPUT
    auto tHostVal = Kokkos::create_mirror(tOutputVal);
    Kokkos::deep_copy(tHostVal, tOutputVal);
    auto tHostGrad = Kokkos::create_mirror(tOutputGrad);
    Kokkos::deep_copy(tHostGrad, tOutputGrad);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldVal = { 1.0 };
    std::vector<Plato::Scalar> tGoldGrad = { 4.539992985607449e-4, 4.539992985607449e-4 };
    TEST_FLOATING_EQUALITY(tHostVal(0), tGoldVal[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(0), tGoldGrad[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(1), tGoldGrad[1], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_ConditionalExpression)
{
    const Plato::OrdinalType tRange = 1;
    Plato::ScalarVector tOuput("Output", 2 /* number of outputs */);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(Plato::OrdinalType tOrdinal)
    {
        Plato::Scalar tConditionalValOne = 5;
        Plato::Scalar tConditionalValTwo = 4;
        Plato::Scalar tConsequentValOne = 2;
        Plato::Scalar tConsequentValTwo = 3;
        tOuput(tOrdinal) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);

        tConditionalValOne = 3;
        tOuput(tOrdinal + 1) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);
    }, "Test inline conditional_expression function");

    auto tHostOuput = Kokkos::create_mirror(tOuput);
    Kokkos::deep_copy(tHostOuput, tOuput);
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tHostOuput(0), 3.0, tTolerance);
    TEST_FLOATING_EQUALITY(tHostOuput(1), 2.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_dot)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::fill(1.0, tVecA);
  Plato::ScalarVector tVecB("Vec B", tNumElems);
  Plato::fill(2.0, tVecB);

  const Plato::Scalar tOutput = Plato::dot(tVecA, tVecB);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(20., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_norm)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::fill(1.0, tVecA);

  const Plato::Scalar tOutput = Plato::norm(tVecA);
  constexpr Plato::Scalar tTolerance = 1e-6;
  TEST_FLOATING_EQUALITY(3.16227766016838, tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_sum)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec", tNumElems);
  Plato::fill(1.0, tVecA);

  Plato::Scalar tOutput = 0.0;
  Plato::local_sum(tVecA, tOutput);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(10., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_fill)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::fill(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_copy)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::fill(2.0, tSomeVector);

  Plato::ScalarVector tSomeOtherVector("some other vector", numVerts);
  Plato::copy(tSomeVector, tSomeOtherVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  auto tSomeOtherVectorHost = Kokkos::create_mirror_view(tSomeOtherVector);
  Kokkos::deep_copy(tSomeOtherVectorHost, tSomeOtherVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), tSomeOtherVectorHost(0), 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), tSomeOtherVectorHost(numVerts-1), 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_scale)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::fill(1.0, tSomeVector);
  Plato::scale(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_update)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tVector_A("vector a", numVerts);
  Plato::ScalarVector tVector_B("vector b", numVerts);
  Plato::fill(1.0, tVector_A);
  Plato::fill(2.0, tVector_B);
  Plato::update(2.0, tVector_A, 3.0, tVector_B);

  auto tVector_B_Host = Kokkos::create_mirror_view(tVector_B);
  Kokkos::deep_copy(tVector_B_Host, tVector_B);
  TEST_FLOATING_EQUALITY(tVector_B_Host(0), 8.0, 1e-17);
  TEST_FLOATING_EQUALITY(tVector_B_Host(numVerts-1), 8.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, PlatoMathHelpers_MatrixTimesVectorPlusVector)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( mesh->nverts(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);

  // create mesh based displacement from host data
  //
  auto stateSize = spaceDim*mesh->nverts();
  Plato::ScalarVector u("state",stateSize);
  auto u_host = Kokkos::create_mirror_view(u);
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( int i = 0; i<stateSize; i++) u_host(i) = (disp += dval);
  Kokkos::deep_copy(u, u_host);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Objective' type='string' value='My Internal Elastic Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='My Internal Elastic Energy'>                           \n"
    "    <Parameter name='Type' type='string' value='Scalar Function'/>            \n"
    "    <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Elliptic'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Model'>                                       \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                           \n"
    "      <Parameter name='Poissons Ratio' type='double' value='0.3'/>            \n"
    "      <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>          \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create objective
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(*mesh, tMeshSets, tDataMap, *tParams, tParams->get<std::string>("Objective"));

  auto dfdx = eeScalarFunction.gradient_x(u,z);

  // create PDE constraint
  //
  Plato::VectorFunction<::Plato::Mechanics<spaceDim>>
    esVectorFunction(*mesh, tMeshSets, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint"));

  auto dgdx = esVectorFunction.gradient_x(u,z);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_before.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }
#endif

  Plato::MatrixTimesVectorPlusVector(dgdx, u, dfdx);

  auto dfdx_host = Kokkos::create_mirror_view(dfdx);
  Kokkos::deep_copy(dfdx_host, dfdx);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_after.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }

  {
    std::ofstream ofile;
    ofile.open("u.dat");
    for(int i=0; i<u_host.size(); i++) 
      ofile << u_host(i) << std::endl;
    ofile.close();
  }

  {
    auto rowMapHost = Kokkos::create_mirror_view(dgdx->rowMap()); 
    Kokkos::deep_copy(rowMapHost, dgdx->rowMap());
    std::ofstream ofile;
    ofile.open("rowMap.dat");
    for(int i=0; i<rowMapHost.size(); i++) 
      ofile << rowMapHost(i) << std::endl;
    ofile.close();
  }

  {
    auto columnIndicesHost = Kokkos::create_mirror_view(dgdx->columnIndices());
    Kokkos::deep_copy(columnIndicesHost, dgdx->columnIndices());
    std::ofstream ofile;
    ofile.open("columnIndices.dat");
    for(int i=0; i<columnIndicesHost.size(); i++) 
      ofile << columnIndicesHost(i) << std::endl;
    ofile.close();
  }

  {
    auto entriesHost = Kokkos::create_mirror_view(dgdx->entries());
    Kokkos::deep_copy(entriesHost, dgdx->entries());
    std::ofstream ofile;
    ofile.open("entries.dat");
    for(int i=0; i<entriesHost.size(); i++) 
      ofile << std::setprecision(18) << entriesHost(i) << std::endl;
    ofile.close();
  }
#endif

  std::vector<Plato::Scalar> dfdx_gold = {
 97.7538461538461831, -62.6884615384616453, -29.1461538461538794, 
91.8173076923076934, -71.4288461538462229, 9.06923076923077076, 
6.02884615384615863, -18.8942307692307878, 30.8653846153846274, 
47.0480769230768630, 30.7442307692307608, -11.8846153846153673, 
-6.70384615384615579, 3.80192307692307541, 25.6788461538461377, 
-1.80000000000000249, 10.5923076923076849, 7.82307692307691660, 
-5.28461538461538716, 23.7230769230769098, -5.97692307692306990, 
-1.22307692307692806, 8.26153846153846061, -3.71538461538461373, 
-13.4019230769230564, 2.43461538461537774, -17.5557692307692150, 
44.2846153846154351, -5.57884615384621085, -19.1826923076923457, 
24.9403846153846658, -30.7211538461539249, -1.35000000000000453, 
3.30000000000000515, 4.96153846153846523, -1.61538461538461231, 
-0.0692307692307672085, 8.58461538461539853, -10.1769230769230710, 
-18.1442307692305782, 22.3730769230769155, 19.8634615384615891, 
-43.6442307692306954, 20.6711538461538638, 39.0692307692307992, 
-40.9730769230769027, 6.64038461538460645, 16.8865384615384571, 
-33.3403846153846359, 9.21923076923075158, -7.03269230769231868, 
5.44038461538462847, 6.82500000000002593, -15.3115384615384951, 
-5.98846153846156071, 2.59615384615385558, 3.15000000000001279, 
7.14807692307694431, -3.63461538461539790, 6.31730769230771649, 
11.7230769230769436, 15.8942307692308269, 1.21730769230769131, 
-60.2653846153847113, -29.1461538461537906, 6.47307692307693650, 
-34.1134615384616211, -19.6961538461538161, 21.7557692307692569, 
-0.230769230769231531, 0.461538461538463007, 0.876923076923080247, 
0.294230769230770239, 0.934615384615388844, 3.40961538461539604, 
-87.7384615384617348, 56.4865384615386503, -56.7057692307692918, 
13.1423076923077282, 6.58269230769232827, -12.8019230769231100};

  for(int iNode=0; iNode<int(dfdx_gold.size()); iNode++){
      TEST_FLOATING_EQUALITY(dfdx_host[iNode], dfdx_gold[iNode], 1e-13);
  }
}

} // namespace PlatoUnitTests
