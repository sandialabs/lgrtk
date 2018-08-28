/*!
  These unit tests are for the Linear elastostatics functionality.
*/

#include "LGRTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "CrsMatrix.hpp"
#include "FEMesh.hpp"
#include "MatrixIO.hpp"
#include "VizOutput.hpp"
#include "MeshFixture.hpp"
#include "plato/PlatoStaticsTypes.hpp"

#include "ElastostaticSolve.hpp"
#include "PlatoTestHelpers.hpp"
#include "EssentialBCs.hpp"
#include "BodyLoads.hpp"

#ifdef HAVE_VIENNA_CL
#include "ViennaSparseLinearProblem.hpp"
#endif

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>

#include <Sacado.hpp>
#include <CrsLinearProblem.hpp>
#include <CrsMatrix.hpp>
#include <Fields.hpp>
#include <FieldDB.hpp>
#include <MeshIO.hpp>
#include <ParallelComm.hpp>

#include <impl/Kokkos_Timer.hpp>


static bool verbose_tests = false;


#ifdef HAVE_AMGX
  typedef lgr::AmgXSparseLinearProblem<Plato::OrdinalType> AmgXLinearProblem;
#endif

using namespace Plato::LinearElastostatics;

TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_FunctorMemberDatum )
{
  (void)out;
  (void)success;
  const int spaceDim =2;
  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);
  auto matrix = Plato::CreateMatrix<Plato::CrsMatrixType,spaceDim>(meshOmegaH.get());

  Plato::BlockMatrixEntryOrdinal<spaceDim, spaceDim> entryOrdinalLookup(matrix, meshOmegaH.get());

  auto matrixEntries = matrix->entries();
  auto entriesLength = matrixEntries.size();

  const int numCells  = meshOmegaH->nelems();
  const int dofsPerCell = (spaceDim+1)*spaceDim;
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  { 

    for (int iDof=0; iDof<dofsPerCell; iDof++)
    {
      for (int jDof=0; jDof<dofsPerCell; jDof++)
      {
          Plato::Scalar integral = 1.0;

          auto entryOrdinal = entryOrdinalLookup(cellOrdinal,iDof,jDof);
          if (entryOrdinal < entriesLength)
          {
              Kokkos::atomic_add(&matrixEntries(entryOrdinal), integral);
          }
      }
    }

  },"initialize");

  Plato::ScalarVector::HostMirror
    matrixEntriesHost = Kokkos::create_mirror_view( matrixEntries );
  Kokkos::deep_copy(matrixEntriesHost, matrixEntries);

//  for( int i=0; i<matrixEntriesHost.size(); i++)
//    std::cout << matrixEntriesHost(i) << std::endl;

}


/******************************************************************************/
/*! Test the CreateMatrix function.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create a CrsMatrix with 'spaceDim' dofs per node, i.e., the second template
       parameter is 'spaceDim'.

  Tests:
   1.  Check the rowMap. The rowMap object stores the offsets into the columnIndices for 
       each row. The offsets in gold_rowMap were computed with doc/gold/graph.py for the 
       2x2 mesh returned by getBoxMesh(2, 2).
   2.  Check the columnIndices.  The columnIndices object is a 1D list of the column ids 
       for all rows.  The rowMap provides offsets into this list for a given row.  The 
       values in gold_columnIndices were computed with graph.py for the 2x2 mesh returned
       by getBoxMesh(2, 2).  Note that the order of the ordinals in the columnIndices 
       list is not the same as the order in the gold_columnIndices list.  The test 
       checks to see that every ordinal in columnIndices occurs once in the gold so
       order isn't important.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_ScalarGraph2D )
{
 // Test setup

  const int spaceDim =2;
  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  if(0) 
    PlatoUtestHelpers::writeConnectivity(meshOmegaH, "mesh2D.connectivity", spaceDim);

  auto matrix = Plato::CreateMatrix<Plato::CrsMatrixType,spaceDim>(meshOmegaH.get());


  auto rowMap = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();

 // Define tests

  // Test 1: (see description above)

  // the rowMap is on the device, pull it to the host
  auto rowMapHost = Kokkos::create_mirror_view( rowMap );
  Kokkos::deep_copy(rowMapHost, rowMap);

  std::vector<int> gold_rowMap = {0,8,16,26,36,42,48,62,76,86,96,104,112,122,132,142,152,158};
  int numEntries = gold_rowMap.size();
  for( int i=0; i<numEntries; i++){
    TEST_EQUALITY(rowMapHost(i), gold_rowMap[i]);
    if( verbose_tests ){
      std::cout << "rowMapHost(i) == gold_rowMap[i]: " << rowMapHost(i) << " == " << gold_rowMap[i] << std::endl;
    }
  }


  // Test 2: (see description above)

  // the columnIndices are on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    columnIndicesHost = Kokkos::create_mirror_view( columnIndices );
  Kokkos::deep_copy(columnIndicesHost, columnIndices);

  std::vector<std::vector<int>> 
   gold_columnIndices = {
    {0,1,2,3,6,7,14,15},
    {0,1,2,3,6,7,14,15},
    {0,1,2,3,4,5,6,7,8,9}, 
    {0,1,2,3,4,5,6,7,8,9},
    {2,3,4,5,8,9}, 
    {2,3,4,5,8,9},
    {0,1,2,3,6,7,8,9,10,11,12,13,14,15},
    {0,1,2,3,6,7,8,9,10,11,12,13,14,15},
    {2,3,4,5,6,7,8,9,10,11},
    {2,3,4,5,6,7,8,9,10,11},
    {6,7,8,9,10,11,12,13},
    {6,7,8,9,10,11,12,13},
    {6,7,10,11,12,13,14,15,16,17},
    {6,7,10,11,12,13,14,15,16,17},
    {0,1,6,7,12,13,14,15,16,17},
    {0,1,6,7,12,13,14,15,16,17},
    {12,13,14,15,16,17},
    {12,13,14,15,16,17}};

  TEUCHOS_TEST_FOR_EXCEPT(gold_columnIndices.size() != gold_rowMap.size());

  int rowMapSize = gold_rowMap.size();
  for(int i=1; i<rowMapSize; i++){
    int begin = rowMapHost(i-1), end = rowMapHost(i);
    int size = gold_columnIndices[i-1].size();
    TEST_EQUALITY(end-begin, size);
    if( verbose_tests ) std::cout << "end-begin == gold_columnIndices[i-1].size(): " << end-begin 
                                  << " == " << gold_columnIndices[i-1].size() << std::endl;
    for(int j=begin; j<end; j++){
      TEST_EQUALITY(count(gold_columnIndices[i-1].begin(),gold_columnIndices[i-1].end(),columnIndicesHost(j)), 1);
      if( verbose_tests ){
        std::cout << columnIndicesHost(j) << " is in { ";
        for( auto id : gold_columnIndices[i-1] ) std::cout << id << " ";
        std::cout << "}" << std::endl;
      }
    }
  }
}

/******************************************************************************/
/*! Test the CreateBlockMatrix function.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create a CrsMatrix with 'spaceDim' dofs per node, i.e., the second template
       parameter is 'spaceDim'.

  Tests:
   1.  Check the rowMap. The rowMap object stores the offsets into the columnIndices for 
       each row. The offsets in gold_rowMap were computed with doc/gold/graph.py for the 
       2x2 mesh returned by getBoxMesh(2, 2).
   2.  Check the columnIndices.  The columnIndices object is a 1D list of the column ids 
       for all rows.  The rowMap provides offsets into this list for a given row.  The 
       values in gold_columnIndices were computed with graph.py for the 2x2 mesh returned
       by getBoxMesh(2, 2).  Note that the order of the ordinals in the columnIndices 
       list is not the same as the order in the gold_columnIndices list.  The test 
       checks to see that every ordinal in columnIndices occurs once in the gold so
       order isn't important.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_BlockGraph2D )
{
 // Test setup

  const int spaceDim =2;
  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  if(0)
    PlatoUtestHelpers::writeConnectivity(meshOmegaH, "mesh2D.connectivity", spaceDim);

  auto matrix = Plato::CreateBlockMatrix<Plato::CrsMatrixType,spaceDim>(meshOmegaH.get());


  auto rowMap = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();

 // Define tests

  // Test 1: (see description above)

  // the rowMap is on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    rowMapHost = Kokkos::create_mirror_view( rowMap );
  Kokkos::deep_copy(rowMapHost, rowMap);

  std::vector<int> gold_rowMap = {0, 4, 9, 12, 19, 24, 28, 33, 38, 41};
  int numEntries = gold_rowMap.size();
  for( int i=0; i<numEntries; i++){
    TEST_EQUALITY(rowMapHost(i), gold_rowMap[i]);
    if( verbose_tests ){
      std::cout << "rowMapHost(i) == gold_rowMap[i]: " << rowMapHost(i) << " == " << gold_rowMap[i] << std::endl;
    }
  }


  // Test 2: (see description above)

  // the columnIndices are on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    columnIndicesHost = Kokkos::create_mirror_view( columnIndices );
  Kokkos::deep_copy(columnIndicesHost, columnIndices);

  int gold_columnIndicesLength = gold_rowMap.back();
  std::vector<std::vector<int>> 
   gold_columnIndices = {
     {0, 1, 3, 7},
     {0, 1, 2, 3, 4},
     {1, 2, 4},
     {0, 1, 3, 4, 5, 6, 7},
     {1, 2, 3, 4, 5},
     {3, 4, 5, 6},
     {3, 5, 6, 7, 8},
     {0, 3, 6, 7, 8},
     {6, 7, 8}};

  TEST_EQUALITY(gold_columnIndicesLength, int(columnIndicesHost.size()));

  int rowMapSize = gold_rowMap.size();
  for(int i=1; i<rowMapSize; i++){
    int begin = rowMapHost(i-1), end = rowMapHost(i);
    int size = gold_columnIndices[i-1].size();
    TEST_EQUALITY(end-begin, size);
    if( verbose_tests ) std::cout << "end-begin == gold_columnIndices[i-1].size(): " << end-begin 
                                  << " == " << gold_columnIndices[i-1].size() << std::endl;
    for(int j=begin; j<end; j++){
      TEST_EQUALITY(count(gold_columnIndices[i-1].begin(),gold_columnIndices[i-1].end(),columnIndicesHost(j)), 1);
      if( verbose_tests ){
        std::cout << columnIndicesHost(j) << " is in { ";
        for( auto id : gold_columnIndices[i-1] ) std::cout << id << " ";
        std::cout << "}" << std::endl;
      }
    }
  }
}


/******************************************************************************/
/*! Test the CreateMatrix function.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create a CrsMatrix with 'spaceDim' dofs per node, i.e., the second template
       parameter is 'spaceDim'.

  Tests:
   1.  Check the rowMap. The rowMap object stores the offsets into the columnIndices for 
       each row. The offsets in gold_rowMap were computed with doc/gold/graph.py for the 
       2x2x2 mesh returned by getBoxMesh(3, 2).
   2.  Check the columnIndices.  The columnIndices object is a 1D list of the column ids 
       for all rows.  The rowMap provides offsets into this list for a given row.  The 
       values in gold_columnIndices were computed with graph.py for the 2x2x2 mesh returned
       by getBoxMesh(3, 2).  Note that the order of the ordinals in the columnIndices 
       list is not the same as the order in the gold_columnIndices list.  The test 
       checks to see that every ordinal in columnIndices occurs once in the gold so
       order isn't important.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_ScalarGraph3D )
{
 // Test setup

  const int spaceDim =3;
  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto matrix = Plato::CreateMatrix<Plato::CrsMatrixType,spaceDim>(meshOmegaH.get());

  if(0)
    PlatoUtestHelpers::writeMesh(meshOmegaH, "GraphTest", spaceDim);

  if(0)
    PlatoUtestHelpers::writeConnectivity(meshOmegaH, "mesh3D.connectivity", spaceDim);

  auto rowMap = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();

 // Define tests

  // Test 1: (see description above)

  // the rowMap is on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    rowMapHost = Kokkos::create_mirror_view( rowMap );
  Kokkos::deep_copy(rowMapHost, rowMap);

  std::vector<int> 
    gold_rowMap = {   0,   24,   48,   72,   99,  126,  153,  168,  183,  198,  231, 
                    264,  297,  318,  339,  360,  375,  390,  405,  426,  447,  468, 
                    483,  498,  513,  540,  567,  594,  627,  660,  693,  714,  735,
                    756,  771,  786,  801,  822,  843,  864,  909,  954,  999, 1032, 
                   1065, 1098, 1125, 1152, 1179, 1212, 1245, 1278, 1305, 1332, 1359, 
                   1383, 1407, 1431, 1458, 1485, 1512, 1545, 1578, 1611, 1644, 1677, 
                   1710, 1731, 1752, 1773, 1788, 1803, 1818, 1839, 1860, 1881, 1908, 
                   1935, 1962, 1977, 1992, 2007};
                   
  TEST_EQUALITY(rowMapHost.size(), gold_rowMap.size());

  int numEntries = gold_rowMap.size();
  for( int i=0; i<numEntries; i++){
    TEST_EQUALITY(rowMapHost(i), gold_rowMap[i]);
    if( verbose_tests ){
      std::cout << "rowMapHost(i) == gold_rowMap[i]: " << rowMapHost(i) << " == " << gold_rowMap[i] << std::endl;
    }
  }


  // Test 2: (see description above)

  // the columnIndices are on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    columnIndicesHost = Kokkos::create_mirror_view( columnIndices );
  Kokkos::deep_copy(columnIndicesHost, columnIndices);

  int gold_columnIndicesLength = 2007;
  std::vector<std::vector<int>> 
   gold_columnIndices = {
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 24, 25, 26, 27, 28, 29, 39, 40, 41, 63, 64, 65, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 24, 25, 26, 27, 28, 29, 39, 40, 41, 63, 64, 65, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 24, 25, 26, 27, 28, 29, 39, 40, 41, 63, 64, 65, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 67, 68},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 67, 68},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 67, 68},
    {3, 4, 5, 6, 7, 8, 12, 13, 14, 42, 43, 44, 66, 67, 68},
    {3, 4, 5, 6, 7, 8, 12, 13, 14, 42, 43, 44, 66, 67, 68},
    {3, 4, 5, 6, 7, 8, 12, 13, 14, 42, 43, 44, 66, 67, 68},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46, 47},
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46, 47},
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 42, 43, 44, 45, 46, 47},
    {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 45, 46, 47},
    {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 45, 46, 47},
    {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 45, 46, 47},
    {9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50},
    {9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50},
    {9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50},
    {18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 48, 49, 50},
    {18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 48, 49, 50},
    {18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 48, 49, 50},
    {0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 48, 49, 50},
    {0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 48, 49, 50},
    {0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 48, 49, 50},
    {0, 1, 2, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 60, 61, 62, 75, 76, 77},
    {0, 1, 2, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 60, 61, 62, 75, 76, 77},
    {0, 1, 2, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 60, 61, 62, 75, 76, 77},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 51, 52, 53, 60, 61, 62, 75, 76, 77, 78, 79, 80},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 51, 52, 53, 60, 61, 62, 75, 76, 77, 78, 79, 80},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 51, 52, 53, 60, 61, 62, 75, 76, 77, 78, 79, 80},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53},
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 48, 49, 50, 51, 52, 53},
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 48, 49, 50, 51, 52, 53},
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 48, 49, 50, 51, 52, 53},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 24, 25, 26, 27, 28, 29, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 24, 25, 26, 27, 28, 29, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 9, 10, 11, 24, 25, 26, 27, 28, 29, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 75, 76, 77},
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 40, 41, 42, 43, 44, 45, 46, 47, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68},
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 40, 41, 42, 43, 44, 45, 46, 47, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68},
    {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 39, 40, 41, 42, 43, 44, 45, 46, 47, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68},
    {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56},
    {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56},
    {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56},
    {9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56},
    {9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56},
    {9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 61, 62},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 61, 62},
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 61, 62},
    {39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62},
    {39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62},
    {39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62},
    {39, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
    {39, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
    {39, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
    {27, 28, 29, 30, 31, 32, 39, 40, 41, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {27, 28, 29, 30, 31, 32, 39, 40, 41, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {27, 28, 29, 30, 31, 32, 39, 40, 41, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {0, 1, 2, 3, 4, 5, 39, 40, 41, 42, 43, 44, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 39, 40, 41, 42, 43, 44, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77},
    {0, 1, 2, 3, 4, 5, 39, 40, 41, 42, 43, 44, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77},
    {3, 4, 5, 6, 7, 8, 42, 43, 44, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71},
    {3, 4, 5, 6, 7, 8, 42, 43, 44, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71},
    {3, 4, 5, 6, 7, 8, 42, 43, 44, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71},
    {57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
    {57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
    {57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
    {57, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {57, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {57, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {0, 1, 2, 27, 28, 29, 30, 31, 32, 39, 40, 41, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {0, 1, 2, 27, 28, 29, 30, 31, 32, 39, 40, 41, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {0, 1, 2, 27, 28, 29, 30, 31, 32, 39, 40, 41, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {30, 31, 32, 60, 61, 62, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {30, 31, 32, 60, 61, 62, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {30, 31, 32, 60, 61, 62, 72, 73, 74, 75, 76, 77, 78, 79, 80}
   };

  TEST_EQUALITY(gold_columnIndicesLength, int(columnIndicesHost.size()));

  int rowMapSize = gold_rowMap.size();
  for(int i=1; i<rowMapSize; i++){
    int begin = rowMapHost(i-1), end = rowMapHost(i);
    int size = gold_columnIndices[i-1].size();
    TEST_EQUALITY(end-begin, size);
    if( verbose_tests ) std::cout << "end-begin == gold_columnIndices[i-1].size(): " << end-begin 
                                  << " == " << gold_columnIndices[i-1].size() << std::endl;
    for(int j=begin; j<end; j++){
      TEST_EQUALITY(count(gold_columnIndices[i-1].begin(),gold_columnIndices[i-1].end(),columnIndicesHost(j)), 1);
      if( verbose_tests ){
        std::cout << columnIndicesHost(j) << " is in { ";
        for( auto id : gold_columnIndices[i-1] ) std::cout << id << " ";
        std::cout << "}" << std::endl;
      }
    }
  }
}

/******************************************************************************/
/*! Test the CreateBlockMatrix function.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create a CrsMatrix with 'spaceDim' dofs per node, i.e., the second template
       parameter is 'spaceDim'.

  Tests:
   1.  Check the rowMap. The rowMap object stores the offsets into the columnIndices for 
       each row. The offsets in gold_rowMap were computed with doc/gold/graph.py for the 
       2x2x2 mesh returned by getBoxMesh(3, 2).
   2.  Check the columnIndices.  The columnIndices object is a 1D list of the column ids 
       for all rows.  The rowMap provides offsets into this list for a given row.  The 
       values in gold_columnIndices were computed with graph.py for the 2x2x2 mesh returned
       by getBoxMesh(3, 2).  Note that the order of the ordinals in the columnIndices 
       list is not the same as the order in the gold_columnIndices list.  The test 
       checks to see that every ordinal in columnIndices occurs once in the gold so
       order isn't important.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_BlockGraph3D )
{
 // Test setup

  const int spaceDim =3;
  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto matrix = Plato::CreateBlockMatrix<Plato::CrsMatrixType,spaceDim>(meshOmegaH.get());

  if(0)
    PlatoUtestHelpers::writeMesh(meshOmegaH, "GraphTest", spaceDim);

  if(0)
    PlatoUtestHelpers::writeConnectivity(meshOmegaH, "mesh3D.connectivity", spaceDim);

  auto rowMap = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();

 // Define tests

  // Test 1: (see description above)

  // the rowMap is on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    rowMapHost = Kokkos::create_mirror_view( rowMap );
  Kokkos::deep_copy(rowMapHost, rowMap);

  std::vector<int> 
    gold_rowMap = {0, 8, 17, 22, 33, 40, 45, 52, 57, 66, 77, 84, 89, 96, 111, 122, 131, 142, 151, 159, 168, 179, 190, 197, 202, 209, 218, 223};
                   
  TEST_EQUALITY(rowMapHost.size(), gold_rowMap.size());

  int numEntries = gold_rowMap.size();
  for( int i=0; i<numEntries; i++){
    TEST_EQUALITY(rowMapHost(i), gold_rowMap[i]);
    if( verbose_tests ){
      std::cout << "rowMapHost(i) == gold_rowMap[i]: " << rowMapHost(i) << " == " << gold_rowMap[i] << std::endl;
    }
  }


  // Test 2: (see description above)

  // the columnIndices are on the device, pull it to the host
  typename Plato::LocalOrdinalVector::HostMirror 
    columnIndicesHost = Kokkos::create_mirror_view( columnIndices );
  Kokkos::deep_copy(columnIndicesHost, columnIndices);

  int gold_columnIndicesLength = gold_rowMap.back();
  std::vector<std::vector<int>> 
   gold_columnIndices = {
    {0, 1, 3, 8, 9, 13, 21, 25},
    {0, 1, 2, 3, 4, 13, 14, 21, 22},
    {1, 2, 4, 14, 22},
    {0, 1, 3, 4, 5, 6, 8, 13, 14, 15, 16},
    {1, 2, 3, 4, 5, 14, 15},
    {3, 4, 5, 6, 15},
    {3, 5, 6, 7, 8, 15, 16},
    {6, 7, 8, 12, 16},
    {0, 3, 6, 7, 8, 9, 12, 13, 16},
    {0, 8, 9, 10, 11, 12, 13, 16, 17, 20, 25},
    {9, 10, 11, 17, 20, 25, 26},
    {9, 10, 11, 12, 17},
    {7, 8, 9, 11, 12, 16, 17},
    {0, 1, 3, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25},
    {1, 2, 3, 4, 13, 14, 15, 18, 19, 21, 22},
    {3, 4, 5, 6, 13, 14, 15, 16, 18},
    {3, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18},
    {9, 10, 11, 12, 13, 16, 17, 18, 20},
    {13, 14, 15, 16, 17, 18, 19, 20},
    {13, 14, 18, 19, 20, 21, 22, 23, 24},
    {9, 10, 13, 17, 18, 19, 20, 21, 24, 25, 26},
    {0, 1, 13, 14, 19, 20, 21, 22, 23, 24, 25},
    {1, 2, 14, 19, 21, 22, 23},
    {19, 21, 22, 23, 24},
    {19, 20, 21, 23, 24, 25, 26},
    {0, 9, 10, 13, 20, 21, 24, 25, 26},
    {10, 20, 24, 25, 26}
   };

  TEST_EQUALITY(gold_columnIndicesLength, int(columnIndicesHost.size()));

  int rowMapSize = gold_rowMap.size();
  for(int i=1; i<rowMapSize; i++){
    int begin = rowMapHost(i-1), end = rowMapHost(i);
    int size = gold_columnIndices[i-1].size();
    TEST_EQUALITY(end-begin, size);
    if( verbose_tests ) std::cout << "end-begin == gold_columnIndices[i-1].size(): " << end-begin 
                                  << " == " << gold_columnIndices[i-1].size() << std::endl;
    for(int j=begin; j<end; j++){
      TEST_EQUALITY(count(gold_columnIndices[i-1].begin(),gold_columnIndices[i-1].end(),columnIndicesHost(j)), 1);
      if( verbose_tests ){
        std::cout << columnIndicesHost(j) << " is in { ";
        for( auto id : gold_columnIndices[i-1] ) std::cout << id << " ";
        std::cout << "}" << std::endl;
      }
    }
  }
}



/******************************************************************************/
/*! Test the system assembly functionality in 2D.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Assemble.

  Tests:
   1.  Check the first two rows of the global matrix.  The entries list stores the
       values that correspond to the columnIndices. The entries in gold_entries were 
       determined in a mathematica notebook (doc/gold/2D_FEM_tri.nb) for the 2x2 mesh 
       returned getBoxMesh(2, 2).  Note that the ordering in the notebook is 
       different from the order in 'gold_entries' for the test comparison.  The 
       values have been reordered to match the values in LGR to simplify comparison.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_ScalarAssemble2D )
{
 // Test setup

  const int spaceDim = 2;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *params);
  paramList.set<bool>("Use Block Matrix",false);
  Plato::LinearElastostatics::ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();
  solver.assemble();

 // Define tests
  
  // Test 1: (see description above)

  // the rowMap is on the device, pull it to the host
  auto entries = solver.getMatrix().entries();
  typename Plato::ScalarVector::HostMirror 
    entriesHost = Kokkos::create_mirror_view( entries );
  Kokkos::deep_copy(entriesHost, entries);

  std::vector<Plato::Scalar> 
    gold_entries = {
    8.65384615384615259e+05, 0.00000000000000000e+00,
   -6.73076923076923005e+05, 2.88461538461538439e+05,
    0.00000000000000000e+00, -4.80769230769230751e+05,
   -1.92307692307692312e+05, 1.92307692307692312e+05,
    0.00000000000000000e+00, 8.65384615384615259e+05,
    1.92307692307692312e+05, -1.92307692307692312e+05,
   -4.80769230769230751e+05, 0.00000000000000000e+00,
    2.88461538461538439e+05, -6.73076923076923005e+05};

  int entriesSize = gold_entries.size();
  for(int i=0; i<entriesSize; i++){
    TEST_FLOATING_EQUALITY(entriesHost(i), gold_entries[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(i) == gold_entries[i]: " << entriesHost(i) << " == " << gold_entries[i] << std::endl;
  }
}

/******************************************************************************/
/*! Test the system assembly functionality in 2D.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Assemble.

  Tests:
   1.  Check the first two rows of the global matrix.  The entries list stores the
       values that correspond to the columnIndices. The entries in gold_entries were 
       determined in a mathematica notebook (doc/gold/2D_FEM_tri.nb) for the 2x2 mesh 
       returned getBoxMesh(2, 2).  Note that the ordering in the notebook is 
       different from the order in 'gold_entries' for the test comparison.  The 
       values have been reordered to match the values in LGR to simplify comparison.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_BlockAssemble2D )
{
 // Test setup

  const int spaceDim = 2;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *params);
  paramList.set<bool>("Use Block Matrix",true);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();
  solver.assemble();

 // Define tests
  
  // Test 1: (see description above)

  // the rowMap is on the device, pull it to the host
  auto entries = solver.getMatrix().entries();
  typename Plato::ScalarVector::HostMirror 
    entriesHost = Kokkos::create_mirror_view( entries );
  Kokkos::deep_copy(entriesHost, entries);

  std::vector<Plato::Scalar> 
    gold_entries = {
    8.65384615384615259e+05, 0.00000000000000000e+00,
    0.00000000000000000e+00, 8.65384615384615259e+05, 
   -6.73076923076923005e+05, 2.88461538461538439e+05,
    1.92307692307692312e+05,-1.92307692307692312e+05, 
    0.00000000000000000e+00,-4.80769230769230751e+05,
   -4.80769230769230751e+05, 0.00000000000000000e+00,
   -1.92307692307692312e+05, 1.92307692307692312e+05, 
    2.88461538461538439e+05,-6.73076923076923005e+05 };

  int entriesSize = gold_entries.size();
  for(int i=0; i<entriesSize; i++){
    TEST_FLOATING_EQUALITY(entriesHost(i), gold_entries[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(i) == gold_entries[i]: " << entriesHost(i) << " == " << gold_entries[i] << std::endl;
  }
}



/******************************************************************************/
/*! Test the system assembly functionality in 3D.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Assemble.

  Tests:
   1.  Check the first two rows of the global matrix.  The entries list stores the
       values that correspond to the columnIndices. The entries in gold_entries were 
       determined in a mathematica notebook (doc/gold/3D_FEM_tet4.nb) for the 2x2x2 mesh 
       returned getBoxMesh(3, 2).  Note that the ordering in the notebook is 
       different from the order in 'gold_entries' for the test comparison.  The 
       values have been reordered to match the values in LGR to simplify comparison.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_ScalarAssemble3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *params);
  paramList.set<bool>("Use Block Matrix",false);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();
  solver.assemble();

 // Define tests

 // the entries list stores the values that correspond to the columnIndices. The
 // entries in gold_entries were determined by hand for the 2x2x2 mesh returned
 // by getBoxMesh(/*spaceDim=*/ 3, /*meshWidth=*/ 2).

  // the entries are on the device, pull them to the host
  auto entries = solver.getMatrix().entries();
  typename Plato::ScalarVector::HostMirror 
    entriesHost = Kokkos::create_mirror_view( entries );
  Kokkos::deep_copy(entriesHost, entries);

  std::vector<Plato::Scalar> 
    gold_entries = {
    3.52564102564102504e+05, 0.00000000000000000e+00, 0.00000000000000000e+00,
   -6.41025641025641016e+04, 3.20512820512820508e+04, 0.00000000000000000e+00,
   -6.41025641025641016e+04, 0.00000000000000000e+00, 3.20512820512820508e+04,
    0.00000000000000000e+00, 3.20512820512820508e+04, 3.20512820512820508e+04,
    0.00000000000000000e+00, -8.01282051282051252e+04, 4.80769230769230708e+04,
    0.00000000000000000e+00, -8.01282051282051252e+04, -8.01282051282051252e+04,
    0.00000000000000000e+00, 4.80769230769230708e+04, -8.01282051282051252e+04,
   -2.24358974358974316e+05, 4.80769230769230708e+04, 4.80769230769230708e+04,
    0.00000000000000000e+00, 3.52564102564102563e+05, 0.00000000000000000e+00,
    4.80769230769230708e+04, -2.24358974358974316e+05, 4.80769230769230708e+04,
    0.00000000000000000e+00, -6.41025641025641016e+04, 3.20512820512820508e+04,
    4.80769230769230708e+04, 0.00000000000000000e+00, -8.01282051282051252e+04,
   -8.01282051282051252e+04, 0.00000000000000000e+00, 4.80769230769230708e+04,
   -8.01282051282051252e+04, 0.00000000000000000e+00, -8.01282051282051252e+04,
    3.20512820512820508e+04, 0.00000000000000000e+00, 3.20512820512820508e+04,
    3.20512820512820508e+04, -6.41025641025641016e+04, 0.00000000000000000e+00
    };

  int entriesSize = gold_entries.size();
  for(int i=0; i<entriesSize; i++){
    TEST_FLOATING_EQUALITY(entriesHost(i), gold_entries[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(i) == gold_entries[i]: " << entriesHost(i) << " == " << gold_entries[i] << std::endl;
  }

}

/******************************************************************************/
/*! Test the system assembly functionality in 3D.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Assemble.

  Tests:
   1.  Check the first two rows of the global matrix.  The entries in gold_entries were 
       determined in a mathematica notebook (doc/gold/3D_FEM_tet4.nb) for the 2x2x2 mesh 
       returned getBoxMesh(3, 2). 
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_BlockAssemble3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *params);
  paramList.set<bool>("Use Block Matrix",true);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();
  solver.assemble();

 // Define tests

 // the entries list stores the values that correspond to the columnIndices. The
 // entries in gold_entries were determined by hand for the 2x2x2 mesh returned
 // by getBoxMesh(/*spaceDim=*/ 3, /*meshWidth=*/ 2).

  // the entries are on the device, pull them to the host
  auto entries = solver.getMatrix().entries();
  typename Plato::ScalarVector::HostMirror 
    entriesHost = Kokkos::create_mirror_view( entries );
  Kokkos::deep_copy(entriesHost, entries);

  std::vector<Plato::Scalar> 
    gold_entries = {
    3.52564102564102504e+05, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 3.52564102564102563e+05, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 3.52564102564102563e+05, 

   -6.41025641025641016e+04, 3.20512820512820508e+04, 0.00000000000000000e+00,
    4.80769230769230708e+04,-2.24358974358974316e+05, 4.80769230769230708e+04,
    0.00000000000000000e+00, 3.20512820512820508e+04,-6.41025641025641016e+04, 

   -6.41025641025641016e+04, 0.00000000000000000e+00, 3.20512820512820508e+04,
    0.00000000000000000e+00,-6.41025641025641016e+04, 3.20512820512820508e+04, 
    4.80769230769230708e+04, 4.80769230769230708e+04,-2.24358974358974316e+05,

    0.00000000000000000e+00, 3.20512820512820508e+04, 3.20512820512820508e+04,
    4.80769230769230708e+04, 0.00000000000000000e+00, -8.01282051282051252e+04,
    4.80769230769230708e+04,-8.01282051282051252e+04, 0.00000000000000000e+00, 

    0.00000000000000000e+00,-8.01282051282051252e+04, 4.80769230769230708e+04,
   -8.01282051282051252e+04, 0.00000000000000000e+00, 4.80769230769230708e+04, 
    3.20512820512820508e+04, 3.20512820512820508e+04, 0.00000000000000000e+00, 
    };

  int entriesSize = gold_entries.size();
  for(int i=0; i<entriesSize; i++){
    TEST_FLOATING_EQUALITY(entriesHost(i), gold_entries[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(i) == gold_entries[i]: " << entriesHost(i) << " == " << gold_entries[i] << std::endl;
  }

}



/******************************************************************************/
/*! Test the system constraint functionality in 3D.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Assemble.

  Tests:
   1.  Check two rows of the global matrix.  The entries in gold_entries were 
       determined in a mathematica notebook (doc/gold/3D_FEM_tet4.nb) for the 2x2x2 mesh 
       returned getBoxMesh(3, 2). 
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_ScalarConstrained3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *params);
  paramList.set<bool>("Use Block Matrix",false);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();

  // constraint x=0 face
  Omega_h::LOs x0_ordinals = PlatoUtestHelpers::getBoundaryNodes_x0(meshOmegaH);
  auto numConstrainedDofs = spaceDim*x0_ordinals.size();
  auto& bcOrdinals = solver.getConstrainedDofs();
  auto& bcValues   = solver.getConstrainedValues();
  Kokkos::resize(bcOrdinals, numConstrainedDofs);
  Kokkos::resize(bcValues,   numConstrainedDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,x0_ordinals.size()), LAMBDA_EXPRESSION(int x0_ordinal)
  {
    auto offset = x0_ordinal * spaceDim;
    for (int iDim=0; iDim<spaceDim; iDim++)
    {
      bcOrdinals[offset+iDim] = spaceDim*x0_ordinals[x0_ordinal]+iDim;
      bcValues  [offset+iDim] = 0.0;
    }
  },"Dirichlet BC");

  solver.assemble();

//  #define WRITE_MATRIX
  #ifdef WRITE_MATRIX
  MatrixIO<Plato::Scalar,Plato::OrdinalType, int, Kokkos::LayoutRight, 
    Kokkos::DefaultExecutionSpace>::writeSparseMatlabMatrix(std::cout, solver.getMatrix());
  #endif

 // Define tests

  // Test 1: (see description above)

  // the entries are on the device, pull them to the host
  auto entries = solver.getMatrix().entries();
  typename Plato::ScalarVector::HostMirror 
    entriesHost = Kokkos::create_mirror_view( entries );
  Kokkos::deep_copy(entriesHost, entries);

  // there are 24 values in the first row:
  std::vector<Plato::Scalar> 
    gold_entries_first_row = {
    1.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
    };

  int rowSize = gold_entries_first_row.size();
  for(int i=0; i<rowSize; i++){
    TEST_FLOATING_EQUALITY(entriesHost(i), gold_entries_first_row[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(i) == gold_entries_first_row[i]: "
                                << entriesHost(i) << " == " 
                                << gold_entries_first_row[i] << std::endl;
  }

  // there are 33 values in row 27:
  const int testRow = 27;
  std::vector<Plato::Scalar> 
    gold_entries_row_27 = {
     0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
     0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
     1.05769230769230728e+06, -1.60256410256410250e+05, -1.60256410256410250e+05,
    -9.61538461538461561e+04, 8.01282051282051252e+04, -4.80769230769230708e+04,
    -1.92307692307692312e+05, -8.01282051282051252e+04, 1.60256410256410250e+05,
     0.00000000000000000e+00, 8.01282051282051252e+04, 8.01282051282051252e+04,
    -3.36538461538461503e+05, 8.01282051282051252e+04, 9.61538461538461415e+04,
     0.00000000000000000e+00, -8.01282051282051252e+04, 4.80769230769230708e+04,
     0.00000000000000000e+00, -8.01282051282051252e+04, -8.01282051282051252e+04,
     0.00000000000000000e+00, 8.01282051282051252e+04, -1.60256410256410250e+05,
    -9.61538461538461561e+04, 8.01282051282051252e+04, -3.20512820512820508e+04
    };

  // the rowMap is on the device, pull it to the host
  auto rowMap = solver.getMatrix().rowMap();
  typename Plato::LocalOrdinalVector::HostMirror 
    rowMapHost = Kokkos::create_mirror_view( rowMap );
  Kokkos::deep_copy(rowMapHost, rowMap);

  auto offset = rowMapHost(testRow);

  int row27Size = gold_entries_row_27.size();
  for(int i=0; i<row27Size; i++){
    TEST_FLOATING_EQUALITY(entriesHost(offset+i), gold_entries_row_27[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(offset+i) == gold_entries_row_27[i]: "
                                << entriesHost(offset+i) << " == " 
                                << gold_entries_row_27[i] << std::endl;
  }
}

/******************************************************************************/
/*! Test the system constraint functionality in 3D for a block matrix.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Assemble.

  Tests:
   1.  Check the first block row of the global matrix.  The entries in gold_entries were 
       determined in a mathematica notebook (doc/gold/3D_FEM_tet4.nb) for the 2x2x2 mesh 
       returned getBoxMesh(3, 2). 
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_BlockConstrained3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *params);
  paramList.set<bool>("Use Block Matrix",true);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();

  // constraint x=0 face
  Omega_h::LOs x0_ordinals = PlatoUtestHelpers::getBoundaryNodes_x0(meshOmegaH);
  auto numConstrainedDofs = spaceDim*x0_ordinals.size();
  auto& bcOrdinals = solver.getConstrainedDofs();
  auto& bcValues   = solver.getConstrainedValues();
  Kokkos::resize(bcOrdinals, numConstrainedDofs);
  Kokkos::resize(bcValues,   numConstrainedDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,x0_ordinals.size()), LAMBDA_EXPRESSION(int x0_ordinal)
  {
    auto offset = x0_ordinal * spaceDim;
    for (int iDim=0; iDim<spaceDim; iDim++)
    {
      bcOrdinals[offset+iDim] = spaceDim*x0_ordinals[x0_ordinal]+iDim;
      bcValues  [offset+iDim] = 0.0;
    }
  },"Dirichlet BC");

  solver.assemble();

//  #define WRITE_MATRIX
  #ifdef WRITE_MATRIX
  MatrixIO<Plato::Scalar,Plato::OrdinalType, int, Kokkos::LayoutRight, 
    Kokkos::DefaultExecutionSpace>::writeSparseMatlabMatrix(std::cout, solver.getMatrix());
  #endif

 // Define tests

  // Test 1: (see description above)

  // the entries are on the device, pull them to the host
  auto entries = solver.getMatrix().entries();
  typename Plato::ScalarVector::HostMirror 
    entriesHost = Kokkos::create_mirror_view( entries );
  Kokkos::deep_copy(entriesHost, entries);

  // There are 8 nodes in the first block row with 9 values per block:
  std::vector<Plato::Scalar> 
    gold_entries_first_row = {
    1.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 1.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 1.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
    };

  int rowSize = gold_entries_first_row.size();
  for(int i=0; i<rowSize; i++){
    TEST_FLOATING_EQUALITY(entriesHost(i), gold_entries_first_row[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(i) == gold_entries_first_row[i]: "
                                << entriesHost(i) << " == " 
                                << gold_entries_first_row[i] << std::endl;
  }

  // There are 5 nodes in block row 26 with 9 values per block:
  const int testRow = 26;
  std::vector<Plato::Scalar> 
    gold_entries_row_26 = {
    -32051.28205128205,  48076.92307692306, -48076.92307692306,
     32051.28205128205, -112179.4871794871,  48076.92307692306,
    -32051.28205128205,  32051.28205128205, -32051.28205128205,
    -224358.9743589743,  32051.28205128205,  32051.28205128205,
     48076.92307692306, -64102.56410256409,  0.000000000000000,
     48076.92307692306,  0.000000000000000, -64102.56410256409,
    -32051.28205128205, -48076.92307692306,  48076.92307692306,
    -32051.28205128205, -32051.28205128205,  32051.28205128205,
     32051.28205128205,  48076.92307692306, -112179.4871794871,
     0.000000000000000,  48076.92307692306,  48076.92307692306,
     32051.28205128205,  0.000000000000000, -80128.20512820511,
     32051.28205128205, -80128.20512820511,  0.000000000000000,
     288461.5384615384, -80128.20512820510, -80128.20512820510,
    -80128.20512820510,  208333.3333333333,  0.000000000000000,
    -80128.20512820510,  0.000000000000000,  208333.3333333333
    };

  // the rowMap is on the device, pull it to the host
  auto rowMap = solver.getMatrix().rowMap();
  typename Plato::LocalOrdinalVector::HostMirror 
    rowMapHost = Kokkos::create_mirror_view( rowMap );
  Kokkos::deep_copy(rowMapHost, rowMap);

  auto offset = rowMapHost(testRow)*spaceDim*spaceDim;

  int row26Size = gold_entries_row_26.size();
  for(int i=0; i<row26Size; i++){
    TEST_FLOATING_EQUALITY(entriesHost(offset+i), gold_entries_row_26[i], 1.0e-15);
    if(verbose_tests) std::cout << "entriesHost(offset+i) == gold_entries_row_26[i]: "
                                << entriesHost(offset+i) << " == " 
                                << gold_entries_row_26[i] << std::endl;
  }
}



/******************************************************************************/
/*! Test the calculation of the body force global vector in 3D.
  
  Test setup:
   1.  Create a box mesh in 'spaceDim' dimensions and 'meshWidth' intervals per side.
   2.  Create an ElastostaticsSolve object.
   3.  Initialize the object.
   4.  Define body forces.
   5.  Assemble.

  Tests:
   1.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_bodyLoad3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> modelParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *modelParams);
  paramList.set<bool>("Use Block Matrix",false);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());
  solver.initialize();

  // create BodyLoads object
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Body Loads'>                          \n"
    "  <ParameterList  name='Gravity Force'>                     \n"
    "    <Parameter  name='Function' type='string' value='1.0'/> \n"
    "    <Parameter  name='Index'    type='int'    value='0'/>   \n"
    "  </ParameterList>                                          \n"
    "</ParameterList>                                            \n"
   );

  Plato::BodyLoads<spaceDim> bl(*params);
  bl.get(*meshOmegaH, solver.getRHS());

  solver.assemble();

 // Define tests

  // the values are on the device, pull them to the host
  auto rhs = solver.getRHS();
  typename Plato::ScalarVector::HostMirror 
    rhsHost = Kokkos::create_mirror_view( rhs );
  Kokkos::deep_copy(rhsHost, rhs);

  std::vector<Plato::Scalar> 
    gold_rhs = {
    -0.03125000000000000, 0, 0, -0.04166666666666666, 0, 0, 
    -0.01041666666666667, 0, 0, -0.06250000000000000, 0, 0, 
    -0.02083333333333333, 0, 0, -0.01041666666666667, 0, 0,
    -0.02083333333333333, 0, 0, -0.01041666666666667, 0, 0, 
    -0.04166666666666666, 0, 0, -0.06250000000000000, 0, 0, 
    -0.02083333333333333, 0, 0, -0.01041666666666667, 0, 0, 
    -0.02083333333333333, 0, 0, -0.1249999999999999, 0, 0, 
    -0.06250000000000000, 0, 0, -0.04166666666666666, 0, 0, 
    -0.06250000000000000, 0, 0, -0.04166666666666666, 0, 0, 
    -0.03125000000000000, 0, 0, -0.04166666666666666, 0, 0, 
    -0.06250000000000000, 0, 0, -0.06250000000000000, 0, 0, 
    -0.02083333333333333, 0, 0, -0.01041666666666667, 0, 0, 
    -0.02083333333333333, 0, 0, -0.04166666666666666, 0, 0, 
    -0.01041666666666667, 0, 0
   };

  int rhsSize = gold_rhs.size();
  for(int i=0; i<rhsSize; i++){
    TEST_FLOATING_EQUALITY(rhsHost(i), gold_rhs[i], 1.0e-15);
    if(verbose_tests) std::cout << "rhsHost(i) == gold_rhs[i]: "
                                << rhsHost(i) << " == " 
                                << gold_rhs[i] << std::endl;
  }
}


/* Test Body ******************************************************************/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_ScalarSolve3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> modelParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *modelParams);
  paramList.set<bool>("Use Block Matrix",false);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();

  // create BodyLoads object
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Body Loads'>                          \n"
    "  <ParameterList  name='Gravity Force'>                     \n"
    "    <Parameter  name='Function' type='string' value='1.0'/> \n"
    "    <Parameter  name='Index'    type='int'    value='0'/>   \n"
    "  </ParameterList>                                          \n"
    "</ParameterList>                                            \n"
   );

  Plato::BodyLoads<spaceDim> bl(*params);
  bl.get(*meshOmegaH, solver.getRHS());

  Omega_h::LOs x0_ordinals = PlatoUtestHelpers::getBoundaryNodes_x0(meshOmegaH);

  Omega_h::Write<Omega_h::LO> bcOrdinals(spaceDim*x0_ordinals.size());
  Omega_h::Write<Plato::Scalar> bcValues(bcOrdinals.size());

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,x0_ordinals.size()), LAMBDA_EXPRESSION(int x0_ordinal)
  {
    auto offset = x0_ordinal * spaceDim;
    for (int iDim=0; iDim<spaceDim; iDim++)
    {
      bcOrdinals[offset+iDim] = spaceDim*x0_ordinals[x0_ordinal]+iDim;
      bcValues  [offset+iDim] = 0.0;
    }
  },"Dirichlet BC");

  solver.setBC(bcOrdinals, bcValues);


  solver.assemble();

  auto A = solver.getMatrix();
  auto b = solver.getRHS();
  auto x = solver.getLHS();

#ifdef HAVE_AMGX
  {
  Kokkos::deep_copy(x,0);
  double cgTol = 1e-10;
  int maxIters = 10000;
  AmgXLinearProblem problem(A, x, b, AmgXLinearProblem::configurationString(AmgXLinearProblem::PCG_NOPREC,cgTol,maxIters));
  problem.solve();
  }
#endif

  // copy into field
  auto lhs = solver.getLHS();
  const typename DefaultFields::geom_array_type disp(lgr::Displacement<DefaultFields>());
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,lhs.size()), LAMBDA_EXPRESSION(int dofOrdinal) {
    disp(dofOrdinal/spaceDim, dofOrdinal%spaceDim) = lhs(dofOrdinal);
  },"copy from LHS");

  Teuchos::RCP<Teuchos::ParameterList> tags_pl =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Tags'>                                   \n"
      "    <Parameter name='Node' type='Array(string)' value='{displacement,coordinates}'/>  \n"
      "    <Parameter name='Element' type='Array(string)' value='{}'/> \n"
      "</ParameterList> \n"
    );

  Omega_h::vtk::Writer writer = Omega_h::vtk::Writer("outfile.vtu", mesh.omega_h_mesh, mesh.omega_h_mesh->dim());
  auto tags = Omega_h::vtk::get_all_vtk_tags(mesh.omega_h_mesh,spaceDim);
  Omega_h::update_tag_set(&tags, mesh.omega_h_mesh->dim(), *tags_pl);
  fields->copyGeomToMesh("displacement",lgr::Displacement<DefaultFields>());
  writer.write(Omega_h::Real(1.0), tags);

  // the values are on the device, pull them to the host
  auto sol = solver.getLHS();
  typename Plato::ScalarVector::HostMirror
    solHost = Kokkos::create_mirror_view( sol );
  Kokkos::deep_copy(solHost, sol);

  std::vector<Plato::Scalar> 
    gold_sol = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    -3.085510190446572e-7, -3.196591149932395e-9, 
    -6.600159388272762e-8, -3.971229520724618e-7, 
     1.978800631773883e-8, -1.022279083607855e-10, 
    -4.345712955615874e-7,  4.552977173570470e-8, 
    -1.690309363198635e-8, -3.201387691185447e-7, 
     4.852663188358565e-8, -9.268618300371249e-8, 
    -3.081394080304532e-7,  1.988205054032027e-9, 
     1.988205054031894e-9, -3.091535272972523e-7, 
    -3.632750818645396e-9,  7.070195844600859e-8, 
    -3.140040726866635e-7,  7.587624577062060e-8, 
     7.587624577062061e-8, -3.091535272972523e-7, 
     7.070195844600873e-8, -3.632750818645396e-9, 
    -4.893375292207479e-7,  6.087070478251171e-8, 
     1.279206773441248e-8, -5.136667444718392e-7, 
     6.460145583053740e-8,  6.460145583053730e-8, 
    -4.893375292207477e-7,  1.279206773441262e-8, 
     6.087070478251140e-8, -4.514508495452884e-7, 
     2.320704327940846e-8,  2.320704327940809e-8, 
    -3.085510190446569e-7, -6.600159388272738e-8, 
    -3.196591149932636e-9, -3.201387691185445e-7, 
    -9.268618300371244e-8,  4.852663188358533e-8, 
    -4.345712955615871e-7, -1.690309363198616e-8, 
     4.552977173570419e-8, -3.971229520724614e-7, 
    -1.022279083603646e-10, 1.978800631773834e-8, 
    -2.928638201953787e-7, -4.816107885611144e-8, 
    -4.816107885611176e-8, -3.544541148194299e-7, 
     2.435057388401205e-9,  2.435057388400677e-9
   };

  int solSize = gold_sol.size();
  for(int i=0; i<solSize; i++){
    TEST_FLOATING_EQUALITY(solHost(i), gold_sol[i], 1.0e-8);
    if(verbose_tests) std::cout << "solHost(i) == gold_sol[i]: "
                                << solHost(i) << " == " 
                                << gold_sol[i] << std::endl;
  }
}

/* Test Body ******************************************************************/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticSolve, ElastostaticSolve_BlockSolve3D )
{
  const int spaceDim = 3;
  using DefaultFields = lgr::Fields<spaceDim>;

  const int meshWidth=2;
  auto meshOmegaH = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto mesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);
  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new DefaultFields(mesh, paramList));

  // create material model input
  //
  Teuchos::RCP<Teuchos::ParameterList> modelParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Isotropic Linear Elastic'>                   \n"
    "  <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
    "  <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/> \n"
    "</ParameterList>                                                   \n"
   );
  paramList.sublist("Material Model").set<Teuchos::ParameterList>("Isotropic Linear Elastic", *modelParams);
  paramList.set<bool>("Use Block Matrix",true);
  ElastostaticSolve<spaceDim> solver(paramList, fields, lgr::getCommMachine());

  solver.initialize();

  // create BodyLoads object
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList  name='Body Loads'>                          \n"
    "  <ParameterList  name='Gravity Force'>                     \n"
    "    <Parameter  name='Function' type='string' value='1.0'/> \n"
    "    <Parameter  name='Index'    type='int'    value='0'/>   \n"
    "  </ParameterList>                                          \n"
    "</ParameterList>                                            \n"
   );

  Plato::BodyLoads<spaceDim> bl(*params);
  bl.get(*meshOmegaH, solver.getRHS());

  Omega_h::LOs x0_ordinals = PlatoUtestHelpers::getBoundaryNodes_x0(meshOmegaH);

  Omega_h::Write<Omega_h::LO> bcOrdinals(spaceDim*x0_ordinals.size());
  Omega_h::Write<Plato::Scalar> bcValues(bcOrdinals.size());

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,x0_ordinals.size()), LAMBDA_EXPRESSION(int x0_ordinal)
  {
    auto offset = x0_ordinal * spaceDim;
    for (int iDim=0; iDim<spaceDim; iDim++)
    {
      bcOrdinals[offset+iDim] = spaceDim*x0_ordinals[x0_ordinal]+iDim;
      bcValues  [offset+iDim] = 0.0;
    }
  },"Dirichlet BC");

  solver.setBC(bcOrdinals, bcValues);

  solver.assemble();

#ifdef HAVE_AMGX
 // get linear solver
  typedef lgr::CrsLinearProblem<Plato::OrdinalType> LinearSolver;
  Teuchos::RCP<LinearSolver>
    linearSolver = solver.getDefaultSolver(/*cgTol=*/1e-15, /*cgMaxIters=*/10000);

  linearSolver->solve();
#endif

  // copy into field
  auto lhs = solver.getLHS();
  const typename DefaultFields::geom_array_type disp(lgr::Displacement<DefaultFields>());
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,lhs.size()), LAMBDA_EXPRESSION(int dofOrdinal) {
    disp(dofOrdinal/spaceDim, dofOrdinal%spaceDim) = lhs(dofOrdinal);
  },"copy from LHS");

  Teuchos::RCP<Teuchos::ParameterList> tags_pl =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Tags'>                                   \n"
      "    <Parameter name='Node' type='Array(string)' value='{displacement,coordinates}'/>  \n"
      "    <Parameter name='Element' type='Array(string)' value='{}'/> \n"
      "</ParameterList> \n"
    );

  Omega_h::vtk::Writer writer = Omega_h::vtk::Writer("outfile.vtu", mesh.omega_h_mesh, mesh.omega_h_mesh->dim());
  auto tags = Omega_h::vtk::get_all_vtk_tags(mesh.omega_h_mesh,spaceDim);
  Omega_h::update_tag_set(&tags, mesh.omega_h_mesh->dim(), *tags_pl);
  fields->copyGeomToMesh("displacement",lgr::Displacement<DefaultFields>());
  writer.write(Omega_h::Real(1.0), tags);

  // the values are on the device, pull them to the host
  auto sol = solver.getLHS();
  Plato::ScalarVector::HostMirror solHost = Kokkos::create_mirror_view( sol );
  Kokkos::deep_copy(solHost, sol);

  std::vector<Plato::Scalar> 
    gold_sol = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    -3.085510190446572e-7, -3.196591149932395e-9, 
    -6.600159388272762e-8, -3.971229520724618e-7, 
     1.978800631773883e-8, -1.022279083607855e-10, 
    -4.345712955615874e-7,  4.552977173570470e-8, 
    -1.690309363198635e-8, -3.201387691185447e-7, 
     4.852663188358565e-8, -9.268618300371249e-8, 
    -3.081394080304532e-7,  1.988205054032027e-9, 
     1.988205054031894e-9, -3.091535272972523e-7, 
    -3.632750818645396e-9,  7.070195844600859e-8, 
    -3.140040726866635e-7,  7.587624577062060e-8, 
     7.587624577062061e-8, -3.091535272972523e-7, 
     7.070195844600873e-8, -3.632750818645396e-9, 
    -4.893375292207479e-7,  6.087070478251171e-8, 
     1.279206773441248e-8, -5.136667444718392e-7, 
     6.460145583053740e-8,  6.460145583053730e-8, 
    -4.893375292207477e-7,  1.279206773441262e-8, 
     6.087070478251140e-8, -4.514508495452884e-7, 
     2.320704327940846e-8,  2.320704327940809e-8, 
    -3.085510190446569e-7, -6.600159388272738e-8, 
    -3.196591149932636e-9, -3.201387691185445e-7, 
    -9.268618300371244e-8,  4.852663188358533e-8, 
    -4.345712955615871e-7, -1.690309363198616e-8, 
     4.552977173570419e-8, -3.971229520724614e-7, 
    -1.022279083603646e-10, 1.978800631773834e-8, 
    -2.928638201953787e-7, -4.816107885611144e-8, 
    -4.816107885611176e-8, -3.544541148194299e-7, 
     2.435057388401205e-9,  2.435057388400677e-9
   };

  int solSize = gold_sol.size();
  for(int i=0; i<solSize; i++){
    // don't add tests that are nearly zero
    if(gold_sol[i] == 0.0){
      TEST_ASSERT(fabs(solHost(i)) < 1e-16);
    } else {
      TEST_FLOATING_EQUALITY(solHost(i), gold_sol[i], 1.0e-8);
    }
    if(verbose_tests) std::cout << "solHost(i) == gold_sol[i]: "
                                << solHost(i) << " == " 
                                << gold_sol[i] << std::endl;
  }
}
