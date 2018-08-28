#include "plato/LGR_App.hpp"
#include <Teuchos_UnitTestHarness.hpp>

using namespace lgr;

TEUCHOS_UNIT_TEST( LGRAppTests, 3D )
{ 
 
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=soap.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "test_appfile.xml", true);

  MPMD_App app(argc, argv, myComm);
}
