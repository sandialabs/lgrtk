#include <cstdlib>

#include "LGRAppIntxTests.hpp"
#include <Teuchos_UnitTestHarness.hpp>

#include "plato/LGR_App.hpp"

TEUCHOS_UNIT_TEST( LGRAppTests, MultipleProblemDefinitions )
{ 
  /*
   * Two operations with different ProblemDefinitions on
   * one performer.
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=MultipleProblemDefinitions_input_1.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "MultipleProblemDefinitions_appfile.xml", true);

  MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<double> stdStateIn(localIDs.size(),0.5);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<double> stdStateOne(localIDs.size());
  std::vector<double> stdStateTwo(localIDs.size());

  // solve 1
  //
  app.compute("Compute Displacement Solution 1");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

  // solve 2
  //
  app.compute("Compute Displacement Solution 2");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateTwo);

  for(int i=0; i<localIDs.size(); i++)
  {
    if( fabs(stdStateOne[i]) > 1e-16 )
      TEST_FLOATING_EQUALITY(stdStateOne[i], -stdStateTwo[i], 1e-12);
  }
}

TEUCHOS_UNIT_TEST( LGRAppTests, OperationParameter )
{ 
  /*
   * One operation with a Parameter.
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=OperationParameter_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "OperationParameter_appfile.xml", true);

  MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<double> stdStateIn(localIDs.size(),0.5);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // set parameter
  //
  FauxParameter fauxParamIn("Traction X", "Compute Displacement Solution", 1.0);
  app.importDataT("Traction X", fauxParamIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<double> stdStateOne(localIDs.size());
  std::vector<double> stdStateTwo(localIDs.size());

  // solve 1
  //
  app.compute("Compute Displacement Solution");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

  // set parameter
  //
  std::vector<double> param(1,-1.0);
  fauxParamIn.setData(param);
  app.importDataT("Traction X", fauxParamIn);

  // solve 2
  //
  app.compute("Compute Displacement Solution");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateTwo);

  for(int i=0; i<localIDs.size(); i++)
  {
    if( fabs(stdStateOne[i]) > 1e-16 )
      TEST_FLOATING_EQUALITY(stdStateOne[i], -stdStateTwo[i], 1e-12);
  }
}

TEUCHOS_UNIT_TEST( LGRAppTests, CellForcing )
{ 
  /*
   * One operation with cell forcing
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=CellForcing_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "CellForcing_appfile.xml", true);

  MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<double> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<double> stdStateOne(localIDs.size());

  // solve
  //
  app.compute("Compute Displacement Solution");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

//  for(int i=0; i<localIDs.size(); i++)
//  {
//    if( fabs(stdStateOne[i]) > 1e-16 )
//      TEST_FLOATING_EQUALITY(stdStateOne[i], -stdStateTwo[i], 1e-12);
//  }
}

TEUCHOS_UNIT_TEST( LGRAppTests, EffectiveEnergy )
{ 
  /*
   * One operation with cell forcing
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=EffectiveEnergy_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "EffectiveEnergy_appfile.xml", true);

  MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<double> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedValue fauxStateOut(1,0.0);
  std::vector<double> stdStateOne(1);

  // solve
  //
  app.compute("Compute Objective Value");

  // export data
  //
  app.exportDataT("Objective Value", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

  TEST_FLOATING_EQUALITY(stdStateOne[0], 17308575.3656760529, 1e-12);
}

TEUCHOS_UNIT_TEST( LGRAppTests, InternalEnergyGradX )
{ 
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=InternalEnergyGradX_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "InternalEnergyGradX_appfile.xml", true);

  MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<double> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<double> stdStateOut(localIDs.size());

  // solve
  //
  app.compute("Compute ObjectiveX");

  // export data
  //
  app.exportDataT("GradientX X", fauxStateOut);
  fauxStateOut.getData(stdStateOut);

//  TEST_FLOATING_EQUALITY(stdStateOut[0], 17308575.3656760529, 1e-10);

  app.exportDataT("GradientX Y", fauxStateOut);
  fauxStateOut.getData(stdStateOut);

//  TEST_FLOATING_EQUALITY(stdStateOut[0], 17308575.3656760529, 1e-11);

  app.exportDataT("GradientX Z", fauxStateOut);
  fauxStateOut.getData(stdStateOut);

//  TEST_FLOATING_EQUALITY(stdStateOut[0], 17308575.3656760529, 1e-12);
}
