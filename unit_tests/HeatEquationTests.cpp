/*!
  These unit tests are for the HeatEquation functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "LGRTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "FEMesh.hpp"
#include "MatrixIO.hpp"
#include "VizOutput.hpp"
#include "MeshFixture.hpp"
#include "StaticsTypes.hpp"

#include "ImplicitFunctors.hpp"
#include "LinearThermalMaterial.hpp"

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>
#include <CrsLinearProblem.hpp>
#include <Fields.hpp>
#include <ParallelComm.hpp>

#include <impl/Kokkos_Timer.hpp>

#include <plato/Simp.hpp>
#include <plato/ApplyWeighting.hpp>
#include <plato/ScalarProduct.hpp>
#include <plato/SimplexFadTypes.hpp>
#include <plato/WorksetBase.hpp>
#include <plato/VectorFunctionInc.hpp>
#include <plato/ScalarFunctionInc.hpp>
#include <plato/StateValues.hpp>
#include "ApplyConstraints.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/Thermal.hpp"

#include <fenv.h>


using namespace lgr;

TEUCHOS_UNIT_TEST( HeatEquationTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int numCells = mesh->nelems();
  int nodesPerCell  = SimplexThermal<spaceDim>::m_numNodesPerCell;
  int dofsPerCell   = SimplexThermal<spaceDim>::m_numDofsPerCell;
  int dofsPerNode   = SimplexThermal<spaceDim>::m_numDofsPerNode;


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( mesh->nverts() );
  Plato::Scalar Tval = 0.0, dval = 1.0;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);


  WorksetBase<SimplexThermal<spaceDim>> worksetBase(*mesh);

  Plato::ScalarArray3DT<Plato::Scalar>
    gradient("gradient",numCells,nodesPerCell,spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tGrad("temperature gradient", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tFlux("thermal flux", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    result("result", numCells, dofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar>
    configWS("config workset",numCells, nodesPerCell, spaceDim);
  worksetBase.worksetConfig(configWS);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stateWS("state workset",numCells, dofsPerCell);
  worksetBase.worksetState(T, stateWS);

  Plato::ComputeGradientWorkset<spaceDim> computeGradient;

  ScalarGrad<spaceDim> scalarGrad;

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <ParameterList name='Material Model'>                                        \n"
    "    <ParameterList name='Isotropic Linear Thermal'>                            \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>               \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e6'/>            \n"
    "      <Parameter name='Conductivity Coefficient' type='double' value='1.0e6'/> \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );


  lgr::ThermalModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create();
  auto cellConductivity = materialModel->getConductivityMatrix();
  auto cellDensity      = materialModel->getMassDensity();
  auto cellSpecificHeat = materialModel->getSpecificHeat();

  ThermalFlux<spaceDim>      thermalFlux(cellConductivity);
  FluxDivergence<spaceDim>  fluxDivergence;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    scalarGrad(cellOrdinal, tGrad, stateWS, gradient);

    thermalFlux(cellOrdinal, tFlux, tGrad);

    fluxDivergence(cellOrdinal, result, tFlux, gradient, cellVolume);
  }, "flux divergence");


  Plato::ScalarMultiVectorT<Plato::Scalar> 
   stateValues("Gauss point temperature at step k", numCells, dofsPerNode);

  Plato::ScalarMultiVectorT<Plato::Scalar> 
    thermalContent("Gauss point heat content at step k", numCells, dofsPerNode);

  Plato::ScalarMultiVectorT<Plato::Scalar> 
    massResult("mass", numCells, dofsPerCell);

  Plato::StateValues computeStateValues;
  ThermalContent     computeThermalContent(cellDensity, cellSpecificHeat);
  Plato::ComputeProjectionWorkset projectThermalContent;

  auto basisFunctions = cubatureRule.getBasisFunctions();

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeStateValues(cellOrdinal, basisFunctions, stateWS, stateValues);
    computeThermalContent(cellOrdinal, thermalContent, stateValues);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, thermalContent, massResult);

  }, "mass");

  
  // test cell volume
  //
  auto cellVolume_Host = Kokkos::create_mirror_view( cellVolume );
  Kokkos::deep_copy( cellVolume_Host, cellVolume );

  std::vector<double> cellVolume_gold = { 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333
  };

  int numGoldCells=cellVolume_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(cellVolume_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(cellVolume_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(cellVolume_Host(iCell), cellVolume_gold[iCell], 1e-13);
    }
  }

  // test state values
  //
  auto stateValues_Host = Kokkos::create_mirror_view( stateValues );
  Kokkos::deep_copy( stateValues_Host, stateValues );

  std::vector<std::vector<double>> stateValues_gold = { 
    { 8.5 },{ 7.0 },{ 5.25},{ 9.75},
    { 8.75},{ 6.5 },{ 6.25},{10.75},
    {11.00},{ 8.25},{ 7.75},{10.0 },
    {11.75},{10.25},{ 9.25},{11.0 }
  };

  numGoldCells=stateValues_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iDof=0; iDof<dofsPerNode; iDof++){
      if(stateValues_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(stateValues_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(stateValues_Host(iCell,iDof), stateValues_gold[iCell][iDof], 1e-13);
      }
    }
  }

  // test thermal content
  //
  auto thermalContent_Host = Kokkos::create_mirror_view( thermalContent );
  Kokkos::deep_copy( thermalContent_Host, thermalContent );

  std::vector<std::vector<double>> thermalContent_gold = { 
    { 2.550e6},{ 2.100e6},{ 1.575e6},{ 2.925e6},
    { 2.625e6},{ 1.950e6},{ 1.875e6},{ 3.225e6},
    { 3.300e6},{ 2.475e6},{ 2.325e6},{ 3.000e6},
    { 3.525e6},{ 3.075e6},{ 2.775e6},{ 3.300e6}
  };

  numGoldCells=thermalContent_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iDof=0; iDof<dofsPerNode; iDof++){
      if(thermalContent_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(thermalContent_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(thermalContent_Host(iCell,iDof), thermalContent_gold[iCell][iDof], 1e-13);
      }
    }
  }


  // test gradient operator
  //
  auto gradient_Host = Kokkos::create_mirror_view( gradient );
  Kokkos::deep_copy( gradient_Host, gradient );

  std::vector<std::vector<std::vector<double>>> gradient_gold = { 
    {{0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
    {{0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}},
    {{0.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0,-2.0, 2.0},{ 2.0, 0.0, 0.0}},
    {{0.0, 0.0,-2.0},{ 2.0,-2.0, 0.0},{ 0.0, 2.0, 0.0},{-2.0, 0.0, 2.0}},
    {{0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
    {{0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}}
  };

  numGoldCells=gradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<spaceDim+1; iNode++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        if(gradient_gold[iCell][iNode][iDim] == 0.0){
          TEST_ASSERT(fabs(gradient_Host(iCell,iNode,iDim)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(gradient_Host(iCell,iNode,iDim), gradient_gold[iCell][iNode][iDim], 1e-13);
        }
      }
    }
  }

  // test temperature gradient
  //
  auto tgrad_Host = Kokkos::create_mirror_view( tGrad );
  Kokkos::deep_copy( tgrad_Host, tGrad );

  std::vector<std::vector<double>> tgrad_gold = { 
    { 2.000 , 16.00, 8.000 },
    { 20.00 , 16.00,-10.00 },
    { 20.00 , 4.000, 2.000 },
    { 40.00 ,-16.00, 2.000 },
    { 20.00 , 4.000, 2.000 },
    { 20.00 , 4.000, 2.000 }
  };

  for(int iCell=0; iCell<int(tgrad_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tgrad_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tgrad_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tgrad_Host(iCell,iDim), tgrad_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test thermal flux
  //
  auto tflux_Host = Kokkos::create_mirror_view( tFlux );
  Kokkos::deep_copy( tflux_Host, tFlux );

  std::vector<std::vector<double>> tflux_gold = { 
   { 2.0e6, 1.6e7, 8.0e6 },
   { 2.0e7, 1.6e7,-1.0e7 },
   { 2.0e7, 4.0e6, 2.0e6 },
   { 4.0e7,-1.6e7, 2.0e6 },
   { 2.0e7, 4.0e6, 2.0e6 },
   { 2.0e7, 4.0e6, 2.0e6 }
  };

  for(int iCell=0; iCell<int(tflux_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tflux_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tflux_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tflux_Host(iCell,iDim), tflux_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test residual
  //
  auto result_Host = Kokkos::create_mirror_view( result );
  Kokkos::deep_copy( result_Host, result );

  std::vector<std::vector<double>> result_gold = { 
   { -666666.6666666666, -250000.0000000000,  583333.3333333333,  333333.3333333333 },
   { -666666.6666666666, 1083333.3333333333,-1250000.0000000000,  833333.3333333333 },
   {  -83333.3333333333, -666666.6666666666,  -83333.3333333333,  833333.3333333333 },
   {  -83333.3333333333, 2333333.3333333333, -666666.6666666666,-1583333.3333333333 },
   { -166666.6666666667,  750000.0000000000, -666666.6666666666,   83333.3333333333 },
   { -166666.6666666667,   83333.3333333333, -750000.0000000000,  833333.3333333333 }
  };

  for(int iCell=0; iCell<int(result_gold.size()); iCell++){
    for(int iDof=0; iDof<dofsPerCell; iDof++){
      if(result_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(result_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(result_Host(iCell,iDof), result_gold[iCell][iDof], 1e-13);
      }
    }
  }

  // test residual
  //
  auto mass_result_Host = Kokkos::create_mirror_view( massResult );
  Kokkos::deep_copy( mass_result_Host, massResult );

  std::vector<std::vector<double>> mass_result_gold = { 
   { 13281.25000000000, 13281.25000000000, 13281.25000000000, 13281.25000000000},
   { 10937.50000000000, 10937.50000000000, 10937.50000000000, 10937.50000000000},
   {  8203.12500000000,  8203.12500000000,  8203.12500000000,  8203.12500000000},
   { 15234.37500000000, 15234.37500000000, 15234.37500000000, 15234.37500000000},
   { 13671.87500000000, 13671.87500000000, 13671.87500000000, 13671.87500000000}
  };

  for(int iCell=0; iCell<int(mass_result_gold.size()); iCell++){
    for(int iNode=0; iNode<nodesPerCell; iNode++){
      if(mass_result_gold[iCell][iNode] == 0.0){
        TEST_ASSERT(fabs(mass_result_Host(iCell,iNode)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(mass_result_Host(iCell,iNode), mass_result_gold[iCell][iNode], 1e-13);
      }
    }
  }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, HeatEquationResidual3D )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( mesh->nverts(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( mesh->nverts() );
  Plato::Scalar Tval = 0.0, dval = 1.0000;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);

  std::vector<Plato::Scalar> Tprev_host( mesh->nverts() );
  Tval = 0.0; dval = 0.5000;
  for( auto& val : Tprev_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    Tprev_host_view(Tprev_host.data(),Tprev_host.size());
  auto Tprev = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), Tprev_host_view);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='Heat Equation'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                   \n"
    "  <ParameterList name='Heat Equation'>                                         \n"
    "    <ParameterList name='Penalty Function'>                                    \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                   \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                      \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Material Model'>                                        \n"
    "    <ParameterList name='Isotropic Linear Thermal'>                            \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>               \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e3'/>            \n"
    "      <Parameter name='Conductivity Coefficient' type='double' value='1.0e6'/> \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Time Integration'>                                      \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>                 \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                    \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  VectorFunctionInc<::Plato::Thermal<spaceDim>> 
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = params->sublist("Time Integration").get<double>("Time Step");
  auto residual = vectorFunction.value(T, Tprev, z, timeStep);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<double> residual_gold = { 
   -2.124953906250000e6, -2.062440625000000e6, -6.249867187500000e5,
   -2.531164257812500e6, -8.124761718750000e5, -6.249874999999999e5,
   -6.562197265625000e5, -1.874828125000000e5,  9.375644531250000e5,
   -2.531119726562500e6, -1.374948437500000e6, -3.749796874999999e5,
    9.379042968750000e4, -9.372355468750008e5, -1.873796874999993e5,
    4.375738281250000e5,  1.875119921875000e6,  1.250945312500003e5,
    1.875804687500000e5,  1.876246093749995e5,  2.250182421875000e6,
    2.625173828125000e6,  1.437553515625000e6,  2.500351562500002e5,
    6.250726562499998e5,  3.375119921875000e6,  6.250359375000002e5
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test gradient wrt state, T. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(T, Tprev, z, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
   1.250023437500000e5,  -4.166588541666666e4,  -4.166588541666666e4,
   7.812500000000000e-1,  7.812500000000000e-1,  2.343750000000000,
   7.812500000000000e-1, -4.166588541666666e4,  -4.166588541666666e4,
   2.083364583333333e5,  -6.249882812500000e4,  -4.166588541666666e4,
   7.812500000000000e-1,  1.562500000000000,     2.343750000000000,
  -6.249882812500000e4,   7.812500000000000e-1, -4.166588541666666e4,
   8.333411458333333e4,  -2.083294270833333e4,   7.812500000000000e-1,
  -2.083294270833333e4,   7.812500000000000e-1, -6.249882812500000e4,
   3.750046875000000e5,  -6.249882812500000e4,  -6.249882812500000e4,
   7.812500000000000e-1, -6.249882812500000e4,  -1.249976562500000e5,
   1.562500000000000,     2.343750000000000,     1.562500000000000
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt previous state, T. (i.e., jacobian)
  //
  auto jacobian_p = vectorFunction.gradient_p(T, Tprev, z, timeStep);

  auto jac_p_entries = jacobian_p->entries();
  auto jac_p_entriesHost = Kokkos::create_mirror_view( jac_p_entries );
  Kokkos::deep_copy(jac_p_entriesHost, jac_p_entries);

  std::vector<Plato::Scalar> gold_jac_p_entries = {
    1.249976562500000e5,  -4.166744791666666e4,  -4.166744791666666e4,
   -7.812500000000000e-1, -7.812500000000000e-1, -2.343750000000000,
   -7.812500000000000e-1, -4.166744791666666e4,  -4.166744791666666e4,
   2.083302083333333e5,   -6.250117187500000e4,  -4.166744791666666e4,
   -7.812500000000000e-1, -1.562500000000000,    -2.343750000000000,
   -6.250117187500000e4,  -7.812500000000000e-1, -4.166744791666666e4,
   8.333255208333333e4,   -2.083372395833333e4,  -7.812500000000000e-1,
   -2.083372395833333e4,  -7.812500000000000e-1, -6.250117187500000e4,
   3.749953125000000e5,   -6.250117187500000e4,  -6.250117187500000e4,
   -7.812500000000000e-1, -6.250117187500000e4,  -1.250023437500000e5,
   -1.562500000000000,    -2.343750000000000,    -1.562500000000000
  };

  int jac_p_entriesSize = gold_jac_p_entries.size();
  for(int i=0; i<jac_p_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_p_entriesHost(i), gold_jac_p_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = vectorFunction.gradient_z(T, Tprev, z, timeStep);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
    -5.31238476562500000e5,  1.56253027343750000e5, -1.56247070312500000e5, 
    -1.79685107421875000e5, -1.79683349609375000e5,  9.37615234374999854e4, 
     2.50004980468750000e5,  5.46880566406250000e5, -1.56220703125000000e4, 
    -5.15610156250000000e5, -1.17183496093749985e5, -1.56246679687500000e5, 
    -1.32810009765625000e5,  1.56322265624999709e4,  4.68869140625000437e4, 
     5.85945019531250000e5,  2.89067626953124942e5, -1.56216796875000018e4, 
    -1.56246679687500000e5, -6.24987792968750073e4,  1.56283203125000146e4, 
     2.18752099609374971e5, -7.03101074218750000e4, -3.90584960937500000e4, 
    -6.32791064453125000e5, -7.81194335937500000e4, -1.40620263671875000e5, 
    -1.56246874999999971e5,  1.79692822265625000e5,  4.06261132812499884e5, 
     1.56257324218750058e5,  1.56262109375000000e5,  2.18758593749999971e5
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = vectorFunction.gradient_x(T, Tprev, z, timeStep);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<lgr::Scalar> gold_grad_x_entries = {
    -5.687544531250000e6,  8.749757812500000e5, -6.252343749999994e4, 
     1.875000000000000e5,  7.499757812499998e5, -5.625000000000002e5, 
     1.812500000000000e6, -4.375000000000001e5,  1.499976562500000e6, 
    -5.625000000000000e5,  9.999890625000000e5,  9.999917968749999e5, 
     1.749980078125000e6,  1.749986718750000e6, -2.062500000000000e6, 
     8.124554687500000e5,  7.499757812500000e5,  1.937476562500000e6, 
    -1.875246093750001e5, -1.625000000000000e6, -1.875152343750000e5, 
     1.874955468749999e6, -3.062500000000000e6, -1.562500000000000e6, 
     1.937484765625000e6, -3.125082031250000e5, -1.249976562500000e6, 
    -9.375060156250000e6,  1.874967968750000e6, -6.250031250000001e5, 
     1.687500000000000e6,  1.624967968750000e6, -4.374917968749998e5, 
     1.812500000000000e6, -4.374999999999999e5,  1.499973437500000e6
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}


#ifdef NOPE
/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, InternalElasticEnergy3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( mesh->nverts(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( spaceDim*mesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);


  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elastostatics'/>      \n"
    "  <Parameter name='Objective' type='string' value='Internal Elastic Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Internal Elastic Energy'>                              \n"
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
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunction<::Plato::Thermal<spaceDim>> 
    eeScalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));


  // compute and test objective value
  //
  auto value = eeScalarFunction.value(u,z);

  double value_gold = 92.25;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(u,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(u,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
    7.111298076923077 ,  7.370192307692310 ,  1.842548076923078,
    6.053365384615385 ,  2.024278846153846 ,  0.9778846153846154,
    1.736538461538462 ,  0.5423076923076925,  2.889663461538463,
    6.010096153846159 ,  2.185817307692309 ,  0.4125000000000006,
    0.7644230769230780, 13.13076923076924  ,  7.064423076923083,
    3.088701923076925 ,  3.189663461538465 ,  1.626201923076927,
    0.6555288461538494,  1.089663461538468 ,  4.197115384615394,
    7.050721153846164 ,  2.869471153846158 ,  0.1096153846153868,
    0.3512019230769271,  6.998076923076927 ,  0.9079326923076940
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(u,z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   48.87692307692307, -31.34423076923076, -14.57307692307692,
   45.90865384615384, -35.71442307692308, 4.534615384615376,
    3.014423076923072, -9.447115384615387, 15.43269230769231,
   23.52403846153845, 15.37211538461538, -5.942307692307695,
   -3.351923076923076, 1.900961538461539, 12.83942307692307,
   -0.8999999999999964, 5.296153846153845, 3.911538461538462,
   -2.642307692307689, 11.86153846153845, -2.988461538461537,
   -0.6115384615384627, 4.130769230769227, -1.857692307692306,
   -6.700961538461539, 1.217307692307693, -8.777884615384618,
   22.14230769230767, -2.789423076923075, -9.591346153846143,
   12.47019230769231, -15.36057692307691, -0.6749999999999896,
    1.650000000000000, 2.480769230769230, -0.8076923076923080,
   -0.03461538461538199, 4.292307692307702, -5.088461538461546,
   -9.072115384615396, 11.18653846153847, 9.931730769230770,
  -21.82211538461539, 10.33557692307692, 19.53461538461539,
  -20.48653846153843, 3.320192307692308, 8.443269230769241,
  -16.67019230769229, 4.609615384615382, -3.516346153846154,
    2.720192307692316, 3.412500000000011, -7.655769230769235,
   -2.994230769230767, 1.298076923076930, 1.574999999999996,
    3.574038461538469, -1.817307692307682, 3.158653846153849,
    5.861538461538451, 7.947115384615393, 0.6086538461538544,
  -30.13269230769227, -14.57307692307693, 3.236538461538461,
  -17.05673076923077, -9.848076923076919, 10.87788461538462,
   -0.1153846153846152, 0.2307692307692237, 0.4384615384615397,
    0.1471153846153925, 0.4673076923076802, 1.704807692307704,
  -43.86923076923075, 28.24326923076922, -28.35288461538462,
    6.571153846153856, 3.291346153846139, -6.400961538461546
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}
#endif

