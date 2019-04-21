/*!
  These unit tests are for the TransientThermomech functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "plato/ImplicitFunctors.hpp"
#include "plato/LinearThermoelasticMaterial.hpp"

#ifdef HAVE_AMGX
#include "plato/alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>
#include <plato/alg/CrsLinearProblem.hpp>
#include <plato/alg/ParallelComm.hpp>
#include <plato/Simp.hpp>
#include <plato/ApplyWeighting.hpp>
#include <plato/SimplexFadTypes.hpp>
#include <plato/WorksetBase.hpp>
#include <plato/VectorFunctionInc.hpp>
#include <plato/ScalarFunctionInc.hpp>
#include <plato/StateValues.hpp>
#include "plato/ApplyConstraints.hpp"
#include "plato/SimplexThermal.hpp"
#include "plato/Thermomechanics.hpp"
#include "plato/ComputedField.hpp"

#include <fenv.h>


using namespace lgr;

TEUCHOS_UNIT_TEST( TransientThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  static constexpr int TDofOffset = spaceDim;

  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int numCells = mesh->nelems();
  constexpr int numVoigtTerms = Plato::SimplexThermomechanics<spaceDim>::m_numVoigtTerms;
  constexpr int nodesPerCell  = Plato::SimplexThermomechanics<spaceDim>::m_numNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexThermomechanics<spaceDim>::m_numDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexThermomechanics<spaceDim>::m_numDofsPerNode;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");

  WorksetBase<Plato::SimplexThermomechanics<spaceDim>> worksetBase(*mesh);

  Plato::ScalarArray3DT<Plato::Scalar>     gradient("gradient",numCells,nodesPerCell,spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> tStrain("strain", numCells, numVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tGrad("temperature gradient", numCells, spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> tStress("stress", numCells, numVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tFlux("thermal flux", numCells, spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> result("result", numCells, dofsPerCell);
  Plato::ScalarArray3DT<Plato::Scalar>     configWS("config workset",numCells, nodesPerCell, spaceDim);
  Plato::ScalarVectorT<Plato::Scalar>      tTemperature("Gauss point temperature", numCells);
  Plato::ScalarVectorT<Plato::Scalar>      tThermalContent("Gauss point heat content at step k", numCells);
  Plato::ScalarMultiVectorT<Plato::Scalar> massResult("mass", numCells, dofsPerCell);
  Plato::ScalarMultiVectorT<Plato::Scalar> stateWS("state workset",numCells, dofsPerCell);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, stateWS);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='1000.0'/>\n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  Plato::ThermoelasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create();
  auto cellDensity      = materialModel->getMassDensity();
  auto cellSpecificHeat = materialModel->getSpecificHeat();

  Plato::ComputeGradientWorkset<spaceDim>  computeGradient;
  TMKinematics<spaceDim>                   kinematics;
  TMKinetics<spaceDim>                     kinetics(materialModel);

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, TDofOffset> interpolateFromNodal;

  FluxDivergence  <spaceDim, dofsPerNode, TDofOffset> fluxDivergence;
  StressDivergence<spaceDim, dofsPerNode> stressDivergence;

  ThermalContent computeThermalContent(cellDensity, cellSpecificHeat);
  ProjectToNode<spaceDim, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();
  auto basisFunctions = cubatureRule.getBasisFunctions();

  Plato::Scalar tTimeStep = 1.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    kinematics(cellOrdinal, tStrain, tGrad, stateWS, gradient);

    interpolateFromNodal(cellOrdinal, basisFunctions, stateWS, tTemperature);

    kinetics(cellOrdinal, tStress, tFlux, tStrain, tGrad, tTemperature);

    stressDivergence(cellOrdinal, result, tStress, gradient, cellVolume, tTimeStep/2.0);

    fluxDivergence(cellOrdinal, result, tFlux, gradient, cellVolume, tTimeStep/2.0);

    computeThermalContent(cellOrdinal, tThermalContent, tTemperature);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, massResult);

  }, "divergence");

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
  auto tTemperature_Host = Kokkos::create_mirror_view( tTemperature );
  Kokkos::deep_copy( tTemperature_Host, tTemperature );

  std::vector<double> tTemperature_gold = { 
    2.9999999999999997e-06, 2.3999999999999999e-06,
    1.6999999999999998e-06, 3.4999999999999995e-06,
    3.1000000000000000e-06, 2.2000000000000001e-06,
    2.0999999999999998e-06, 3.8999999999999999e-06,
    3.9999999999999998e-06, 2.9000000000000002e-06,
    2.7000000000000000e-06, 3.5999999999999998e-06
  };

  numGoldCells=tTemperature_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tTemperature_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tTemperature_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tTemperature_Host(iCell), tTemperature_gold[iCell], 1e-13);
    }
  }

  // test thermal content
  //
  auto tThermalContent_Host = Kokkos::create_mirror_view( tThermalContent );
  Kokkos::deep_copy( tThermalContent_Host, tThermalContent );

  std::vector<double> tThermalContent_gold = { 
    0.90, 0.72, 0.51, 1.05, 0.93, 0.66, 0.63, 1.17, 1.20, 0.87, 0.81, 1.08 
  };

  numGoldCells=tThermalContent_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tThermalContent_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tThermalContent_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tThermalContent_Host(iCell), tThermalContent_gold[iCell], 1e-13);
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
    {8.0e-07,  6.4e-06,  3.2e-06}, 
    {8.0e-06,  6.4e-06, -4.0e-06},
    {8.0e-06,  1.6e-06,  8.0e-07},
    {1.6e-05, -6.4e-06,  8.0e-07}
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
   {8.0e-04,  6.4e-03,  3.2e-03}, 
   {8.0e-03,  6.4e-03, -4.0e-03},
   {8.0e-03,  1.6e-03,  8.0e-04},
   {1.6e-02, -6.4e-03,  8.0e-04}
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
   {-1602.56410256410254, -12099.2027243589710, -5128.20512820512704, -0.000133333333333333313,
     6169.71554487179492, -3525.64102564102450, -9695.35657051281487, -0.0000499999999999999685,
    -5688.94631410256352,  10496.6386217948711,  4006.41025641025590,  0.000116666666666666638,
     1121.79487179487114,  5128.20512820512704,  10817.1514423076878,  0.0000666666666666666428},
   {-4487.17948717948639, -7772.31089743589382, -2243.58974358974365, -0.000133333333333333313,
     480.769230769230489,  5528.72115384615336,  4407.17628205128221,  0.000216666666666666630,
    -1842.82371794871665, -2243.58974358974274, -6169.99679487179219, -0.000249999999999999951,
     5849.23397435897186,  4487.17948717948639,  4006.41025641025590,  0.000166666666666666634}
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
    {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896},
    {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986}
  };

  for(int iCell=0; iCell<int(mass_result_gold.size()); iCell++){
    for(int iDof=0; iDof<dofsPerCell; iDof++){
      if(mass_result_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(mass_result_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(mass_result_Host(iCell,iDof), mass_result_gold[iCell][iDof], 1e-13);
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
TEUCHOS_UNIT_TEST( TransientThermomechTests, TransientThermomechResidual3D )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector statePrev("prev state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;
     statePrev(aNodeOrdinal*tNumDofsPerNode+0) = (4e-7)*aNodeOrdinal;
     statePrev(aNodeOrdinal*tNumDofsPerNode+1) = (3e-7)*aNodeOrdinal;
     statePrev(aNodeOrdinal*tNumDofsPerNode+2) = (2e-7)*aNodeOrdinal;
     statePrev(aNodeOrdinal*tNumDofsPerNode+3) = (1e-7)*aNodeOrdinal;

  }, "state");

  // create mesh based density from host data
  //
//  std::vector<Plato::Scalar> z_host( mesh->nverts(), 1.0 );
//  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//    z_host_view(z_host.data(),z_host.size());
//  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based temperature from host data
  //
//  std::vector<Plato::Scalar> T_host( mesh->nverts() );
//  Plato::Scalar Tval = 0.0, dval = 1.0000;
//  for( auto& val : T_host ) val = (Tval += dval);
//  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//    T_host_view(T_host.data(),T_host.size());
//  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);

//  std::vector<Plato::Scalar> Tprev_host( mesh->nverts() );
//  Tval = 0.0; dval = 0.5000;
//  for( auto& val : Tprev_host ) val = (Tval += dval);
//  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//    Tprev_host_view(Tprev_host.data(),Tprev_host.size());
//  auto Tprev = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), Tprev_host_view);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='First Order'/>         \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                   \n"
    "  <ParameterList name='First Order'>                                           \n"
    "    <ParameterList name='Penalty Function'>                                    \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                   \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                      \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Material Model'>                                        \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='1000.0'/>\n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                    \n"
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
  VectorFunctionInc<::Plato::Thermomechanics<spaceDim>> 
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = params->sublist("Time Integration").get<double>("Time Step");
  auto residual = vectorFunction.value(state, statePrev, z, timeStep);

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


  // compute and test gradient wrt state. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(state, statePrev, z, timeStep);

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


  // compute and test gradient wrt previous state (i.e., jacobian)
  //
  auto jacobian_p = vectorFunction.gradient_p(state, statePrev, z, timeStep);

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
  auto gradient_z = vectorFunction.gradient_z(state, statePrev, z, timeStep);
  
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
  auto gradient_x = vectorFunction.gradient_x(state, statePrev, z, timeStep);
  
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
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( TransientThermomechTests, InternalThermalEnergy3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  // create mesh based temperature from host data
  //
  int tNumSteps = 3;
  int tNumNodes = mesh->nverts();
  Plato::ScalarMultiVector T("temperature history", tNumSteps, tNumNodes);
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     for( int i=0; i<tNumSteps; i++){
       T(i, aNodeOrdinal) = (i+1)*aNodeOrdinal;
     }
  }, "temperature history");


  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Heat Equation'/>      \n"
    "  <Parameter name='Objective' type='string' value='Internal Thermal Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Internal Thermal Energy'>                              \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Model'>                                       \n"
    "    <ParameterList name='Isotropic Linear Thermal'>                           \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>              \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e3'/>           \n"
    "      <Parameter name='Conductivity Coefficient' type='double' value='1.0e6'/>\n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Time Integration'>                                     \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>                \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                   \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create objective
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunctionInc<::Plato::Thermal<spaceDim>> 
    scalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));

  auto timeStep = params->sublist("Time Integration").get<double>("Time Step");
  int timeIncIndex = 1;

  // compute and test objective value
  //
  auto value = scalarFunction.value(T, z, timeStep);

  double value_gold = 7.95166666666666603e9;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(T, z, timeStep, timeIncIndex);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
    -2.266666666666666e7, -2.200000000000000e7, -6.666666666666666e6, 
    -2.700000000000000e7, -8.666666666666666e6, -6.666666666666667e6, 
    -6.999999999999999e6, -2.000000000000001e6,  9.999999999999998e6, 
    -2.699999999999999e7, -1.466666666666667e7, -3.999999999999999e6, 
     1.000000000000002e6, -1.000000000000000e7, -2.000000000000007e6, 
     4.666666666666670e6,  2.000000000000000e7,  1.333333333333331e6, 
     2.000000000000002e6,  2.000000000000002e6,  2.399999999999999e7, 
     2.800000000000001e7,  1.533333333333333e7,  2.666666666666666e6, 
     6.666666666666670e6,  3.600000000000000e7,  6.666666666666666e6
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
  auto grad_z = scalarFunction.gradient_z(T, z, timeStep);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
    6.613750000000000e8, 6.175000000000000e8, 1.543750000000000e8, 
    3.813333333333334e8, 1.121250000000000e8, 5.741666666666666e7, 
    1.235000000000000e8, 3.900000000000000e7, 1.987916666666666e8, 
    5.492500000000000e8, 2.334583333333333e8, 2.275000000000000e7, 
    3.791666666666666e7, 1.168916666666667e9, 5.774166666666667e8, 
    1.803750000000000e8, 2.074583333333333e8, 1.056250000000000e8, 
    4.820833333333334e7, 1.197083333333333e8, 4.495833333333334e8, 
    7.133750000000000e8, 2.811250000000000e8, 1.191666666666667e7, 
    3.087500000000001e7, 7.659166666666666e8, 1.023750000000000e8
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(T, z, timeStep);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
     2.457000000000000e9, -2.656333333333333e9, -1.261000000000000e9, 
     2.437500000000000e9, -2.710500000000000e9,  2.860000000000001e8, 
     6.825000000000000e8, -8.775000000000000e8,  1.321666666666667e9, 
     1.895833333333333e9,  1.124500000000000e9, -4.550000000000001e8, 
     4.506666666666666e8,  1.928333333333333e8,  7.821666666666667e8, 
     4.073333333333333e8,  4.030000000000000e8,  2.296666666666667e8, 
     1.603333333333333e8,  8.493333333333333e8, -2.296666666666667e8, 
     3.900000000000002e7,  2.860000000000000e8, -1.170000000000000e8, 
    -1.371500000000000e9, -5.156666666666666e8, -5.914999999999999e8, 
     2.223000000000000e9, -5.134999999999999e8, -1.180833333333333e9, 
     1.284833333333333e9, -1.852500000000000e9,  1.213333333333331e8, 
     9.100000000000001e7,  1.950000000000001e8,  1.299999999999999e8, 
    -4.333333333333340e6,  1.906666666666667e8, -2.903333333333335e8, 
     1.419166666666667e9,  4.289999999999999e8,  1.219833333333333e9, 
     6.825000000000000e8,  3.965000000000000e8,  1.759333333333333e9, 
    -3.943333333333335e8,  3.705000000000000e8,  4.918333333333333e8, 
    -1.232833333333334e9,  1.690000000000002e8, -1.928333333333334e8, 
     3.531666666666668e8,  2.665000000000002e8, -1.170000000000002e8, 
     4.333333333333328e6,  1.083333333333333e8,  1.430000000000000e8, 
     2.145000000000001e8,  1.083333333333333e8,  3.358333333333334e8, 
     6.933333333333373e7,  2.329166666666666e9, -3.228333333333337e8, 
    -4.454666666666666e9, -1.109333333333333e9,  3.293333333333337e8, 
    -2.433166666666667e9, -6.153333333333334e8,  1.042166666666667e9, 
    -2.166666666666649e7,  4.333333333333302e7,  4.766666666666681e7, 
    -4.116666666666666e7,  1.559999999999996e8,  1.755000000000002e8, 
    -5.650666666666667e9,  2.775500000000000e9, -2.901166666666666e9, 
     7.323333333333335e8,  4.571666666666666e8, -7.561666666666669e8
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}

#endif
