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

  Plato::WorksetBase<Plato::SimplexThermomechanics<spaceDim>> worksetBase(*mesh);

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
  Plato::TMKinematics<spaceDim>                   kinematics;
  Plato::TMKinetics<spaceDim>                     kinetics(materialModel);

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, TDofOffset> interpolateFromNodal;

  Plato::FluxDivergence  <spaceDim, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::StressDivergence<spaceDim, dofsPerNode> stressDivergence;

  Plato::ThermalContent computeThermalContent(cellDensity, cellSpecificHeat);
  Plato::ProjectToNode<spaceDim, dofsPerNode, TDofOffset> projectThermalContent;

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


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='First Order'/>                  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                            \n"
    "  <ParameterList name='First Order'>                                                    \n"
    "    <ParameterList name='Penalty Function'>                                             \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                               \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
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
    "  <ParameterList name='Time Integration'>                                               \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>                          \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                             \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
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
   -18669.59575320513,   -14903.70552884615,   -19551.14663461538,    0.02413541666666667, 
   -17427.51482371794,   -15745.01001602564,   -16586.51762820513,    0.03118750000000000,
    1562.601562500000,   -6370.137620192306,   -1682.848557692307,    0.006822916666666667,
   -20191.77644230769,   -9695.429487179485,   -25641.05689102564,    0.04497656250000000,
   -3164.858373397435,   -3205.112580128204,   -120.4136618589738,    0.01215104166666667,
   -2563.956730769230,   -841.4164663461538,   -3325.396033653845,    0.006354166666666668,
   -1562.213541666663,   -40.32972756410084,   -6570.533653846154,    0.01607031250000000
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
    8.81410256410256195e9,  0.00000000000000000,    0.00000000000000000,   13020.8333333333321, 
    0.00000000000000000,    8.81410256410256195e9,  0.00000000000000000,   13020.8333333333321,
    0.00000000000000000,    0.00000000000000000,    8.81410256410256195e9, 13020.8333333333321,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   2468.750000000000000,
   -1.60256410256410241e9,  8.01282051282051206e8,  0.00000000000000000,   0.00000000000000000,
    1.20192307692307663e9, -5.60897435897435760e9,  1.20192307692307663e9, 13020.8333333333321, 
    0.00000000000000000,    8.01282051282051206e8, -1.60256410256410241e9, 0.00000000000000000,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   739.583333333333371,
   -1.60256410256410241e9,  0.00000000000000000,    8.01282051282051206e8, 0.00000000000000000,
    0.00000000000000000,   -1.60256410256410241e9,  8.01282051282051206e8, 0.00000000000000000,
    1.20192307692307663e9,  1.20192307692307663e9, -5.60897435897435760e9, 13020.8333333333321,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   739.583333333333371,
    0.00000000000000000,    8.01282051282051206e8,  8.01282051282051206e8, 0.00000000000000000,
    1.20192307692307663e9,  0.00000000000000000,   -2.00320512820512772e9, 6510.41666666666606,
    1.20192307692307663e9, -2.00320512820512772e9,  0.00000000000000000,   6510.41666666666606,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   781.250000000000000
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
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2218.75000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -822.916666666666629,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -822.916666666666629,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -781.250000000000000
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
   -4667.39893830128131, -3725.92638221153720, -4887.78665865384573, 0.00603385416666666653,
   -651.022135416666629,  2003.16997195512795,  1051.69831730769238, 0.00163411458333333312,
    390.647786458333599, -1592.53700921474319, -420.706931089743875, 0.00147135416666666660,
   -701.095102163461092, -50.0911959134616325, -1231.98677884615336, 0.00114127604166666652
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
   -34615.3846153846025, -37980.7692307692196, -79807.6923076922831, -0.0267395833333333305,
    11858.9743589743539, -8333.33333333333394,  13942.3076923076842, -0.0123645833333333316,
   -160.256410256407435,  480.769230769233445, -2884.61538461538294, -0.0122083333333333315,
   -320.512820512819644, -4727.40785256410163, -4647.43589743589473,  0.0000624999999999999742,
    10015.8693910256370,  16666.6666666666642,  2804.36217948717967, -0.0124062499999999989,
   -6570.51282051281851, -7131.28525641025590, -320.512820512821236, -0.000187499999999999950, 
   -19391.0256410256334,  5929.48717948717967,  41426.4643429487041,  0.000604166666666666480,
   -160.256410256411073, -4647.43589743589564, -14503.1165865384537, -0.000145833333333333318,
    11217.7664262820472,  10416.5781249999964,  13942.3076923076915, -0.0116875000000000000,
   -21474.3589743589691,  1362.30448717948912, -881.321714743589837, -0.000187499999999999950,
    1522.31089743589655,  7532.05128205127949,  15384.6518429487151, -0.00529166666666666674,
    9935.80889423076587,  5048.04046474358620,  3685.89743589743739, -0.00365104166666666658
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}

