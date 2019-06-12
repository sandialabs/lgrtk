/*!
  These unit tests are for the stabilized two-field formulation
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
#include "plato/PressureDivergence.hpp"
#include "plato/Thermomechanics.hpp"
#include "plato/ComputedField.hpp"

#include <fenv.h>


using namespace lgr;

TEUCHOS_UNIT_TEST( TwoFieldThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int tNumCells = mesh->nelems();
  constexpr int numVoigtTerms = Plato::SimplexTwoFieldThermomechanics<spaceDim>::m_numVoigtTerms;
  constexpr int nodesPerCell  = Plato::SimplexTwoFieldThermomechanics<spaceDim>::m_numNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexTwoFieldThermomechanics<spaceDim>::m_numDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexTwoFieldThermomechanics<spaceDim>::m_numDofsPerNode;

  static constexpr int PDofOffset = spaceDim;
  static constexpr int TDofOffset = spaceDim+1;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+2); // displacements + pressure + temperature
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarMultiVector tPGradWS("Projected pressure gradient workset", tNumDofs, spaceDim*nodesPerCell);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
  {
      for(int iNode=0; iNode<nodesPerCell; iNode++)
      {
          for(int iDim=0; iDim<spaceDim; iDim++)
          {
              tPGradWS(aCellOrdinal, iNode*spaceDim+iDim) = (4e-7)*(iNode+1)*(iDim+1)*(aCellOrdinal+1);
          }
      }
  }, "projected pgrad");

  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (5e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+4) = (4e-7)*aNodeOrdinal;
  }, "state");

  Plato::WorksetBase<Plato::SimplexTwoFieldThermomechanics<spaceDim>> worksetBase(*mesh);

  Plato::ScalarArray3D     gradient           ("gradient",           tNumCells, nodesPerCell, spaceDim);
  Plato::ScalarArray3D     configWS           ("config workset",     tNumCells, nodesPerCell, spaceDim);
  Plato::ScalarMultiVector tDispGrad          ("sym disp grad",      tNumCells, numVoigtTerms);
  Plato::ScalarMultiVector tDevStress         ("deviatoric stress",  tNumCells, numVoigtTerms);
  Plato::ScalarMultiVector tStressDivResult   ("stress div",         tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tPressureDivResult ("pressure div",       tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tStabDivResult     ("stabilization div",  tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tFluxDivResult     ("thermal flux div",   tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tVolResult         ("volume diff proj",   tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tMassResult        ("mass",               tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tStateWS           ("state workset",      tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tPressureGrad      ("pressure grad",      tNumCells, spaceDim);
  Plato::ScalarMultiVector tProjectedPGrad    ("projected p grad",   tNumCells, spaceDim);
  Plato::ScalarMultiVector tTGrad             ("Temperature grad",   tNumCells, spaceDim);
  Plato::ScalarMultiVector tCellStab          ("cell stabilization", tNumCells, spaceDim);
  Plato::ScalarMultiVector tTFlux             ("thermal flux",       tNumCells, spaceDim);
  Plato::ScalarVector      tVolStrain         ("volume strain",      tNumCells);
  Plato::ScalarVector      tTemperature       ("GP temperature",     tNumCells);
  Plato::ScalarVector      tPressure          ("GP pressure",        tNumCells);
  Plato::ScalarVector      tThermalContent    ("GP heat at step k",  tNumCells);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, tStateWS);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='1000.0'/>\n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  Plato::ThermoelasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel    = mmfactory.create();
  auto cellDensity      = materialModel->getMassDensity();
  auto cellSpecificHeat = materialModel->getSpecificHeat();

  Plato::ComputeGradientWorkset<spaceDim>    computeGradient;
  Plato::TwoFieldTMKinematics<spaceDim>      kinematics;
  Plato::TwoFieldTMKinetics<spaceDim>        kinetics(materialModel);

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, TDofOffset>  interpolateTemperatureFromNodal;
  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, PDofOffset>  interpolatePressureFromNodal;
  Plato::InterpolateFromNodal<spaceDim, spaceDim, 0, spaceDim>    interpolatePGradFromNodal;

  Plato::FluxDivergence     <spaceDim, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::FluxDivergence     <spaceDim, dofsPerNode, PDofOffset> stabDivergence;
  Plato::StressDivergence   <spaceDim, dofsPerNode>             stressDivergence;
  Plato::PressureDivergence <spaceDim, dofsPerNode>             pressureDivergence;

  Plato::ThermalContent computeThermalContent(cellDensity, cellSpecificHeat);

  Plato::ProjectToNode<spaceDim, dofsPerNode, PDofOffset> projectVolumeStrain;
  Plato::ProjectToNode<spaceDim, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();
  auto basisFunctions = cubatureRule.getBasisFunctions();

  Plato::Scalar tTimeStep = 2.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",tNumCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    kinematics(cellOrdinal, tDispGrad, tPressureGrad, tTGrad, tStateWS, gradient);

    interpolatePGradFromNodal        ( cellOrdinal, basisFunctions, tPGradWS, tProjectedPGrad );
    interpolatePressureFromNodal     ( cellOrdinal, basisFunctions, tStateWS, tPressure       );
    interpolateTemperatureFromNodal  ( cellOrdinal, basisFunctions, tStateWS, tTemperature    );

    kinetics(cellOrdinal, cellVolume,
             tProjectedPGrad, tPressure,     tTemperature,
             tDispGrad,       tPressureGrad, tTGrad,
             tDevStress,      tVolStrain,    tTFlux,  tCellStab);

    stressDivergence   (cellOrdinal, tStressDivResult,   tDevStress, gradient, cellVolume, tTimeStep/2.0);
    pressureDivergence (cellOrdinal, tPressureDivResult, tPressure,  gradient, cellVolume, tTimeStep/2.0);
    stabDivergence     (cellOrdinal, tStabDivResult,     tCellStab,  gradient, cellVolume, tTimeStep/2.0);
    fluxDivergence     (cellOrdinal, tFluxDivResult,     tTFlux,     gradient, cellVolume, tTimeStep/2.0);

    computeThermalContent(cellOrdinal, tThermalContent, tTemperature);
    projectVolumeStrain  (cellOrdinal, cellVolume, basisFunctions, tVolStrain, tVolResult);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, tMassResult);

  }, "divergence");

  {
    // test deviatoric stress
    //
    auto tDevStress_Host = Kokkos::create_mirror_view( tDevStress );
    Kokkos::deep_copy( tDevStress_Host, tDevStress );

    std::vector<std::vector<double>> gold = {
      {-1.93445965977317989e8,-1.93245832555036485e8,-1.93299201467644870e8,213475.650433622417,46697.7985323549146,66711.1407605069835},
      {-7.32464325550366193e7,-7.31663791861240268e7,-7.35799882588391453e7,93395.5970647098002,166777.851901267481,186791.194129419571},
      {-1.13273817011340767e8,-1.13353870380253404e8,-1.13367212608405486e8,53368.9126084056043,206804.536357571720,146764.509673115390}
    };

    int tNumCells=gold.size(), numVoigt=6;
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iVoigt=0; iVoigt<numVoigt; iVoigt++){
        TEST_FLOATING_EQUALITY(tDevStress_Host(iCell, iVoigt), gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  {
    // test volume strain
    //
    auto tVolStrain_Host = Kokkos::create_mirror_view( tVolStrain );
    Kokkos::deep_copy( tVolStrain_Host, tVolStrain );

    std::vector<double> gold = { 5.79990999999984923e-6,2.19992799999987954e-6,3.39994899999991458e-6 };

    int tNumCells=gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      TEST_FLOATING_EQUALITY(tVolStrain_Host(iCell), gold[iCell], 1e-13);
    }
  }

  {
    // test cell stabilization
    //
    auto tCellStab_Host = Kokkos::create_mirror_view( tCellStab );
    Kokkos::deep_copy( tCellStab_Host, tCellStab );

    std::vector<std::vector<double>> gold = {
      {7.21000736213553247e-34, 6.80965942163844400e-18, 1.13494323693973951e-18},
      {9.07954589551792534e-18, 4.53977294775896267e-18,-1.24843756063371466e-17},
      {7.94460265857818390e-18,-4.53977294775896267e-18,-9.07954589551792534e-18}
    };

    int tNumCells=gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        TEST_FLOATING_EQUALITY(tCellStab_Host(iCell, iDim), gold[iCell][iDim], 1e-13);
      }
    }
  }

  {
    // test thermal flux
    //
    auto tflux_Host = Kokkos::create_mirror_view( tTFlux );
    Kokkos::deep_copy( tflux_Host, tTFlux );

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
  }

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

  {
    // test projected pressure gradient
    //
    auto tProjectedPGrad_Host = Kokkos::create_mirror_view( tProjectedPGrad );
    Kokkos::deep_copy( tProjectedPGrad_Host, tProjectedPGrad );

    std::vector<std::vector<double>> gold = { 
      {1.00000000000000000e-6, 2.00000000000000000e-6, 3.00000000000000000e-6},
      {2.00000000000000000e-6, 4.00000000000000000e-6, 6.00000000000000000e-6},
      {3.00000000000000000e-6, 6.00000000000000000e-6, 9.00000000000000000e-6}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        TEST_FLOATING_EQUALITY(tProjectedPGrad_Host(iCell,iDim), gold[iCell][iDim], 1e-13);
      }
    }
  }

  {
    // test pressure gradient
    //
    auto tPressureGrad_Host = Kokkos::create_mirror_view( tPressureGrad );
    Kokkos::deep_copy( tPressureGrad_Host, tPressureGrad );

    std::vector<std::vector<double>> gold = { 
      {1.00000000000000059e-6, 7.99999999999999964e-6, 3.99999999999999897e-6},
      {9.99999999999999912e-6, 7.99999999999999964e-6,-4.99999999999999956e-6},
      {9.99999999999999912e-6, 2.00000000000000033e-6, 9.99999999999999955e-7}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        TEST_FLOATING_EQUALITY(tPressureGrad_Host(iCell,iDim), gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test temperature gradient
  //
  auto tgrad_Host = Kokkos::create_mirror_view( tTGrad );
  Kokkos::deep_copy( tgrad_Host, tTGrad );

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

  // test stress divergence and local assembly
  //
  {
    auto tStressDivResult_Host = Kokkos::create_mirror_view( tStressDivResult );
    Kokkos::deep_copy( tStressDivResult_Host, tStressDivResult );

    std::vector<std::vector<double>> gold = { 
      {-2779.63086502112401,    8.05190968979318626e6, -8894.81876806760010,    0.0000000000000000, 0.000000000000000,
       -8.06219432399376482e6, -6115.18790304647700,    8.05607913609071821e6,  0.0000000000000000, 0.000000000000000,
        8.06302821325327083e6, -8.05468932065820787e6,  6949.07716255281230,    0.0000000000000000, 0.000000000000000,
        1945.74160551478803,    8894.81876806760010,   -8.05413339448520262e6,  0.0000000000000000, 0.000000000000000},
      {-7782.96642205914850,    3.04859913275516778e6, -3891.48321102957470,    0.0000000000000000, 0.000000000000000,
        833.889259506337112,   -3.05249061596619710e6,  3.06972432732932689e6,  0.0000000000000000, 0.000000000000000,
        3.05888376695574494e6, -3891.48321102957379,   -3.07278192128085019e6,  0.0000000000000000, 0.000000000000000,
       -3.05193468979319232e6,  7782.96642205914850,    6949.07716255281139,    0.0000000000000000, 0.000000000000000},
      {-8616.85568156548834,   -2223.70469201690003,    4.72363385868356191e6,  0.0000000000000000, 0.000000000000000,
        4.72585756337557826e6, -4.72919312041360512e6, -6393.15098954858786,    0.0000000000000000, 0.000000000000000,
        2501.66777851901361,    4.72530163720257487e6, -4.72585756337557826e6,  0.0000000000000000, 0.000000000000000,
       -4.71974237547253165e6,  6115.18790304647428,    8616.85568156548834,    0.0000000000000000, 0.000000000000000}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tStressDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tStressDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  // test pressure divergence and local assembly
  //
  {
    auto tPressureDivResult_Host = Kokkos::create_mirror_view( tPressureDivResult );
    Kokkos::deep_copy( tPressureDivResult_Host, tPressureDivResult );

    std::vector<std::vector<double>> gold = { 
      { 0.0000000000000000,    -1.56249999999999986e-7, 0.0000000000000000,     0.0000000000000000, 0.0000000000000000,
        1.56249999999999986e-7, 0.0000000000000000,    -1.56249999999999986e-7, 0.0000000000000000, 0.0000000000000000,
       -1.56249999999999986e-7, 1.56249999999999986e-7, 0.0000000000000000,     0.0000000000000000, 0.0000000000000000,
        0.0000000000000000,     0.0000000000000000,     1.56249999999999986e-7, 0.0000000000000000, 0.0000000000000000},
      { 0.0000000000000000,    -1.24999999999999994e-7, 0.0000000000000000,     0.0000000000000000, 0.0000000000000000,
        0.0000000000000000,     1.24999999999999994e-7,-1.24999999999999994e-7, 0.0000000000000000, 0.0000000000000000,
       -1.24999999999999994e-7, 0.0000000000000000,     1.24999999999999994e-7, 0.0000000000000000, 0.0000000000000000,
        1.24999999999999994e-7, 0.0000000000000000,     0.0000000000000000,     0.0000000000000000, 0.0000000000000000},
      { 0.0000000000000000,     0.0000000000000000,    -8.85416666666666571e-8, 0.0000000000000000, 0.0000000000000000, 
       -8.85416666666666571e-8, 8.85416666666666571e-8, 0.0000000000000000,     0.0000000000000000, 0.0000000000000000, 
        0.0000000000000000,    -8.85416666666666571e-8, 8.85416666666666571e-8, 0.0000000000000000, 0.0000000000000000, 
        8.85416666666666571e-8, 0.0000000000000000,     0.0000000000000000,     0.0000000000000000, 0.0000000000000000}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tPressureDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tPressureDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  // test stabilization divergence and local assembly
  //
  {
    auto tStabDivResult_Host = Kokkos::create_mirror_view( tStabDivResult );
    Kokkos::deep_copy( tStabDivResult_Host, tStabDivResult );

    std::vector<std::vector<double>> gold = { 
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -2.83735809234935167e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -4.72893015391557809e-20, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  2.83735809234935119e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  4.72893015391558110e-20, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1.89157206156623436e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  7.09339523087337821e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -8.98496729243961329e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  3.78314412313246873e-19, 0.0000000000000000},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  3.78314412313246873e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -5.20182316930714408e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1.89157206156623436e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  3.31025110774090996e-19, 0.0000000000000000}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tStabDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tStabDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  // test thermal flux divergence and local assembly
  //
  {
    auto tFluxDivResult_Host = Kokkos::create_mirror_view( tFluxDivResult );
    Kokkos::deep_copy( tFluxDivResult_Host, tFluxDivResult );

    std::vector<std::vector<double>> gold = { 
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.000266666666666666625,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.0000999999999999999506,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.000233333333333333277,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.000133333333333333286},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.000266666666666666625,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.000433333333333333313,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.000499999999999999902,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.000333333333333333268},
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.0000333333333333333282,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.000266666666666666571,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.0000333333333333333282,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.000333333333333333268}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tFluxDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tFluxDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test projected volume eqn
    //
    auto tVolResult_Host = Kokkos::create_mirror_view( tVolResult );
    Kokkos::deep_copy( tVolResult_Host, tVolResult );

    std::vector<std::vector<double>> gold = { 
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 3.02078645833325437e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 3.02078645833325437e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 3.02078645833325437e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 3.02078645833325437e-8, 0.000000000000000000},
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.14579583333327059e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.14579583333327059e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.14579583333327059e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.14579583333327059e-8, 0.000000000000000000},
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.77080677083328873e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.77080677083328873e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.77080677083328873e-8, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.77080677083328873e-8, 0.000000000000000000}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tVolResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tVolResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test projected mass
    //
    auto tMassResult_Host = Kokkos::create_mirror_view( tMassResult );
    Kokkos::deep_copy( tMassResult_Host, tMassResult );

    std::vector<std::vector<double>> gold = { 
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896},
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tMassResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tMassResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }
}

//
//
#ifdef NOPE
//
//


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         TransientTwoFieldThermomechResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( TransientTwoFieldThermomechTests, TransientTwoFieldThermomechResidual3D )
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
  Plato::VectorFunctionInc<::Plato::TwoFieldThermomechanics<spaceDim>>
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
#endif
