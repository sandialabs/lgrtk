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
#include <plato/VectorFunctionVMS.hpp>
#include <plato/StateValues.hpp>
#include <plato/Plato_Solve.hpp>
#include "plato/ApplyConstraints.hpp"
#include "plato/PressureDivergence.hpp"
#include "plato/StabilizedThermomechanics.hpp"
#include "plato/ThermalContent.hpp"
#include "plato/ComputedField.hpp"
#include "plato/PlatoMathHelpers.hpp"

#include <fenv.h>


using namespace lgr;

TEUCHOS_UNIT_TEST( StabilizedThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int tNumCells = mesh->nelems();
  constexpr int numVoigtTerms = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumVoigtTerms;
  constexpr int nodesPerCell  = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumDofsPerNode;

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

  Plato::WorksetBase<Plato::SimplexStabilizedThermomechanics<spaceDim>> worksetBase(*mesh);

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
  Plato::StabilizedTMKinematics<spaceDim>      kinematics;
  Plato::StabilizedTMKinetics<spaceDim>        kinetics(materialModel);

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

    std::vector<std::vector<Plato::Scalar>> gold = {
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

    std::vector<Plato::Scalar> gold = { 5.79990999999984923e-6,2.19992799999987954e-6,3.39994899999991458e-6 };

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

    std::vector<std::vector<Plato::Scalar>> gold = {
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

    std::vector<std::vector<Plato::Scalar>> tflux_gold = { 
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

  std::vector<Plato::Scalar> cellVolume_gold = { 
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

  std::vector<Plato::Scalar> tTemperature_gold = { 
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

  std::vector<Plato::Scalar> tThermalContent_gold = { 
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

  std::vector<std::vector<std::vector<Plato::Scalar>>> gradient_gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

  std::vector<std::vector<Plato::Scalar>> tgrad_gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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

    std::vector<std::vector<Plato::Scalar>> gold = { 
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


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         TransientStabilizedThermomechResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( TransientStabilizedThermomechTests, TransientStabilizedThermomechResidual3D )
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
  int tNumDofsPerNode = (spaceDim+2);
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarVector tState        ("state",         tNumDofs);
  Plato::ScalarVector tControl      ("control",       tNumNodes);
  Plato::ScalarVector tProjPGrad    ("ProjPGrad",     tNumNodes*spaceDim);
  Plato::ScalarVector tProjectState ("Project state", tNumNodes);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     tControl(aNodeOrdinal) = 1.0;

     tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+4) = (4e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+3) =    0.0*aNodeOrdinal;
  }, "state");


  // create input for stabilized thermomechanics
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                            \n"
    "  <ParameterList name='Stabilized Elliptic'>                                            \n"
    "    <ParameterList name='Penalty Function'>                                             \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                               \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
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
    "  <ParameterList name='Time Stepping'>                                                  \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>                          \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                             \n"
    "  </ParameterList>                                                                      \n"
    "  <ParameterList name='Newton Iteration'>                                               \n"
    "    <Parameter name='Number Iterations' type='int' value='2'/>                          \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::VectorFunctionVMS<::Plato::StabilizedThermomechanics<spaceDim>>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // create input pressure gradient projector
  //
  Teuchos::RCP<Teuchos::ParameterList> paramsProjector =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "</ParameterList>                                                                        \n"
  );

  // copy projection state
  Plato::extract<Plato::StabilizedThermomechanics<spaceDim>::mNumDofsPerNode,
                 Plato::StabilizedThermomechanics<spaceDim>::ProjectorT::SimplexT::mProjectionDof>(tState, tProjectState);


  // create constraint evaluator
  //
  Plato::VectorFunctionVMS<::Plato::Projection<spaceDim>>
    tProjectorVectorFunction(*mesh, tMeshSets, tDataMap, *paramsProjector, "State Gradient Projection");

  auto tProjResidual = tProjectorVectorFunction.value      (tProjPGrad, tProjectState, tControl);
  auto tProjJacobian = tProjectorVectorFunction.gradient_u (tProjPGrad, tProjectState, tControl);


  Plato::Solve::RowSummed<spaceDim>(tProjJacobian, tProjPGrad, tProjResidual);


  // compute and test value
  //
  auto timeStep = params->sublist("Time Stepping").get<Plato::Scalar>("Time Step");
  auto tResidual = vectorFunction.value(tState, tProjPGrad, tControl, timeStep);

  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<Plato::Scalar> tGold = { 
    491552.996442073258,    1.10832751111852229e7,  6.61308108739158791e6, 6.87466874999999915e-8, -0.00226666666666666640,
    5.78493201578829531e6,  1.41424701745608039e7, -57571.6922392702400,   9.99957499999999840e-8, -0.00220000000000000013,
    1.95391799810984917e6,  4.71027496386479307e6, -6.68670797198131401e6, 2.49990624999999995e-8, -0.000666666666666666753,
    2.05292256059595048e7, -6.71488761396485474e6,  9.37602287080274895e6, 1.54160557291666644e-7, -0.00269999999999999928,
    1.36293128168779034e7, -567903.585723814205,   -1.30839242411607616e7, 6.66650104166666554e-8, -0.000866666666666666627,
    7.78551271959083341e6, -4.17850657938625384e6, -3.63035503947075596e6, 2.91657916666666688e-8, -0.000666666666666666753,
    5.28334197798532061e6, -7.51597482210361958e6,  2.20428060929508461e6, 3.33311614583333369e-8, -0.000700000000000000101,
    4.16620953691349411e6, -5.56226395374693722e6,  1.38604774571936741e6, 2.08320833333333332e-8, -0.000200000000000000010
  };

  for(int iNode=0; iNode<int(tGold.size()); iNode++){
    if(tGold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tGold[iNode], 1e-13);
    }
  }


  // compute and test gradient wrt state. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(tState, tProjPGrad, tControl, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
-5.53702468312207617e12, 0, 0, -0.0208333333333333322,
-2.08333333333333172e7, 0, -5.53702468312207422e12, 0,
-0.0208333333333333322, -2.08333333333333172e7, 0, 0,
-5.53702468312207422e12, -0.0208333333333333322,
-2.08333333333333172e7, -0.0208333333333333322,
-0.0208333333333333322, -0.0208333333333333322,
-3.12500000000000246e-16, -2.34375000000000006e-7, 0, 0, 0, 0,
499.999999999999943, -5.55926173004224968e9,
2.77963086502112484e9, 0, 0, 0, -2.77963086502112207e12,
5.54814320658216016e12, -2.77963086502112207e12,
-0.0208333333333333322, -2.08333333333333172e7, 0,
2.77963086502112484e9, -5.55926173004224968e9, 0, 0,
-0.0104166666666666661, 0.0208333333333333322,
-0.0104166666666666661, -1.04166666666666749e-16,
-7.81250000000000064e-8, 0, 0, 0, 0, -166.666666666666657,
-5.55926173004224968e9, 0, 2.77963086502112484e9, 0, 0, 0,
-5.55926173004224968e9, 2.77963086502112484e9, 0, 0,
-2.77963086502112207e12, -2.77963086502112207e12,
5.54814320658216016e12, -0.0208333333333333322,
-2.08333333333333172e7, -0.0104166666666666661,
-0.0104166666666666661, 0.0208333333333333322,
-1.04166666666666749e-16, -7.81250000000000064e-8, 0, 0, 0, 0,
-166.666666666666657, 0, 2.77963086502112484e9,
2.77963086502112484e9, 0, 0, -2.77963086502112207e12, 0,
2.77685123415610107e12, -0.0104166666666666661,
-1.04166666666666586e7, -2.77963086502112207e12,
2.77685123415610107e12, 0, -0.0104166666666666661,
-1.04166666666666586e7, -0.0208333333333333322,
0.0104166666666666661, 0.0104166666666666661,
-1.04166666666666749e-16, -7.81250000000000064e-8, 0, 0, 0, 0, 0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    if(gold_jac_entries[i] == 0.0){
      TEST_ASSERT(fabs(jac_entriesHost(i)) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
    }
  }

//
//
#ifdef NOPE
//
//


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
#endif

}

TEUCHOS_UNIT_TEST( PlatoMathFunctors, RowSumSolve )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based solution from host data
  //
  int tNumNodes = mesh->nverts();
  Plato::ScalarVector tProjectState ("state",     tNumNodes);
  Plato::ScalarVector tProjPGrad    ("ProjPGrad", tNumNodes*spaceDim);
  Plato::ScalarVector tControl      ("Control",   tNumNodes);
  Plato::fill( 1.0, tControl );
  Plato::fill( 0.0, tProjPGrad );
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     tProjectState(aNodeOrdinal) = 1.0*aNodeOrdinal;
  }, "state");

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "</ParameterList>                                                                        \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::VectorFunctionVMS<::Plato::Projection<spaceDim>>
    tVectorFunction(*mesh, tMeshSets, tDataMap, *params, "State Gradient Projection");

  auto tResidual = tVectorFunction.value      (tProjPGrad, tProjectState, tControl);
  auto tJacobian = tVectorFunction.gradient_u (tProjPGrad, tProjectState, tControl);


  { // test residual
    //

    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar> tGold = {
      -0.9479166666666665,   0.1458333333333333,  -0.01041666666666666,
      -1.250000000000000,    0.2500000000000000,  -0.08333333333333336,
      -0.3125000000000000,   0.06249999999999999, -0.02083333333333334,
      -1.250000000000000,   -0.3333333333333333,   0.1249999999999999,
      -0.4166666666666667,  -0.06249999999999999, -0.04166666666666666,
      -0.2083333333333333,  -0.04166666666666666,  0.00000000000000000,
      -0.4166666666666666,  -0.08333333333333331,  0.08333333333333331,
      -0.1562500000000000,   0.02083333333333334, -0.03124999999999999,
      -0.4999999999999998,  -0.2708333333333333,   0.0000000000000000,
      -0.4791666666666667,   0.3125000000000000,  -0.6666666666666665,
      -0.04166666666666670,  0.3541666666666667,  -0.3750000000000001
    };
    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(i), tGold[i], 2.0e-14);
      }
    }
  }

  { // test jacobian
    //

    auto tJacobian_Host = Kokkos::create_mirror_view( tJacobian->entries() );
    Kokkos::deep_copy( tJacobian_Host, tJacobian->entries() );

    std::vector<Plato::Scalar> tGold = {
      0.00781249999999999913, 0, 0, 0, 0.00781249999999999913, 0, 0, 0,
      0.00781249999999999913, 0.00260416666666666652, 0, 0, 0,
      0.00260416666666666652, 0, 0, 0, 0.00260416666666666652,
      0.00260416666666666652, 0, 0, 0, 0.00260416666666666652, 0, 0, 0,
      0.00260416666666666652, 0.00260416666666666652, 0, 0, 0,
      0.00260416666666666652, 0, 0, 0, 0.00260416666666666652
    };

    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tJacobian_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tJacobian_Host(i), tGold[i], 2.0e-14);
      }
    }
  }


  { // test row sum functor
    //
    Plato::RowSum rowSum(tJacobian);

    Plato::ScalarVector tRowSum("row sum", tResidual.extent(0));

    auto tNumBlockRows = tJacobian->rowMap().size() - 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumBlockRows), LAMBDA_EXPRESSION(int blockRowOrdinal)
    {
      // compute row sum
      rowSum(blockRowOrdinal, tRowSum);

    }, "row sum inverse");

    auto tRowSum_Host = Kokkos::create_mirror_view( tRowSum );
    Kokkos::deep_copy( tRowSum_Host, tRowSum );

    std::vector<Plato::Scalar> tRowSum_gold = {
      0.0312500000000000000, 0.0312500000000000000, 0.0312500000000000000,
      0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
      0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
      0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
      0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
      0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
      0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
      0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
      0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
      0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
      0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
      0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
      0.0208333333333333356, 0.0208333333333333356, 0.0208333333333333356,
      0.124999999999999972,  0.124999999999999972,  0.124999999999999972,
      0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
      0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
      0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
      0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
      0.0312500000000000000, 0.0312500000000000000, 0.0312500000000000000,
      0.0416666666666666574, 0.0416666666666666574, 0.0416666666666666574,
      0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
      0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
      0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
      0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
      0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
      0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
      0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661
    };

    int tNumGold = tRowSum_gold.size();
    for(int i=0; i<tNumGold; i++){
      TEST_FLOATING_EQUALITY(tRowSum_Host(i), tRowSum_gold[i], 2.0e-14);
    }
  }


  { // test row summed solve
    //
    Plato::Solve::RowSummed<spaceDim>(tJacobian, tProjPGrad, tResidual);

    auto tProjPGrad_Host = Kokkos::create_mirror_view( tProjPGrad );
    Kokkos::deep_copy( tProjPGrad_Host, tProjPGrad );

    std::vector<Plato::Scalar> tGold = {
      30.3333333333333286, -4.66666666666666696,  0.333333333333333037,
      29.9999999999999929, -6.00000000000000000,  2.00000000000000044,
      30.0000000000000000, -5.99999999999999822,  2.00000000000000000,
      20.0000000000000000,  5.33333333333333215, -1.99999999999999911,
      20.0000000000000000,  2.99999999999999911,  1.99999999999999956,
      20.0000000000000000,  4.00000000000000000,  0.00000000000000000,
      20.0000000000000000,  3.99999999999999911, -3.99999999999999911,
      15.0000000000000000, -2.00000000000000089,  2.99999999999999867
    };

    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tProjPGrad_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tProjPGrad_Host(i), tGold[i], 2.0e-14);
      }
    }
  }
}
