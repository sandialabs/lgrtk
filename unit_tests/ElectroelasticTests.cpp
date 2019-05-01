/*!
  These unit tests are for the Electroelastic functionality.
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
#include "plato/Simp.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/ScalarFunction.hpp"
#include "plato/StateValues.hpp"
#include "plato/ApplyConstraints.hpp"
#include "plato/SimplexElectromechanics.hpp"
#include "plato/Electromechanics.hpp"
#include "plato/ComputedField.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/LinearElectroelasticMaterial.hpp"

#include <fenv.h>


using namespace lgr;

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElectroelasticTests, InternalElectroelasticEnergy3D )
{ 
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
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");


  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                \n"
    "  <Parameter name='PDE Constraint' type='string' value='Electroelastostatics'/>     \n"
    "  <Parameter name='Objective' type='string' value='Internal Electroelastic Energy'/>\n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                         \n"
    "  <ParameterList name='Internal Electroelastic Energy'>                             \n"
    "    <ParameterList name='Penalty Function'>                                         \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                        \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                           \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "  <ParameterList name='Electroelastostatics'>                                       \n"
    "    <ParameterList name='Penalty Function'>                                         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                           \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                        \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "  <ParameterList name='Material Model'>                                             \n"
    "    <ParameterList name='Isotropic Linear Electroelastic'>                          \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                 \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>              \n"
    "      <Parameter  name='p11' type='double' value='1.0e-10'/>                        \n"
    "      <Parameter  name='p33' type='double' value='1.4e-10'/>                        \n"
    "      <Parameter  name='e33' type='double' value='15.8'/>                           \n"
    "      <Parameter  name='e31' type='double' value='-5.4'/>                           \n"
    "      <Parameter  name='e15' type='double' value='12.3'/>                           \n"
    "      <Parameter  name='Alpha' type='double' value='1e10'/>                         \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "</ParameterList>                                                                    \n"
  );

  // create constraint
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  VectorFunction<::Plato::Electromechanics<spaceDim>>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));
  // compute and test constraint value
  //

  auto residual = vectorFunction.value(state, z);

  auto residualHost = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy(residualHost, residual);

  std::vector<double> residual_gold = {
   -197679.4871794871, -36815.38461538461, -359338.4615384615,
   -177766.6666666667, -64311.53846153846, -57580.76923076921,
   -336946.1538461537, -190175.0000000000,  131050.0000000000,
   -48280.76923076923, -86397.43589743591, -64525.00000000000,
   -95169.23076923077, -15782.05128205127, -455664.1025641026,
   -215725.0000000000,  115739.7435897436,  3579.487179487175,
   -103580.7692307692, -89233.33333333333,  30743.58974358975,
    7134.615384615390, -96467.94871794872, -60725.00000000000,
   -18849.99999999999, -3960.256410256401, -112382.0512820513
  };

  for(int iVal=0; iVal<int(residual_gold.size()); iVal++){
    if(residual_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(residualHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residualHost[iVal], residual_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<double> jacobian_gold = {
   3.52564102564102478e10, 0.00000000000000000,     0.00000000000000000,     0.00000000000000000,
   0.00000000000000000,    3.52564102564102478e10,  0.00000000000000000,     0.00000000000000000,
   0.00000000000000000,    0.00000000000000000,     3.52564102564102478e10,  6.73333333333333282e10,
   0.00000000000000000,    0.00000000000000000,     6.73333333333333282e10, -5.66666666666666603e9,
  -6.41025641025640965e9,  3.20512820512820482e9,   0.00000000000000000,     0.00000000000000000,
   4.80769230769230652e9, -2.24358974358974304e10,  4.80769230769230652e9,  -4.50000000000000000e9,
   0.00000000000000000,    3.20512820512820482e9,  -6.41025641025640965e9,  -2.05000000000000000e10,
   0.00000000000000000,    1.02500000000000000e10, -2.05000000000000000e10,  1.66666666666666651e9,
  -6.41025641025640965e9,  0.00000000000000000,     3.20512820512820482e9,   1.02500000000000000e10,
   0.00000000000000000,   -6.41025641025640965e9,   3.20512820512820482e9,   1.02500000000000000e10,
   4.80769230769230652e9,  4.80769230769230652e9,  -2.24358974358974304e10, -2.63333333333333321e10,
  -4.50000000000000000e9, -4.50000000000000000e9,  -2.63333333333333321e10,  2.33333333333333302e9,
   0.00000000000000000,    3.20512820512820482e9,   3.20512820512820482e9,   1.02500000000000000e10,
   4.80769230769230652e9,  0.00000000000000000,    -8.01282051282051086e9,  -5.75000000000000000e9,
   4.80769230769230652e9, -8.01282051282051086e9,   0.00000000000000000,     0.00000000000000000,
  -4.50000000000000000e9, -5.75000000000000000e9,   0.00000000000000000,     0.00000000000000000
  };

  for(int iVal=0; iVal<int(jacobian_gold.size()); iVal++){
    if(jacobian_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(jac_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost[iVal], jacobian_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint gradient_z
  //
  auto gradient_z = vectorFunction.gradient_z(state, z);

  auto gradz_entries = gradient_z->entries();
  auto gradz_entriesHost = Kokkos::create_mirror_view( gradz_entries );
  Kokkos::deep_copy(gradz_entriesHost, gradz_entries);

  std::vector<double> gradient_z_gold = { 
    -49419.8717948717822, -9203.84615384615245, -89834.6153846153757, -44441.6666666666715,
    -11054.1666666666642,  262.820512820512704,  26165.0641025641016,  16022.9166666666642,
     32762.4999999999927, -12070.1923076923067, -21599.3589743589655, -16131.2500000000000, 
     5645.51282051282124,  7549.67948717948639, -29961.2179487179419, -18079.1666666666679
  };

  for(int iVal=0; iVal<int(gradient_z_gold.size()); iVal++){
    if(gradient_z_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradz_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradz_entriesHost[iVal], gradient_z_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint gradient_x
  //
  auto gradient_x = vectorFunction.gradient_x(state, z);

  auto gradx_entries = gradient_x->entries();
  auto gradx_entriesHost = Kokkos::create_mirror_view( gradx_entries );
  Kokkos::deep_copy(gradx_entriesHost, gradx_entries);

  std::vector<double> gradient_x_gold = { 
 -138461.538461538439, -151923.076923076878, -1.13543076923076902e6,
 -543483.333333333372,  47435.8974358974156, -33333.3333333333067,
  184569.230769230722,  85666.6666666666570, -641.025641025600635,
  1923.07692307693469, -24405.1282051281742, -8449.99999999999636,
 -1282.05128205127858, -86510.2564102563774,  6010.25641025640653,
  18050.0000000000073,  107664.102564102534,  66666.6666666666570, 
  97951.2820512820472,  57583.3333333333067, -26282.0512820512740,
 -5525.64102564103086, -75082.0512820513104, -37849.9999999999927,
 -169564.102564102534,  125317.948717948748,  471171.794871794758,
  123633.333333333299, -111441.025641025626, -27789.7435897435826,
 -131746.153846153757, -27966.6666666666388,  44871.7948717948748,
  41666.6666666666788,  257235.897435897437,  134566.666666666628,
 -131897.435897435847,  17048.7179487179419, -79658.9743589743593,
 -15233.3333333333430,  31289.7435897435935,  39328.2051282051252,
  206738.461538461503,  86883.3333333333285,  39743.5897435897350,
 -2807.69230769232308,  145943.589743589720,  76233.3333333333285
  };

  for(int iVal=0; iVal<int(gradient_x_gold.size()); iVal++){
    if(gradient_x_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradx_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradx_entriesHost[iVal], gradient_x_gold[iVal], 1e-13);
    }
  }


  // create objective
  //
  Plato::ScalarFunction<::Plato::Electromechanics<spaceDim>>
    scalarFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("Objective"));

  // compute and test objective value
  //
  auto value = scalarFunction.value(state, z);

  double value_gold = 21.1519092307692240;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(state, z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
   -395358.9743589743, -73630.76923076922, -718676.9230769230,
   -355533.3333333334, -128623.0769230768, -115161.5384615384,
   -673892.3076923075, -380350.0000000000,  262100.0000000000,
   -96561.53846153847, -172794.8717948718, -129050.0000000000,
   -190338.4615384615, -31564.10256410257, -911328.2051282051,
   -431450.0000000001,  231479.4871794872,  7158.974358974370,
   -207161.5384615385, -178466.6666666666,  61487.17948717948,
    14269.23076923077, -192935.8974358974, -121450.0000000000,
   -37700.00000000001, -7920.512820512809, -224764.1025641025
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
  auto grad_z = scalarFunction.gradient_z(state, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
   1.720800064102564,   1.618625897435897,   0.4046564743589744,
   1.004125000000000,   0.3144719230769231,  0.1548513461538462,
   0.3144121794871795,  0.1081301923076923,  0.5520684615384616,
   1.527362692307692,   0.5988296153846151,  0.09487416666666657,
   0.1515557692307692,  3.085979230769230,   1.522474230769231,
   0.4872359615384614,  0.5878107051282053,  0.3662033974358975,
   0.1298328205128205,  0.3157279487179486,  1.174941666666666,
   1.865417500000000,   0.7232746794871795,  0.03222512820512814,
   0.09945391025641004, 1.948115641025640,   0.2484526282051280
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(state, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   6.90518205128204965,  -6.85273076923076729,  -3.10481179487179482,
   6.77807615384615225,  -7.12497615384615379,   0.780404615384615474,
   1.34823564102564086,  -2.26802897435897410,   3.43037333333333283,
   5.21713897435897422,   2.96319948717948689,  -1.21622358974358913,
   0.277901794871794561,  0.403572820512820951,  2.09036794871794829,
   0.710353333333332726,  1.01094641025640986,   0.617697179487179504,
   0.543010512820512159,  2.22852538461538430,  -0.631490769230768856,
   0.0718433333333334256, 0.798704102564102736, -0.339847948717949100,
  -3.54309769230769200,  -1.09399384615384609,  -1.41852025641025614,
   6.03633538461538421,  -0.762717435897436014, -2.63330923076923051,
   3.45966128205128109,  -4.56554205128204948,   0.454507948717948529,
   0.376695128205128127,  0.735363846153846423,  0.300203589743589516,
   0.0368887179487180916, 0.891396153846154560, -0.897257179487180201,
   4.12960230769230563,   0.328286153846154338,  3.06306384615384619,
   0.849002820512820833,  0.913744615384613490,  4.57430435897435839,
  -1.81647333333333494,   0.841072307692308008,  1.31697076923076972,
  -3.10649871794872112,   0.569011282051284262, -0.829053589743591668,
   1.01224871794871940,   0.883736923076923753, -0.987517435897435125,
  -0.0820646153846170956, 0.249462564102562179,  0.360167179487179911,
   0.784505641025641465, -0.0102317948717979834, 0.873601025641025375,
   0.436167948717949727,  5.71724307692307665,  -1.11944538461539045,
  -11.2721102564102544,  -3.05945743589743513,   0.894082564102564481,
  -6.22121410256410279,  -1.70029589743590304,   2.69736128205128445,
  -0.0347199999999993625, 0.0672533333333309447, 0.129993846153851067,
  -0.0476807692307683884, 0.300592307692305938,  0.505793076923074625,
  -14.6053533333333316,   7.27858999999999590,  -7.05406846153846345,
   1.75636307692307758,   1.25727358974358783,  -1.85734692307692240
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-12);
  }
}

