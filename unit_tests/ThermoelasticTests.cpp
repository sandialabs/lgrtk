/*!
  These unit tests are for the Thermoelastic functionality.
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
#include "plato/ApplyWeighting.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/ScalarFunction.hpp"
#include "plato/StateValues.hpp"
#include "plato/ApplyConstraints.hpp"
#include "plato/SimplexThermomechanics.hpp"
#include "plato/Thermomechanics.hpp"
#include "plato/ComputedField.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/LinearThermoelasticMaterial.hpp"

#include <fenv.h>


using namespace lgr;

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ThermoelasticTests, InternalThermoelasticEnergy3D )
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
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Thermoelastostatics'/>          \n"
    "  <Parameter name='Objective' type='string' value='Internal Thermoelastic Energy'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                             \n"
    "  <ParameterList name='Thermoelastostatics'>                                            \n"
    "    <ParameterList name='Penalty Function'>                                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                            \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "  <ParameterList name='Internal Thermoelastic Energy'>                                  \n"
    "    <ParameterList name='Penalty Function'>                                             \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                               \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  VectorFunction<::Plato::Thermomechanics<spaceDim>> 
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = vectorFunction.value(state, z);

  auto residualHost = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy(residualHost, residual);

  std::vector<double> residual_gold = { 
    -74678.38301282050,    -59614.82211538460,     -78204.58653846153,
    -0.002062666666666666, -69710.05929487177,     -62980.04006410255,
    -66346.07051282052,    -0.002002000000000000,   6250.406250000000,
    -25480.55048076922,    -6731.394230769230,     -0.0006066666666666668,
    -80767.10576923075,    -38781.71794871794,     -102564.2275641025,
    -0.002457000000000000, -12659.43349358974,     -12820.45032051281,
    -481.6546474358953,    -0.0007886666666666667, -10255.82692307692,
    -3365.665865384615,    -13301.58413461538,     -0.0006066666666666667,
    -6248.854166666652,    -161.3189102564033,     -26282.13461538462
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
   3.52564102564102478e10, 0.00000000000000000,    0.00000000000000000,    52083.3333333333285, 
   0.00000000000000000,    3.52564102564102478e10, 0.00000000000000000,    52083.3333333333285,
   0.00000000000000000,    0.00000000000000000,    3.52564102564102478e10, 52083.3333333333285,
   0.00000000000000000,    0.00000000000000000,    0.00000000000000000,    454.999999999999943,
  -6.41025641025640965e9,  3.20512820512820482e9,  0.00000000000000000,    0.00000000000000000,
   4.80769230769230652e9, -2.24358974358974304e10, 4.80769230769230652e9,  52083.3333333333285, 
   0.00000000000000000,    3.20512820512820482e9, -6.41025641025640965e9,  0.00000000000000000, 
   0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   -151.666666666666657,
  -6.41025641025640965e9,  0.00000000000000000,    3.20512820512820482e9,  0.00000000000000000,
   0.00000000000000000,   -6.41025641025640965e9,  3.20512820512820482e9,  0.00000000000000000,
   4.80769230769230652e9,  4.80769230769230652e9, -2.24358974358974304e10, 52083.3333333333285,
   0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   -151.666666666666657,
   0.00000000000000000,    3.20512820512820482e9,  3.20512820512820482e9,  0.00000000000000000,
   4.80769230769230652e9,  0.00000000000000000,   -8.01282051282051086e9,  26041.6666666666642,
   4.80769230769230652e9, -8.01282051282051086e9,  0.00000000000000000,    26041.6666666666642,
   0.00000000000000000,    0.00000000000000000,   0.00000000000000000,     0.00000000000000000
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
   -18669.5957532051252, -14903.7055288461488, -19551.1466346153829, -0.000515666666666666552,
   -2604.08854166666652,  8012.67988782051179,  4206.79326923076951,  0.000151666666666666649,
    1562.59114583333439, -6370.14803685897277, -1682.82772435897550, -0.000151666666666666649,
   -2804.38040865384437, -200.364783653846530, -4927.94711538461343, -0.000174416666666666633
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
   -138461.538461538410,    -151923.076923076878,     -319230.769230769132,
   -0.00552066666666666504,  47435.8974358974156,     -33333.3333333333358,
    55769.2307692307368,     0.000849333333333333342, -641.025641025629739,
    1923.07692307693378,     -11538.4615384615317,    -0.0000606666666666664969,
   -1282.05128205127858,    -18909.6314102564065,     -18589.7435897435789,
    0.000181999999999999979, 40063.4775641025481,      66666.6666666666570,
    11217.4487179487187,     0.000727999999999999915, -26282.0512820512740,
   -28525.1410256410236,    -1282.05128205128494,     -0.000545999999999999936,
   -77564.1025641025335,     23717.9487179487187,      165705.857371794817,
    0.00175933333333333280, -641.025641025644290,     -18589.7435897435826,
   -58012.4663461538148,    -0.000424666666666666617,  44871.0657051281887,
    41666.3124999999854,     55769.2307692307659,      0.00145599999999999983,
   -85897.4358974358765,     5449.21794871795646,     -3525.28685897435935,
   -0.000545999999999999827, 6089.24358974358620,      30128.2051282051179,
    61538.6073717948602,     0.000970666666666666444,  39743.2355769230635,
    20192.1618589743448,     14743.5897435897496,      0.000970666666666666336
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
  ScalarFunction<::Plato::Thermomechanics<spaceDim>> 
    scalarFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("Objective"));

  // compute and test objective value
  //
  auto value = scalarFunction.value(state, z);

  double value_gold = 3.99325969691123239;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(state, z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
   -149357.8701923077,   -119230.2067307692,   -156409.7147435897,
   -0.1760003333333333,  -139421.5977564102,   -125960.8092948718,
   -132692.2243589743,   -0.2540039999999999,   12500.40624999999,
   -50961.31971153845,   -13462.16346153846,   -0.06371333333333332,
   -161536.3365384615,   -77563.76923076922,   -205128.3301282051,
   -0.3903306666666666,  -25319.68990384614,   -25640.96314102563,
   -962.4238782051143,   -0.1682440000000000,  -20512.23717948717,
   -6731.050480769230,   -26602.86618589742,   -0.07413000000000002,
   -12498.85416666665,   -321.5753205128094,   -52564.18589743588,
   -0.08460733333333334, -21473.91105769230,    11537.62820512823,
   -13140.64022435896,   -0.05244733333333334, -109934.6370192308,
    44872.06570512819,    61218.95913461538,   -0.2585966666666666
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
    0.3006646220307975,  0.2828518382060641,  0.07071297869214102,
    0.2014571253646834,  0.06703482120772307, 0.03167447648441282,
    0.05844522383912947, 0.02116164326532563, 0.1211208163690948,
    0.3125138830947090,  0.1180923674785154,  0.02885361640063332,
    0.04527148502001922, 0.5649346204235975,  0.2730600666633808,
    0.1003915937669949,  0.1293972579725719,  0.09587150695223083,
    0.02509555855275397, 0.05871878443368219, 0.2281261532124168,
    0.3215640965314669,  0.1200080860211962,  0.006538731463594921,
    0.02452832023762705, 0.3393603844997411,  0.04580963872672825
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
    1.415436782967118,     -1.088392815648369,   -0.5040375881616612,
    1.447588668646153,     -1.250091771703753,    0.1578841810878153,
    0.1306132382529746,    -0.3847761196036412,   0.5916655752193332,
    1.020192508092308,      0.5763455956072205,  -0.2397432740062565,
   -0.2497054688307384,     0.04301339975588712,  0.4116322178948820,
   -0.02582992877120004,    0.1731726564366770,   0.1257363320594461,
   -9.21562170871081942e-5, 0.4083635774612511,  -0.1133975593030360,
   -0.02791564539653321,    0.1617290307673025,  -0.07330046252834882,
   -0.6810599201684920,    -0.1161559894293126,  -0.2451581005350564,
  };



  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-11);
  }
}

