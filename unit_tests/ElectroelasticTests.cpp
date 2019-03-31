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
#include "plato/ApplyWeighting.hpp"
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

  // create objective
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunction<::Plato::Electromechanics<spaceDim>> 
    scalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));

  // compute and test objective value
  //
  auto value = scalarFunction.value(state, z);

  double value_gold = -10.20702939102564;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(state, z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
   122450.8547008547 ,  -25758.97435897436 ,  283731.7307692308  ,
   250747.9166666666 ,   -8417.147435897419,   -8136.698717948726,
   273473.0769230769 ,  260595.3125000000  , -127139.5833333333  ,
    22213.30128205129,   81045.94017094019 ,   85202.60416666667 ,
    11334.6153846154 ,  -25095.08547008547 ,  356182.7457264956  ,
   309031.7708333333 , -131602.5106837607  ,  -17275.8547008547  ,
   105227.8846153846 ,  116064.5833333333  ,  -42281.5170940171  ,
   -10858.97435897436,   84345.08547008547 ,   82531.77083333334 ,
    12602.08333333334,    3872.489316239319,   86798.66452991453
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
    -0.8430550507478631 , -0.7908609882478632 , -0.1977152470619658 ,
    -0.4780779513888889 , -0.1458235363247863 , -0.07292542334401711,
    -0.1529542601495726 , -0.05154157585470086, -0.2572132959401709 ,
    -0.7237269577991452 , -0.2930817094017093 , -0.03877574652777775,
    -0.06204588675213674, -1.496178314636752  , -0.739614564636752  ,
    -0.2290521327457265 , -0.2725698637820512 , -0.1600898250534188 ,
    -0.06201914262820514, -0.1523734802350428 , -0.5695864583333333 ,
    -0.9150726041666666 , -0.3577581663995726 , -0.01523024038461538,
    -0.04391547409188033, -0.9621676255341879 , -0.1236038688568376
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
   -3.181933589743589  ,  3.42746267094017   ,  1.568700341880342  ,
   -3.098169262820512  ,  3.484757275641026  , -0.3676219871794872 ,
   -0.7207964850427347 ,  1.117182489316239  , -1.699597670940171  ,
   -2.50306329059829   , -1.433395384615384  ,  0.5848191452991456 ,
   -0.3402299465811963 , -0.2176012179487179 , -1.001196837606838  ,
   -0.4493456410256411 , -0.5027489743589745 , -0.2938962820512822 ,
   -0.2926587286324785 , -1.091194903846154  ,  0.3044784615384616 ,
   -0.05198638888888892, -0.3821194658119658 ,  0.1592865384615385 ,
    1.758668803418803  ,  0.6097938675213675 ,  0.7044508333333332 ,
   -2.886511602564103  ,  0.4212044017094013 ,  1.442545363247864  ,
   -1.680306944444444  ,  2.308642649572649  , -0.2053918055555554 ,
   -0.1547276495726495 , -0.3180604273504272 , -0.1735164529914529 ,
   -0.01515108974358978, -0.3576426602564101 ,  0.3837701602564101 ,
   -1.929620790598291  , -0.418803482905982  , -1.517785790598291  ,
   -0.6458789423076929 , -0.5082928418803421 , -2.257961816239316  ,
    0.6335001816239321 , -0.4486205662393158 , -0.6258619337606837 ,
    1.487866698717948  , -0.25790141025641   ,  0.3374094123931625 ,
   -0.4796918482905985 , -0.4090008012820514 ,  0.325530395299145  ,
    0.01062425213675221, -0.1338604273504275 , -0.1796057478632478 ,
   -0.3101020085470085 , -0.08645029914529914, -0.4290325213675215 ,
   -0.1030446153846147 , -3.039669722222223  ,  0.5709875961538466 ,
    5.562050181623929  ,  1.57176017094017   , -0.4314812927350423 ,
    3.097139893162394  ,  0.8434269017094015 , -1.345133087606838  ,
    0.02014162393162398, -0.03959641025641031, -0.06185200854700852,
    0.03494016025641028, -0.1669968589743589 , -0.2311284829059827 ,
    7.126409999999999  , -3.390138643162393  ,  3.524454561965812  ,
   -0.8881229700854699 , -0.5821359294871788 ,  0.9146309081196579
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}

