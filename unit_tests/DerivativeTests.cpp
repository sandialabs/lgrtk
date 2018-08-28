/*!
  These unit tests are for the Derivative functionality.
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
#include "LinearElasticMaterial.hpp"

#ifdef HAVE_VIENNA_CL
#include "ViennaSparseLinearProblem.hpp"
#endif

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
#include <plato/SimplexMechanics.hpp>
#include <plato/WorksetBase.hpp>
#include <plato/AbstractVectorFunction.hpp>
#include <plato/VectorFunction.hpp>
#include <plato/AbstractScalarFunction.hpp>
#include <plato/InternalElasticEnergy.hpp>
#include <plato/ScalarFunction.hpp>
#include "ApplyConstraints.hpp"
#include "PlatoProblem.hpp"
#include "plato/Mechanics.hpp"
#include "plato/Thermal.hpp"

#include <fenv.h>


using namespace lgr;

TEUCHOS_UNIT_TEST( DerivativeTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int numCells = mesh->nelems();
  int nodesPerCell  = Plato::SimplexMechanics<spaceDim>::m_numNodesPerCell;
  int numVoigtTerms = Plato::SimplexMechanics<spaceDim>::m_numVoigtTerms;
  int dofsPerCell   = Plato::SimplexMechanics<spaceDim>::m_numDofsPerCell;

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( spaceDim*mesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);


  WorksetBase<Plato::SimplexMechanics<spaceDim>> worksetBase(*mesh);

  Plato::ScalarArray3DT<Plato::Scalar>
    gradient("gradient",numCells,nodesPerCell,spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    strain("strain",numCells,numVoigtTerms);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stress("stress",numCells,numVoigtTerms);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    result("result",numCells,dofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar>
    configWS("config workset",numCells, nodesPerCell, spaceDim);
  worksetBase.worksetConfig(configWS);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stateWS("state workset",numCells, dofsPerCell);
  worksetBase.worksetState(u, stateWS);

  Plato::ComputeGradientWorkset<spaceDim> computeGradient;
  Strain<spaceDim>                        voigtStrain;

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Material Model'>                                       \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                           \n"
    "      <Parameter name='Poissons Ratio' type='double' value='0.3'/>            \n"
    "      <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>          \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );


  Plato::ElasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create();
  auto m_cellStiffness = materialModel->getStiffnessMatrix();

  LinearStress<spaceDim>      voigtStress(m_cellStiffness);
  StressDivergence<spaceDim>  stressDivergence;

  Plato::Scalar quadratureWeight = 1.0/6.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    voigtStrain(cellOrdinal, strain, stateWS, gradient);

    voigtStress(cellOrdinal, stress, strain);

    stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
  }, "gradient");


  // test gradient
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

  int numGoldCells=gradient_gold.size();
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

  // test strain
  //
  auto strain_Host = Kokkos::create_mirror_view( strain );
  Kokkos::deep_copy( strain_Host, strain );

  std::vector<std::vector<double>> strain_gold = { 
    {0.0006, 0.0048, 0.0024, 0.0072, 0.003 , 0.0054},
    {0.006 , 0.0048,-0.0030, 0.0018, 0.003 , 0.0108},
    {0.006 , 0.0012, 0.0006, 0.0018, 0.0066, 0.0072},
    {0.012 ,-0.0048, 0.0006,-0.0042, 0.0126, 0.0072},
    {0.006 , 0.0012, 0.0006, 0.0018, 0.0066, 0.0072},
    {0.006 , 0.0012, 0.0006, 0.0018, 0.0066, 0.0072}
  };

  for(int iCell=0; iCell<int(strain_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
      if(strain_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(strain_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(strain_Host(iCell,iVoigt), strain_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  // test stress
  //
  auto stress_Host = Kokkos::create_mirror_view( stress );
  Kokkos::deep_copy( stress_Host, stress );

  std::vector<std::vector<double>> stress_gold = { 
   { 4961.538461538461, 8192.307692307691, 6346.153846153846, 2769.230769230769, 1153.846153846154, 2076.923076923077 },
   { 9115.384615384613, 8192.307692307690, 2192.307692307691, 692.3076923076922, 1153.846153846154, 4153.846153846153 },
   { 9115.384615384612, 5423.076923076921, 4961.538461538460, 692.3076923076922, 2538.461538461539, 2769.230769230769 },
   { 13730.76923076923, 807.6923076923071, 4961.538461538460,-1615.384615384614, 4846.153846153846, 2769.230769230769 },
   { 9115.384615384612, 5423.076923076921, 4961.538461538460, 692.3076923076924, 2538.461538461539, 2769.230769230769 },
   { 9115.384615384612, 5423.076923076921, 4961.538461538460, 692.3076923076922, 2538.461538461539, 2769.230769230769 }
  };

  for(int iCell=0; iCell<int(stress_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
      if(stress_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(stress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(stress_Host(iCell,iVoigt), stress_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  // test residual
  //
  auto result_Host = Kokkos::create_mirror_view( result );
  Kokkos::deep_copy( result_Host, result );

  std::vector<std::vector<double>> result_gold = { 
   {-86.53846153846153, -341.3461538461538, -115.3846153846154,   158.6538461538462,
    -28.84615384615385, -216.3461538461538, -120.1923076923077,   254.8076923076923,
     67.30769230769231,   48.07692307692308, 115.3846153846154,   264.4230769230770},
   {-173.0769230769231, -341.3461538461538,  -28.84615384615385,  125.0000000000000,
     312.5000000000000,  -62.49999999999994,-331.7307692307691,  -144.2307692307692,
      43.26923076923080, 379.8076923076923,  173.0769230769231,    48.07692307692308},
   {-105.7692307692308,  -28.84615384615385,-206.7307692307692,  -264.4230769230767,
     110.5769230769231,  -76.92307692307692,  -9.615384615384613,-197.1153846153846,
     177.8846153846154,  379.8076923076923,  115.3846153846154,   105.7692307692308},
   {-201.9230769230769,   67.30769230769229,-206.7307692307692,   456.7307692307693,
      81.73076923076928, 269.2307692307692,  115.3846153846154,    33.65384615384622,
     -67.30769230769229,-370.1923076923075, -182.6923076923077,     4.807692307692264},
   {-115.3846153846154, -225.9615384615384,  -28.84615384615384,  274.0384615384615,
      86.53846153846152,-100.9615384615383, -264.4230769230767,   110.5769230769230,
     -76.92307692307688, 105.7692307692307,   28.84615384615384,  206.7307692307692},
   {-115.3846153846154, -225.9615384615384,  -28.84615384615384,    9.615384615384613,
     197.1153846153846, -177.8846153846153, -274.0384615384614,   -86.53846153846155,
     100.9615384615384,  379.8076923076923,  115.3846153846154,   105.7692307692308}
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
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, ElastostaticResidual3D )
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


  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( spaceDim*mesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);



  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elastostatics'/>      \n"
    "  <Parameter name='Objective' type='string' value='Internal Elastic Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Elastostatics'>                                        \n"
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

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  VectorFunction<::Plato::Mechanics<spaceDim>> 
    esVectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = esVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<double> residual_gold = { 
    -1903.846153846153,  -894.2307692307692,-1038.461538461538,
    -2062.499999999999, -1024.038461538461,  -692.3076923076922,
     -379.8076923076920, -379.8076923076922,  182.6923076923077,
    -2379.807692307691,  -793.2692307692305, -894.2307692307687,
     -798.0769230769225, -235.5769230769230,  283.6538461538459,
     -538.4615384615381,  -19.23076923076923, -19.23076923076923,
     -605.7692307692301,  259.6153846153844, -259.6153846153845,
     -173.0769230769229,  173.0769230769230, -173.0769230769230,
     -485.5769230769228,  336.5384615384618, -139.4230769230768,
     -615.3846153846150,-1120.192307692307, -1754.807692307692,
     -264.4230769230765, -610.5769230769226, -394.2307692307692,
        0.0000000000000,    0.0000000000000, -346.1538461538459,
       28.84615384615405, 374.9999999999998, -317.3076923076922,
     1274.038461538463,  -673.0769230769218, -312.4999999999985,
     1341.346153846153,  -302.8846153846144,  663.4615384615385,
      913.4615384615381,  668.2692307692305,  552.8846153846155,
     1033.653846153846,  1336.538461538461,   514.4230769230774,
      437.5000000000005,  379.8076923076925,  451.9230769230770,
      451.9230769230770,  221.1538461538464,  221.1538461538462,
      302.8846153846157, -490.3846153846151,   72.11538461538484,
      971.1538461538465, -399.0384615384608,  783.6538461538462,
     1269.230769230769,   721.1538461538468,  721.1538461538469,
      658.6538461538461,   96.15384615384637, 572.1153846153854,
       48.07692307692318, 134.6153846153847,   48.07692307692324,
       62.49999999999966, 365.3846153846159,  149.0384615384621,
     1365.384615384615,  1610.576923076923,   860.5769230769234,
       48.07692307692358, 264.4230769230770,  264.4230769230767
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, u. (i.e., jacobian)
  //
  auto jacobian = esVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    3.52564102564102504e+05, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 3.52564102564102563e+05, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 3.52564102564102563e+05, 

   -6.41025641025641016e+04, 3.20512820512820508e+04, 0.00000000000000000e+00,
    4.80769230769230708e+04,-2.24358974358974316e+05, 4.80769230769230708e+04,
    0.00000000000000000e+00, 3.20512820512820508e+04,-6.41025641025641016e+04, 

   -6.41025641025641016e+04, 0.00000000000000000e+00, 3.20512820512820508e+04,
    0.00000000000000000e+00,-6.41025641025641016e+04, 3.20512820512820508e+04, 
    4.80769230769230708e+04, 4.80769230769230708e+04,-2.24358974358974316e+05,

    0.00000000000000000e+00, 3.20512820512820508e+04, 3.20512820512820508e+04,
    4.80769230769230708e+04, 0.00000000000000000e+00, -8.01282051282051252e+04,
    4.80769230769230708e+04,-8.01282051282051252e+04, 0.00000000000000000e+00, 

    0.00000000000000000e+00,-8.01282051282051252e+04, 4.80769230769230708e+04,
   -8.01282051282051252e+04, 0.00000000000000000e+00, 4.80769230769230708e+04, 
    3.20512820512820508e+04, 3.20512820512820508e+04, 0.00000000000000000e+00
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = esVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
    -475.9615384615383,  -223.5576923076923,   -259.6153846153846, 
       1.201923076923094, 141.8269230769231,      1.201923076923091, 
     -94.95192307692304,  -94.95192307692307,    45.67307692307691, 
    -149.0384615384614,    -8.413461538461540,   -8.413461538461529, 
      -8.413461538461519,  -8.413461538461512, -149.0384615384615, 
     341.3461538461538,    88.94230769230769,   125.0000000000000, 
     123.7980769230769,   -16.82692307692301,   123.7980769230769, 
     262.0192307692307,   121.3942307692307,    121.3942307692308
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = esVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<lgr::Scalar> gold_grad_x_entries = {
  -4153.84615384615245,   -2278.84615384615336,  -3192.30769230769147, 
   1423.07692307692287,    -500.000000000000000,   557.692307692307281, 
    -19.2307692307692832,    28.8461538461539817, -115.384615384615515, 
      9.61538461538469846, -153.846153846154266,  -307.692307692307509, 
    586.538461538461434,    999.999999999999773,   355.769230769230717, 
   -480.769230769230717,   -730.769230769230717,    67.3076923076923208, 
   -403.846153846153470,    423.076923076922867,  1028.84615384615358, 
    -96.1538461538464730,  -230.769230769230717,  -701.923076923076565, 
   1384.61538461538430,     692.307692307692150,   557.692307692307395, 
  -1134.61538461538430,    -451.923076923076678,  -182.692307692307651, 
    586.538461538461434,    403.846153846153697,   557.692307692307622, 
    990.384615384615017,    490.384615384615472,    67.3076923076923208
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalElasticEnergy3D )
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
  ScalarFunction<::Plato::Mechanics<spaceDim>> 
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
    -3807.692307692307, -1788.461538461538, -2076.923076923077,
    -4125.000000000000, -2048.076923076922, -1384.615384615384,
     -759.6153846153842, -759.6153846153843,  365.3846153846152,
    -4759.615384615382, -1586.538461538461, -1788.461538461537,
    -1596.153846153845,  -471.1538461538461,  567.3076923076919,
    -1076.923076923076,   -38.46153846153851, -38.46153846153845,
    -1211.538461538461,   519.2307692307691, -519.2307692307688,
     -346.1538461538458,  346.1538461538459, -346.1538461538460,
     -971.1538461538457,  673.0769230769235, -278.8461538461537,
    -1230.769230769230, -2240.384615384614, -3509.615384615385,
     -528.8461538461531,-1221.153846153846,  -788.4615384615381,
        0.0000000000000,    0.0000000000000, -692.3076923076916,
       57.69230769230813, 749.9999999999995, -634.6153846153845,
     2548.076923076926, -1346.153846153844,  -624.9999999999970,
     2682.692307692307,  -605.7692307692291, 1326.923076923077,
     1826.923076923077,  1336.538461538461,  1105.769230769231,
     2067.307692307692,  2673.076923076923,  1028.846153846155,
      875.0000000000007,  759.6153846153850,  903.8461538461543,
      903.8461538461540,  442.3076923076925,  442.3076923076924,
      605.7692307692316, -980.7692307692299,  144.2307692307695,
     1942.307692307694,  -798.0769230769212, 1567.307692307693,
     2538.461538461539,  1442.307692307693,  1442.307692307693,
     1317.307692307692,   192.3076923076927, 1144.230769230770,
       96.15384615384613, 269.2307692307692,   96.15384615384653,
      124.9999999999997,  730.7692307692319,  298.0769230769239,
     2730.769230769230,  3221.153846153846,  1721.153846153847,
       96.15384615384704, 528.8461538461538,  528.8461538461534
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



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, StressPNorm3D )
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
    "  <Parameter name='Objective' type='string' value='Stress P-Norm'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Stress P-Norm'>                                        \n"
    "    <Parameter name='Exponent' type='double' value='12.0'/>                   \n"
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
  ScalarFunction<::Plato::Mechanics<spaceDim>> 
    eeScalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));


  // compute and test objective value
  //
  auto value = eeScalarFunction.value(u,z);

  double value_gold = 14525.25169157000;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(u,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
   -1503430.743610086,    -87943.46429698351,  -253405.7951760115,
    -419311.2039113952,   -45900.11226120282,   -36121.19195937364,
    -127338.4661959158,   -19117.73429961895,    92407.24120276771,
     -50087.19124766427,   12774.33242060617,    -3333.510853871279,
     -23558.49126781739,    3380.937063817981,   14893.84056534542,
      -9519.888210062301,   3348.092294628147,    3560.944140991056,
      -5240.218786677828,   8738.583477630442,   -4060.103637689512,
        -47.36161489014854,  173.0873916025785,   -137.7668652438236,
      -5174.016020690595,  26242.73416532713,   -14003.94012109160
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
   2972.321313190315,        831.6877699932147,    207.9219424983036,
     85.16009409571750,       30.23970839774167,     9.793975869956283,
     10.67428715222397,        0.1739233563059383,  32.70029626817538,
   1960.655369272931,          3.837915000705916,    0.06595545018012441,
      0.1109427260128550,   3191.518788255666,     633.0928752380449,
     29.38214582086229,       11.06169917284905,     0.3508304842757221,
      0.002762687660171751,    0.002953941115371589, 4.977103004996600,
   1375.578043139706,        394.9305857993233,      9.661752822222865e-7,
      6.955608440499901e-4, 2737.188926384874,       1.820787841892929
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
    25140.94641961542,   -14100.50148913174,     1367.291551437471,
    5454.778632091190,    -2261.028458537603,     300.7995048791828,
     515.4758244370678,    -264.9379458613482,    171.0440039078628,
     207.3155368336267,      90.12589972024526,   -18.43667104475125,
      14.65117107126713,      9.528794705847330,   16.35882582027728,
       9.135793400021047,     6.459878463616483,    3.202460912352750,
      -0.3251353455947416,    7.854573864154236,   -3.859633813615325,
       0.01424862142881591,   0.1087242510851228,  -0.06517564756324878,
     -40.80884087925770,    -15.10656437218265,    -1.627492863831697
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         EffectiveEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, EffectiveEnergy3D_NormalCellProblem )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto numVerts = mesh->nverts();

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( numVerts, 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);



  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                         \n"
    "  <Parameter name='Objective' type='string' value='Effective Energy'/>                       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                                 \n"
    "  <ParameterList name='Effective Energy'>                                                    \n"
    "    <Parameter name='Assumed Strain' type='Array(double)' value='{1.0,0.0,0.0,0.0,0.0,0.0}'/>\n"
    "    <ParameterList name='Penalty Function'>                                                  \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                                 \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                                    \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Cell Problem Forcing'>                                                \n"
    "    <Parameter name='Column Index' type='int' value='0'/>                                    \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Material Model'>                                                      \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                                          \n"
    "      <Parameter name='Poissons Ratio' type='double' value='0.3'/>                           \n"
    "      <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                         \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "</ParameterList>                                                                             \n"
  );

  Plato::ScalarVectorT<Plato::Scalar> solution("solution",spaceDim*numVerts);

  // create mesh based displacement
  //
  std::vector<Plato::Scalar> solution_gold = {
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0173669389188933626, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0411192268700119809,  0.00965747852017450475, 
    0.0000000000000000000,  0.0355244194737336372,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.00494803288197820032, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0511217432387267717,  0.0000000000000000000, 
   -0.0739957825759057081, -0.0162917876901311660,  0.0000000000000000000, 
    0.0000000000000000000, -0.0427458935980062973,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.142903245392837525,   0.0000000000000000000,  0.0000000000000000000, 
   -0.0172027445162594265,  0.00106322367588203379, 0.00106322367588205396, 
    0.0406024627433991536,  0.0154135960960958291,  0.0000000000000000000, 
    0.149780596048765896,   0.0000000000000000000,  0.0000000000000000000, 
    0.148483915692292107,   0.0000000000000000000,  0.00236800003074055191, 
    0.0000000000000000000,  0.0000000000000000000, -0.0113344145950257241, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000, -0.0450892189148590500,  0.0000000000000000000, 
    0.0000000000000000000, -0.0439820074688949750, -0.0125202591190575266, 
   -0.181877235524798508,   0.0000000000000000000, -0.00324619162477584461, 
   -0.117196040731724044,   0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000, -0.0121695070062508570, 
   -0.225131719618738901,   0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000 };


  // push gold data from host to device
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(solution_gold.data(), solution_gold.size());
  auto tDeviceView = Kokkos::create_mirror_view(solution);
  Kokkos::deep_copy(tDeviceView, tHostView);
  Kokkos::deep_copy(solution, tDeviceView);


  // create objective
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunction<::Plato::Mechanics<spaceDim>> 
    eeScalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));


  // compute and test objective value
  //
  auto value = eeScalarFunction.value(solution,z);

  double value_gold = 1346153.84615384578;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(solution,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
    112179.4871794871,      48076.92307692306,    48076.92307692306,
    168269.2307692307,      72115.38461538460,    0.000000000000000, 
    56089.74358974357,      24038.46153846153,   -48076.92307692306, 
    336538.4615384614,      0.000000000000000,    0.000000000000000, 
    168269.2307692307,      0.000000000000000,   -72115.38461538460, 
    112179.4871794871,     -24038.46153846153,   -24038.46153846153, 
    168269.2307692307,     -72115.38461538460,    0.000000000000000, 
    56089.74358974357,     -48076.92307692306,    24038.46153846153, 
    168269.2307692307,      0.000000000000000,    72115.38461538460,
   -1.455191522836685e-11,  0.000000000000000,    144230.7692307692, 
   -168269.2307692307,      0.000000000000000,    72115.38461538460, 
   -56089.74358974357,     -24038.46153846153,    48076.92307692306, 
    0.000000000000000,     -72115.38461538460,    72115.38461538460,
   -4.365574568510056e-11,  0.000000000000000,    0.000000000000000, 
   -1.455191522836685e-11,  0.000000000000000,   -144230.7692307692, 
    0.000000000000000,     -72115.38461538460,   -72115.38461538460,
   -1.455191522836685e-11, -144230.7692307692,    0.000000000000000,
   -168269.2307692307,     -72115.38461538460,    0.000000000000000, 
   -112179.4871794871,     -48076.92307692306,   -48076.92307692306, 
   -168269.2307692307,      0.000000000000000,   -72115.38461538460, 
   -336538.4615384614,      0.000000000000000,    0.000000000000000,
   -1.455191522836685e-11,  144230.7692307692,    0.000000000000000,
    0.000000000000000,      72115.38461538460,   -72115.38461538460,
   -56089.74358974357,      48076.92307692306,   -24038.46153846153,
   -168269.2307692307,      72115.38461538460,    0.000000000000000,
    0.000000000000000,      72115.38461538460,    72115.38461538460, 
   -112179.4871794871,      24038.46153846153,    24038.46153846153
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(fabs(grad_u_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(solution,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
    51415.0373984630496, 63626.4105765648710, 14999.0837995235233, 
    76149.3934262146504, 24360.5475584209853, 10370.2306261117192,
    20703.0116870963320, 10506.9555951488292, 53433.8245831477980,
    92117.0824403273000, 21902.6070614885975, 14770.1235645518063,
    31203.5021472171320, 188920.219746857270, 92794.1264836712799,
    55120.4014680251421, 87513.1858953364572, 60597.6279469306246,
    47551.5814146090779, 50405.0419535135588, 69397.7622101499728,
    86836.1418519924773, 31203.5021472171284, 10277.9953601771122,
    18245.0711901639443, 53355.1550102794354, 8378.22301064559724 };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(solution,z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   -106445.574316611834,     -157158.660963500093,     -144060.939883280807, 
   -163071.856024346896,     -222065.945634533884,     -4.54747350886464119e-11, 
   -55216.7877938377933,     -70071.6843249799276,      118322.772423217699, 
   -335191.527413369680,     -2.54658516496419907e-11, -8.52651282912120223e-12,
   -168055.897552366805,     -1.81898940354585648e-12,  143098.431691335514, 
   -116566.465917270718,      39525.3193318040503,      40995.3379179461335, 
   -174216.476848969964,      118267.067436839250,      4.54747350886464119e-12,
   -58309.6335107410923,      79140.0925266976701,     -42516.7069225371306, 
   -169079.626776330930,      1.09139364212751389e-11, -168056.776846833644,
   -4.84305928694084287e-11,  4.36557456851005554e-11, -373311.568021968415, 
    167903.506027127150,     -3.63797880709171295e-12, -127195.711052508646,
    54579.5763299848841,      72665.4801528035023,     -117071.140959200449, 
    6.82121026329696178e-12,  185243.311753953109,     -177461.297782159119, 
   -1.45519152283668518e-11,  7.27595761418342590e-11,  4.18367562815546989e-11,
    2.91038304567337036e-11,  1.45519152283668518e-11,  362612.108049641713, 
    0.000000000000000000,     152077.959759364254,      166904.448650544538,
    2.91038304567337036e-11,  336527.149233340984,      0.00000000000000000,
    162587.036085733649,      217499.896431886649,     -2.18278728425502777e-11, 
    106750.357367091114,      145207.569527156214,      132109.848446936900, 
    168594.806837717682,      1.73727698893344495e-11,  163490.727644186351, 
    336465.950341075426,     -4.00177668780088425e-11,  0.00000000000000000, 
    2.91038304567337036e-11, -347226.609205667686,     -1.45519152283668518e-11,
    7.27595761418342590e-12, -189194.940956436098,      173509.668579676159, 
    57672.4220468881977,     -77888.4610626804642,      45110.5027503607125, 
    174064.085323730309,     -102364.346798012397,     -1.45519152283668518e-11, 
    2.18278728425502777e-11, -148763.542020734254,     -163590.030911914480, 
    117536.105794497213,     -31419.6551873009557,     -32889.6737734430353
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(fabs(grad_x_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         EffectiveEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, EffectiveEnergy3D_ShearCellProblem )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto numVerts = mesh->nverts();

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( numVerts, 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);



  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                         \n"
    "  <Parameter name='Objective' type='string' value='Effective Energy'/>                       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                                 \n"
    "  <ParameterList name='Effective Energy'>                                                    \n"
    "    <Parameter name='Assumed Strain' type='Array(double)' value='{0.0,0.0,0.0,1.0,0.0,0.0}'/>\n"
    "    <ParameterList name='Penalty Function'>                                                  \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                                 \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                                    \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Cell Problem Forcing'>                                                \n"
    "    <Parameter name='Column Index' type='int' value='3'/>                                    \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Material Model'>                                                      \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                                          \n"
    "      <Parameter name='Poissons Ratio' type='double' value='0.3'/>                           \n"
    "      <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                         \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "</ParameterList>                                                                             \n"
  );

  Plato::ScalarVectorT<Plato::Scalar> solution("solution",spaceDim*numVerts);

  // create mesh based displacement
  //
  std::vector<Plato::Scalar> solution_gold = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125000000000000028, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.43954359943216618e-18, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0.125000000000000028, 
-7.03590337326575184e-18, -5.50739874438680158e-18, 0, 0, 
2.12992705742669426e-18, 0, 0, 0, 0, 1.11083808367229628e-18, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125000000000000028, 0, 0, 0, 
-2.36483063822016535e-18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0
  };

  // push gold data from host to device
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(solution_gold.data(), solution_gold.size());
  auto tDeviceView = Kokkos::create_mirror_view(solution);
  Kokkos::deep_copy(tDeviceView, tHostView);
  Kokkos::deep_copy(solution, tDeviceView);


  // create objective
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunction<::Plato::Mechanics<spaceDim>> 
    eeScalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));


  // compute and test objective value
  //
  auto value = eeScalarFunction.value(solution,z);

  double value_gold = 384615.384615384275;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(solution,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
0, 32051.28205128205, 32051.28205128205, 0, 0, 48076.92307692307, 0, 
-32051.28205128205, 16025.64102564102, 0, 0, 0, 0, 
-48076.92307692307, 0, 0, -16025.64102564102, -16025.64102564102, 0, 
0, -48076.92307692307, 0, 16025.64102564102, -32051.28205128205, 0, 
48076.92307692307, 0, 0, 96153.84615384616, 0, 0, 48076.92307692307, 
0, 0, 32051.28205128205, -16025.64102564102, 0, 48076.92307692307, 
-48076.92307692307, 0, 0, 7.275957614183426e-12, 0, 
-96153.84615384616, 0, 0, -48076.92307692307, -48076.92307692307, 0, 
0, -96153.84615384616, 0, 0, -48076.92307692307, 0, 
-32051.28205128205, -32051.28205128205, 0, -48076.92307692307, 0, 0, 
0, 0, 0, 0, 96153.84615384616, 0, -48076.92307692307, 
48076.92307692307, 0, -16025.64102564102, 32051.28205128205, 0, 0, 
48076.92307692307, 0, 48076.92307692307, 48076.92307692307, 0, 
16025.64102564102, 16025.64102564102
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(fabs(grad_u_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(solution,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
   12019.2307692307695, 16025.6410256410272, 4006.41025641025590, 
   24038.4615384615427, 8012.82051282051179, 4006.41025641025590, 
   8012.82051282051179, 4006.41025641025590, 16025.6410256410272, 
   24038.4615384615427, 8012.82051282051179, 4006.41025641025590, 
   8012.82051282051179, 48076.9230769230635, 24038.4615384615427, 
   16025.6410256410272, 24038.4615384615427, 16025.6410256410272, 
   12019.2307692307695, 16025.6410256410272, 24038.4615384615427, 
   24038.4615384615427, 8012.82051282051179, 4006.41025641025590, 
   8012.82051282051179, 16025.6410256410272, 4006.41025641025590
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(solution,z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    -32051.2820512820472, -32051.2820512820472, -32051.2820512820472, 
    -48076.9230769230708, -48076.9230769230708, 0, -16025.6410256410236, 
    -16025.6410256410236, 32051.2820512820472, -96153.8461538461561, 
    2.61113508235193827e-13, 0, -48076.9230769230708, 0, 
    48076.9230769230708, -32051.2820512820472, 16025.6410256410236, 
    16025.6410256410236, -48076.9230769230708, 48076.9230769230708, 0, 
    -16025.6410256410236, 32051.2820512820472, -16025.6410256410236, 
    -48076.9230769230708, -2.61113508235193827e-13, 
    -48076.9230769230708, -4.89905329768894236e-14, 0, 
    -96153.8461538461561, 48076.9230769230708, 0, -48076.9230769230708, 
    16025.6410256410236, 16025.6410256410236, -32051.2820512820472, 
    3.56037847330864148e-14, 48076.9230769230708, 
    -48076.9230769230708, 7.27595761418342590e-12, 
    -7.27595761418342590e-12, 0, 0, 0, 96153.8461538461561, 0, 
    48076.9230769230708, 48076.9230769230708, 0, 96153.8461538461561, 0, 
    48076.9230769230708, 48076.9230769230708, 0, 32051.2820512820472, 
    32051.2820512820472, 32051.2820512820472, 48076.9230769230708, 0, 
    48076.9230769230708, 96153.8461538461561, 0, 0, 0, 
    -96153.8461538461561, 0, 0, -48076.9230769230708, 
    48076.9230769230708, 16025.6410256410236, -32051.2820512820472, 
    16025.6410256410236, 48076.9230769230708, -48076.9230769230708, 0, 0, 
    -48076.9230769230708, -48076.9230769230708, 32051.2820512820472, 
    -16025.6410256410236, -16025.6410256410236
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(fabs(grad_x_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ThermostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, ThermostaticResidual3D )
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


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> t_host( mesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for( auto& val : t_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    t_host_view(t_host.data(),t_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), t_host_view);



  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Thermostatics'/>      \n"
    "  <Parameter name='Objective' type='string' value='Internal Thermal Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Thermostatics'>                                        \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Model'>                                       \n"
    "    <ParameterList name='Isotropic Linear Thermal'>                           \n"
    "      <Parameter name='Conductivity Coefficient' type='double' value='100.'/> \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  VectorFunction<::Plato::Thermal<spaceDim>> 
    tsVectorFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = tsVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<double> residual_gold = { 
    -56.66666666666666, -54.99999999999999, -16.66666666666667,
    -67.49999999999999, -21.66666666666666, -16.66666666666666,
    -17.50000000000000,  -5.000000000000000, 24.99999999999999,
    -67.50000000000000, -36.66666666666667,  -9.99999999999999,
      2.499999999999995,-24.99999999999998,  -5.000000000000021,
     11.66666666666666,  50.00000000000000,   3.333333333333323,
      5.000000000000004,  4.999999999999984, 60.00000000000004,
     69.99999999999999,  38.33333333333336,   6.666666666666675,
     16.66666666666663,  89.99999999999997,  16.66666666666668
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, u. (i.e., jacobian)
  //
  auto jacobian = tsVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
   49.99999999999999,-16.66666666666666,-16.66666666666666,
    0.0             ,  0.0             ,  0.0             ,
    0.0             ,-16.66666666666666,-16.66666666666666,
   83.33333333333330,-25.00000000000000,-16.66666666666666,
    0.0             ,  0.0             ,  0.0             ,
  -25.00000000000000,  0.0             ,-16.66666666666666,
   33.33333333333333, -8.33333333333333,  0.0             ,
   -8.33333333333333,  0.0             ,-25.00000000000000,
  150.00000000000000,-25.00000000000000,-25.00000000000000,
    0.0             ,-25.00000000000000,-49.99999999999999,
    0.0             ,  0.0             ,  0.0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient = tsVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
  -14.16666666666666,   4.166666666666666, -4.166666666666666, 
   -4.791666666666666, -4.791666666666666,  2.500000000000000, 
    6.666666666666666, 14.58333333333333,  -0.4166666666666666, 
  -13.75000000000000,  -3.125000000000000, -4.166666666666667, 
   -3.541666666666666,  0.4166666666666679, 1.249999999999997, 
   15.62500000000000,   7.708333333333336, -0.4166666666666667, 
   -4.166666666666667, -1.666666666666667,  0.4166666666666656, 
    5.833333333333334, -1.875000000000000, -1.041666666666667, 
  -16.87500000000000,  -2.083333333333334, -3.750000000000000, 
   -4.166666666666666,  4.791666666666666, 10.83333333333333, 
    4.166666666666666,  4.166666666666666,  5.833333333333333
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 1.0e-14);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = tsVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<lgr::Scalar> gold_grad_x_entries = {
  -151.666666666666686,   23.3333333333333499,  -1.66666666666667229, 
     4.99999999999999467, 19.9999999999999929, -14.9999999999999964, 
    48.3333333333333286, -11.6666666666666714,  40.0000000000000071, 
   -15.0000000000000000,  26.6666666666666607,  26.6666666666666643
};

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }


}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalThermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalThermalEnergy3D )
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


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> u_host( mesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Thermostatics'/>      \n"
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
    "      <Parameter name='Conductivity Coefficient' type='double' value='100.'/> \n"
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

  double value_gold = 611.666666666666;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(u,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
 -113.3333333333333  , -110.0000000000000  , -33.33333333333334 , 
 -135.0000000000000  ,  -43.33333333333333 , -33.33333333333333 , 
  -35.00000000000000 ,   -9.999999999999998,  49.99999999999999 , 
 -135.0000000000000  ,  -73.33333333333334 , -19.99999999999999 , 
    4.999999999999993,  -49.99999999999996 , -10.00000000000003 , 
   23.33333333333333 ,  100.0000000000000  ,   6.666666666666636, 
   10.00000000000001 ,    9.99999999999997 , 120.0000000000001  , 
  140.0000000000000  ,   76.66666666666669 ,  13.33333333333336 , 
   33.33333333333329 ,  179.9999999999999  ,  33.33333333333336
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
   50.87500000000000 , 47.50000000000001 , 11.87500000000000 ,
   29.33333333333334 ,  8.625000000000000,  4.416666666666666,
    9.500000000000000,  3.000000000000000, 15.29166666666666 ,
   42.25000000000000 , 17.95833333333334 ,  1.749999999999999,
    2.916666666666666, 89.91666666666667 , 44.41666666666667 ,
   13.87500000000000 , 15.95833333333333 ,  8.125000000000002,
    3.708333333333334,  9.208333333333339, 34.58333333333334 ,
   54.87499999999999 , 21.62500000000001 ,  0.916666666666669,
    2.374999999999999, 58.91666666666666 ,  7.875000000000002
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
    189.0000000000000, -204.3333333333334,  -96.99999999999997,
    187.5000000000000, -208.5000000000001,   22.00000000000002,
     52.50000000000001, -67.50000000000000, 101.6666666666667,
    145.8333333333333,   86.49999999999997, -35.00000000000000,
     34.66666666666666,  14.83333333333334,  60.16666666666667,
     31.33333333333334,  31.00000000000001,  17.66666666666666,
     12.33333333333334,  65.33333333333334, -17.66666666666667,
      2.999999999999999, 22.00000000000000,  -9.000000000000002,
   -105.5000000000000,  -39.66666666666666, -45.50000000000001
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, FluxPNorm3D )
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


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> u_host( mesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);


  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Thermostatics'/>      \n"
    "  <Parameter name='Objective' type='string' value='Flux P-Norm'/>             \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Flux P-Norm'>                                          \n"
    "    <Parameter name='Exponent' type='double' value='12.0'/>                   \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Model'>                                       \n"
    "    <ParameterList name='Isotropic Linear Thermal'>                           \n"
    "      <Parameter name='Conductivity Coefficient' type='double' value='100.'/> \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create objective
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunction<::Plato::Thermal<spaceDim>> 
    scalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Objective"));


  // compute and test objective value
  //
  auto value = scalarFunction.value(u,z);

  double value_gold = 444.0866631427854;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(u,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<double> grad_u_gold = { 
  -121.7118130636210,       -8.402628906798824,    -2.660758498915989, 
    -0.03344798696042541,   -0.002037610215525936, -0.001781170602554662, 
    -0.003041532729695166,  -5.050930310677827e-6,  0.02320609420829947, 
   -75.88535722079789,      -1.308256330626368,    -0.00001183857667807545, 
     5.995437722669788e-6,   3.556025967524545,    -2.098814435019494, 
     0.001683242825810107,   0.005889361126491251, -0.0003613381503557746, 
     2.342203318998678e-6,   0.0002564538923140549, 0.9048692212497024, 
    14.99208220618024,       5.181659302537033,     2.904509920748619e-7, 
     6.118314328490756e-6, 186.8076430023642,       0.6349853856300433
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
  auto grad_z = scalarFunction.gradient_z(u,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
   105.2893297588999,        6.512518326309412,     1.628129581577354, 
     0.01125162852821403,    0.0007942956716896681, 0.0004666278049140509, 
     0.001684178438260436,   0.0001822327241451453, 0.009432160604870156, 
    84.39181736274085,       0.6670203358970709,    2.010529010655542e-6, 
     2.509074030348219e-6, 107.0962504385074,       4.884922663631097, 
     0.001399891275939550,   0.002617540834021655,  0.0003206243285508645, 
     9.696548053968441e-6,   0.0005276215833734041, 0.8443271287750858, 
    24.65690798374454,       3.255913149884657,     5.522257190584811e-8, 
     8.782235515878686e-7, 104.4974671615945,       0.3333672998334975
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(u,z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   539.2780958447977,      -344.3254644667109,         51.30519608139448, 
    30.35370155226329,      -13.44365133910082,         1.680525781359764, 
     9.557593722324338,      -4.257193045977536,        1.617571420834767, 
     0.06078243908477833,     0.04221148137571083,     -0.02386572317775400, 
     0.003666589190819278,    0.0007099999693335820,    0.0007931864745622613, 
     0.003251256001833290,    0.0009335797177886892,    0.00008997372650920974, 
     0.005150465829598891,    0.002200974863685567,    -0.002236017567744606, 
     4.461655107765414e-6,    0.0001204782967012946, 
    -0.0001168584633119755,  -0.04615901982588366,     -0.02941544225033991, 
     0.01672873982832506,   374.5747628696193,       -186.5714100889252, 
     5.431619301086632,       0.4840866327084146,      -3.961748259004554, 
     2.171832760560541,      -3.298465252935005e-7,     6.405629870294046e-6, 
     0.00001286593933992018, -1.777115766619513e-7, 
    -1.840364441525098e-6,   -7.574624338084612e-6, 
   -20.46907727348694,       44.81554408077304,        36.34671896862719, 
     8.398344211022307,      -1.192340496837429,        1.505360601912677, 
    -0.003055497909891946,   -0.0004267854516753923,    0.0003756639687016820, 
    -0.01084107572206273,    -0.002150213408694142,     0.003688080724534325, 
     0.0001490782155791348,  -0.00001664571881389717, 
     0.0006438775089781428,  -3.268801220809508e-6, 
     4.200746621914837e-6,    3.190216649208285e-6, 
    -0.00009288986744798501,  0.0004080010998888928, 
     0.0001365110296481171,  -0.2505471535468692,       2.887162020614315, 
    -1.173911589542639,     -61.19398924435590,         8.634730918059427, 
    13.58656329864013,      -19.64033440714168,         6.120046121120866, 
     0.04900105104713143,    -1.462766547177769e-7, 
     1.955457457226577e-7,    6.691130582506415e-8, 
    -3.073668111453556e-6,    4.342332759457572e-6, 
     4.344541561312716e-6, -861.2007333539822,        489.4700276363467, 
  -111.2467574119000,         0.09524778942965621,      1.809708187258011, 
    -1.269970591057625
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Volume3D )
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
    "  <Parameter name='Linear Constraint' type='string' value='Volume'/>          \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Volume'>                                               \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create objective
  //
  Plato::DataMap dataMap;
  Omega_h::MeshSets tMeshSets;
  ScalarFunction<::Plato::Mechanics<spaceDim>> 
    volScalarFunction(*mesh, tMeshSets, dataMap, *params, params->get<std::string>("Linear Constraint"));


  // compute and test objective value
  //
  auto value = volScalarFunction.value(u,z);

  double value_gold = 1.0;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test objective gradient wrt state, u
  //
  auto grad_u = volScalarFunction.gradient_u(u,z);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  // this ScalarFunction is the volume in the reference configation which is 
  // independent of the solution, u, so gradient_u is zero.
  std::vector<double> grad_u_gold( grad_u.size(), 0.0);

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = volScalarFunction.gradient_z(u,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<double> grad_z_gold = {
    0.03125000000000000, 0.04166666666666666, 0.01041666666666667,
    0.06250000000000000, 0.02083333333333333, 0.01041666666666667, 
    0.02083333333333333, 0.01041666666666667, 0.04166666666666666, 
    0.06250000000000000, 0.02083333333333333, 0.01041666666666667, 
    0.02083333333333333, 0.1249999999999999,  0.06250000000000000, 
    0.04166666666666666, 0.06250000000000000, 0.04166666666666666, 
    0.03125000000000000, 0.04166666666666666, 0.06250000000000000, 
    0.06250000000000000, 0.02083333333333333, 0.01041666666666667, 
    0.02083333333333333, 0.04166666666666666, 0.01041666666666667
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = volScalarFunction.gradient_x(u,z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   -0.08333333333333333, -0.08333333333333333, -0.08333333333333333, 
   -0.1250000000000000, -0.1250000000000000, 0, -0.04166666666666666, 
   -0.04166666666666666, 0.08333333333333333, -0.2500000000000000, 
    0.00000000000000000, 0.0, -0.1250000000000000, 0.0, 
    0.1250000000000000, -0.08333333333333333, 0.04166666666666666, 
    0.04166666666666666, -0.1250000000000000, 0.1250000000000000, 0.0, 
   -0.04166666666666666, 0.08333333333333333, -0.04166666666666666, 
   -0.1250000000000000, 0.0, -0.1250000000000000, 
    0.0000000000000000, 0.0000000000000000,
   -0.2500000000000000, 0.1250000000000000, 0.0, -0.1250000000000000, 
    0.04166666666666666, 0.04166666666666666, -0.08333333333333333, 0.0, 
    0.1250000000000000, -0.1250000000000000, 0.000000000000000000,
    0.0000000000000000, 0.0000000000000000,
    0.0000000000000000, 0.0, 0.2500000000000000, 0.0, 
    0.1250000000000000, 0.1250000000000000,    0.0000000000000000,
    0.2500000000000000, 0.0, 0.1250000000000000, 0.1250000000000000, 0.0, 
    0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 
    0.1250000000000000, 0.0, 0.1250000000000000, 0.2500000000000000, 
    0.0000000000000000, 0.0, 0.0000000000000000,
   -0.2500000000000000,  0.0000000000000000, 0.0, 
   -0.1250000000000000, 0.1250000000000000, 0.04166666666666666, 
   -0.08333333333333333, 0.04166666666666666, 0.1250000000000000, 
   -0.1250000000000000, 0.0, 0.0, -0.1250000000000000, -0.1250000000000000, 
   0.08333333333333333, -0.04166666666666666, -0.04166666666666666
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

