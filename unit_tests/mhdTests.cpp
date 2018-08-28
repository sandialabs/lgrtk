/*!
  These unit tests are for mhd.
*/

#include "LGRTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_YamlParameterListCoreHelpers.hpp>

#include "CrsMatrix.hpp"
#include "MeshFixture.hpp"
#include "StaticsTypes.hpp"

#include "PlatoTestHelpers.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

#include <Fields.hpp>
#include <FieldDB.hpp>
#include <MeshIO.hpp>
#include <ParallelComm.hpp>
#include <InitialConditions.hpp>

#include "MagnetoHydroDynamics.hpp"
#include "ExplicitFunctors.cpp"

#include <impl/Kokkos_Timer.hpp>


using namespace lgr;


/******************************************************************************/
TEUCHOS_UNIT_TEST( MHD, ElementMagnetics )
{
const char *yaml = R"Love(
%YAML 1.1
---
ANONYMOUS:
  Associations:
    Element Sets:
      eb_1: [[3,13]]
    Node Sets:
      ns_1: [[1,1],[1,3],[1,4],[1,5],[1,7],[1,9],[1,10],[1,11],[1,12]]
      ns_2: [[1,13],[1,14],[1,25]]
    Side Sets:
      ss_1: [[2,4],[2,10],[2,12],[2,13],[2,14],[2,16],[2,22]]
  Initial Conditions:
    MHD:
      Type: Function
      Variable: Magnetic Flux Vector Potential
      Value: 'vector(z, x, 0)'
      Element Block: eb_1
...
)Love";

/*
  B = curl{z, x, 0.0} = {0.0, +1.0, +1.0}
*/

  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromYamlString(yaml);

  Teuchos::ParameterList &assoc_pl =
    params->sublist("Associations");
  Teuchos::ParameterList &initialCond =
    params->sublist("Initial Conditions", false, "Initial conditions.");

  constexpr int spaceDim  = 3;
  constexpr int meshWidth = 2;
  constexpr Scalar aX_scaling = 1.0;
  constexpr Scalar aY_scaling = 2.0;
  constexpr Scalar aZ_scaling = 3.0;
  using Fields = Fields<spaceDim>;

  Teuchos::RCP<Omega_h::Mesh> meshOmegaH = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth, aX_scaling, aY_scaling, aZ_scaling);
  Omega_h::Assoc  assoc;
  Omega_h::update_assoc(&assoc, assoc_pl);
  Omega_h::MeshSets mesh_sets = Omega_h::invert(meshOmegaH.get(), assoc);

  FEMesh<spaceDim> femesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);

  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new Fields(femesh, paramList));

  InitialConditions<Fields> ic(initialCond);
  ic.set( mesh_sets[Omega_h::SIDE_SET],
          mesh_sets[Omega_h::ELEM_SET],
	  MagneticFaceFlux<Fields>(), 
	  femesh );

  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "mhdTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    Omega_h::Write<Scalar> errorPlusOneCell(elemLids.size());
    const int ElemNodeCount = Fields::ElemNodeCount;
    const auto elem_node_ids = femesh.elem_node_ids;
    const auto node_coords = femesh.node_coords;
    lgr::MHD<spaceDim> mhd(*fields);
    Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
	lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
	const size_t ielem = elemLids[e];
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }       
	Omega_h::Vector<3> Bexact; 
	Bexact(0) = 0.0; Bexact(1) = +1.0; Bexact(2) = +1.0;
	const Omega_h::Vector<3> B = mhd.elementMagneticFluxDensity(ielem,x,y,z);
	errorPlusOneCell[e] = Omega_h::norm(Bexact-B)/Omega_h::norm(Bexact) + 1.0; 
      }, "element B Check");
    Omega_h::HostWrite<Scalar> errorPlusOne(errorPlusOneCell);
    for (int i=0; i<elemLids.size(); ++i) 
      TEST_FLOATING_EQUALITY(errorPlusOne[i], 1.0, 1.0e-15);
  }

  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "mhdTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    Omega_h::Write<Scalar> errorPlusOneCell(elemLids.size());
    const int ElemNodeCount = Fields::ElemNodeCount;
    const auto elem_node_ids = femesh.elem_node_ids;
    const auto node_coords = femesh.node_coords;
    lgr::MHD<spaceDim> mhd(*fields);
    Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
	lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
	const size_t ielem = elemLids[e];
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }       
	Omega_h::Vector<3> Bexact; 
	Bexact(0) = 0.0; Bexact(1) = +1.0; Bexact(2) = +1.0;
	const Omega_h::Matrix<3,3> sigma = mhd.elementMagneticStressTensor(ielem,x,y,z);
	const Scalar pi = 4.0*atan(1.0);
	const Scalar mu = 4.0 * pi * 1.e-7;
	const Omega_h::Matrix<3,3> sigmaExact = 
	  (
	   Omega_h::outer_product(Bexact,Bexact)
	   - 0.5 * Omega_h::norm_squared(Bexact) * Omega_h::identity_matrix<3,3>() 
	  ) / mu;
	errorPlusOneCell[e] = Omega_h::norm(sigmaExact-sigma)/Omega_h::norm(sigmaExact) + 1.0; 
      }, "element sigma Check");
    Omega_h::HostWrite<Scalar> errorPlusOne(errorPlusOneCell);
    for (int i=0; i<elemLids.size(); ++i) 
      TEST_FLOATING_EQUALITY(errorPlusOne[i], 1.0, 1.0e-15);
  }

  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "mhdTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    Omega_h::Write<Scalar> errorPlusOneCell(elemLids.size());
    const int ElemNodeCount = Fields::ElemNodeCount;
    const auto elem_node_ids = femesh.elem_node_ids;
    const auto node_coords = femesh.node_coords;
    lgr::MHD<spaceDim> mhd(*fields);
    Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
	lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
	const size_t ielem = elemLids[e];
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }       
	Omega_h::Vector<3> Bexact; 
	Bexact(0) = 0.0; Bexact(1) = +1.0; Bexact(2) = +1.0;
	const Scalar Emag = mhd.elementMagneticEnergy(ielem,x,y,z);
	const Scalar pi = 4.0*atan(1.0);
	const Scalar mu = 4.0 * pi * 1.e-7;
	const Scalar EmagExact = 0.5 * Omega_h::norm_squared(Bexact) * tet4Volume(x,y,z) / mu;
	errorPlusOneCell[e] = fabs(Emag-EmagExact)/EmagExact + 1.0; 
      }, "element energy Check");
    Omega_h::HostWrite<Scalar> errorPlusOne(errorPlusOneCell);
    for (int i=0; i<elemLids.size(); ++i) 
      TEST_FLOATING_EQUALITY(errorPlusOne[i], 1.0, 1.0e-15);
  }

  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "mhdTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    Omega_h::Write<Scalar> errorPlusOneCell(elemLids.size());
    const int ElemNodeCount = Fields::ElemNodeCount;
    const auto elem_node_ids = femesh.elem_node_ids;
    const auto node_coords = femesh.node_coords;
    lgr::MHD<spaceDim> mhd(*fields);
    Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
	lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
	const size_t ielem = elemLids[e];
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }       
	Omega_h::Vector<3> Bexact; 
	Bexact(0) = 0.0; Bexact(1) = +1.0; Bexact(2) = +1.0;
	const Scalar K = mhd.elementAlfvenWaveModulus(ielem,x,y,z);
	const Scalar pi = 4.0*atan(1.0);
	const Scalar mu = 4.0 * pi * 1.e-7;
	const Scalar Kexact = Omega_h::norm_squared(Bexact) / mu;
	errorPlusOneCell[e] = fabs(K-Kexact)/Kexact + 1.0; 
      }, "element alfven Check");
    Omega_h::HostWrite<Scalar> errorPlusOne(errorPlusOneCell);
    for (int i=0; i<elemLids.size(); ++i) 
      TEST_FLOATING_EQUALITY(errorPlusOne[i], 1.0, 1.0e-15);
  }

}


