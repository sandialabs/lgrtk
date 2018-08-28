/*!
  These unit tests are for the Linear elastostatics functionality.
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

#include <impl/Kokkos_Timer.hpp>


using namespace lgr;

/******************************************************************************/
TEUCHOS_UNIT_TEST( InitialConditions, ConstantInitialConditions )
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
    initial density:
      Type: Constant
      Variable: Density
      Element Block: eb_1
      Value: 1.0
    initial energy:
      Type: Constant
      Variable: Specific Internal Energy 
      Element Block: eb_1
      Value: 1.0e-12
    X Velocity block translation:
      Type: Constant 
      Variable: Velocity
      Value: [-1.0, 0.0, 0.0]
      Nodeset: ns_1
    X Velocity left wall:
      Type: Constant
      Variable: Velocity
      Value: [0.0,0.0,0.0]
      Nodeset: ns_2
...
)Love";

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
  typedef typename Fields::state_array_type state_array_type;
  typedef typename Fields::array_type             array_type;
  typedef typename Fields::geom_state_array_type  geom_state_array_type;
  typedef typename Fields::geom_array_type        geom_array_type;


  Teuchos::RCP<Omega_h::Mesh> meshOmegaH = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth, aX_scaling, aY_scaling, aZ_scaling);
  Omega_h::Assoc  assoc;
  Omega_h::update_assoc(&assoc, assoc_pl);
  Omega_h::MeshSets mesh_sets = Omega_h::invert(meshOmegaH.get(), assoc);

  FEMesh<spaceDim> femesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);

  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new Fields(femesh, paramList));

  InitialConditions<Fields> ic(initialCond);
  ic.set(
      mesh_sets[Omega_h::NODE_SET], Velocity<Fields>(),
      Displacement<Fields>(), femesh);
  ic.set(
      mesh_sets[Omega_h::ELEM_SET], MassDensity<Fields>(),
      InternalEnergyPerUnitMass<Fields>(), femesh);
  ic.set(
      mesh_sets[Omega_h::SIDE_SET],
      mesh_sets[Omega_h::ELEM_SET],
      MagneticFaceFlux<Fields>(), femesh);


  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "InitialConditionTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    const state_array_type density_state = MassDensity<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      array_type density(Fields::getFromSA(density_state, i));
      Omega_h::Write<Scalar> F(elemLids.size());
      Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
        F[e] = density(elemLids[e]);
      }, "Density Check");
      constexpr Scalar x = 1.0;
      Omega_h::HostWrite<Scalar> H(F);
      for (int ii=0; ii<elemLids.size(); ++ii) 
        TEST_FLOATING_EQUALITY(H[ii], x, 1.0e-15);
    }
  }
  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "InitialConditionTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    const state_array_type energy_state = InternalEnergyPerUnitMass<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      array_type energy(Fields::getFromSA(energy_state, i));
      Omega_h::Write<Scalar> F(elemLids.size());
      Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
        F[e] = energy(elemLids[e]);
      }, "Energy Check");
      constexpr Scalar x = 1.0e-12;
      Omega_h::HostWrite<Scalar> H(F);
      for (int ii=0; ii<elemLids.size(); ++ii) 
        TEST_FLOATING_EQUALITY(H[ii], x, 1.0e-15);
    }
  }
  {
    const auto iter = mesh_sets[Omega_h::NODE_SET].find("ns_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::NODE_SET].end(), std::invalid_argument,
      "InitialConditionTests node set ns_1 doesn't exist!\n");
    auto nodeLids = iter->second;
    const geom_state_array_type velocity_state = Velocity<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      geom_array_type velocity(Fields::getGeomFromSA(velocity_state, i));
      Omega_h::Write<Scalar> F(3*nodeLids.size());
      Kokkos::parallel_for(nodeLids.size(), LAMBDA_EXPRESSION(int n) {
        for (int ii=0; ii<3; ++ii) 
          F[3*n+ii] = velocity(nodeLids[n],ii);
      }, "Velocity Check");
      constexpr Scalar x[3] = {-1.0, 0.0, 0.0};
      Omega_h::HostWrite<Scalar> H(F);
      for (int n=0; n<nodeLids.size(); ++n) 
        for (int ii=0; ii<3; ++ii) 
          TEST_FLOATING_EQUALITY(H[3*n+ii], x[ii], 1.0e-15);
    }
  }
  {
    const auto iter = mesh_sets[Omega_h::NODE_SET].find("ns_2");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::NODE_SET].end(), std::invalid_argument,
      "InitialConditionTests node set ns_2 doesn't exist!\n");
    auto nodeLids = iter->second;
    const geom_state_array_type velocity_state = Velocity<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      geom_array_type velocity(Fields::getGeomFromSA(velocity_state, i));
      Omega_h::Write<Scalar> F(3*nodeLids.size());
      Kokkos::parallel_for(nodeLids.size(), LAMBDA_EXPRESSION(int n) {
        for (int ii=0; ii<3; ++ii) 
          F[3*n+ii] = velocity(nodeLids[n],ii);
      }, "Velocity Check");
      constexpr Scalar x[3] = {0.0, 0.0, 0.0};
      Omega_h::HostWrite<Scalar> H(F);
      for (int n=0; n<nodeLids.size(); ++n) 
        for (int ii=0; ii<3; ++ii) 
          TEST_FLOATING_EQUALITY(H[3*n+ii], x[ii], 1.0e-15);
    }
  }
}


/******************************************************************************/
TEUCHOS_UNIT_TEST( InitialConditions, FunctionalInitialConditions )
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
    initial density:
      Type: Function
      Variable: Density
      Element Block: eb_1
      Value: '1.0'
    initial energy:
      Type: Function
      Variable: Specific Internal Energy 
      Element Block: eb_1
      Value: '1.0e-12'
    X Velocity block translation:
      Type: Function
      Variable: Velocity
      Value: 'vector(-1.0, 0.0, 0.0)'
      Nodeset: ns_1
    X Velocity left wall:
      Type: Function
      Variable: Velocity
      Value: 'vector(0.0, 0.0, 0.0)'
      Nodeset: ns_2
    MHD 1:
      Type: Function
      Variable: Magnetic Flux Vector Potential
      Value: 'vector(0, x, 0)'
      Faceset: ss_1
    MHD 2:
      Type: Function
      Variable: Magnetic Flux Vector Potential
      Value: 'vector(0, x, 0)'
      Element Block: eb_1
...
)Love";

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
  typedef typename Fields::state_array_type state_array_type;
  typedef typename Fields::array_type             array_type;
  typedef typename Fields::geom_state_array_type  geom_state_array_type;
  typedef typename Fields::geom_array_type        geom_array_type;


  Teuchos::RCP<Omega_h::Mesh> meshOmegaH = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth, aX_scaling, aY_scaling, aZ_scaling);
  Omega_h::Assoc  assoc;
  Omega_h::update_assoc(&assoc, assoc_pl);
  Omega_h::MeshSets mesh_sets = Omega_h::invert(meshOmegaH.get(), assoc);

  FEMesh<spaceDim> femesh = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH);

  Teuchos::ParameterList paramList;
  auto fields = Teuchos::rcp(new Fields(femesh, paramList));

  InitialConditions<Fields> ic(initialCond);
  ic.set(
      mesh_sets[Omega_h::NODE_SET], Velocity<Fields>(),
      Displacement<Fields>(), femesh);
  ic.set(
      mesh_sets[Omega_h::ELEM_SET], MassDensity<Fields>(),
      InternalEnergyPerUnitMass<Fields>(), femesh);
  ic.set(
      mesh_sets[Omega_h::SIDE_SET], 
      mesh_sets[Omega_h::ELEM_SET],
      MagneticFaceFlux<Fields>(), femesh);


  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "InitialConditionTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    const state_array_type density_state = MassDensity<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      array_type density(Fields::getFromSA(density_state, i));
      Omega_h::Write<Scalar> F(elemLids.size());
      Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
        F[e] = density(elemLids[e]);
      }, "Density Check");
      constexpr Scalar x = 1.0;
      Omega_h::HostWrite<Scalar> H(F);
      for (int ii=0; ii<elemLids.size(); ++ii) 
        TEST_FLOATING_EQUALITY(H[ii], x, 1.0e-15);
    }
  }
  {
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "InitialConditionTests element set eb_1 doesn't exist!\n");
    auto elemLids = iter->second;
    const state_array_type energy_state = InternalEnergyPerUnitMass<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      array_type energy(Fields::getFromSA(energy_state, i));
      Omega_h::Write<Scalar> F(elemLids.size());
      Kokkos::parallel_for(elemLids.size(), LAMBDA_EXPRESSION(int e) {
        F[e] = energy(elemLids[e]);
      }, "Energy Check");
      constexpr Scalar x = 1.0e-12;
      Omega_h::HostWrite<Scalar> H(F);
      for (int ii=0; ii<elemLids.size(); ++ii) 
        TEST_FLOATING_EQUALITY(H[ii], x, 1.0e-15);
    }
  }
  {
    const auto iter = mesh_sets[Omega_h::NODE_SET].find("ns_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::NODE_SET].end(), std::invalid_argument,
      "InitialConditionTests node set ns_1 doesn't exist!\n");
    auto nodeLids = iter->second;
    const geom_state_array_type velocity_state = Velocity<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      geom_array_type velocity(Fields::getGeomFromSA(velocity_state, i));
      Omega_h::Write<Scalar> F(3*nodeLids.size());
      Kokkos::parallel_for(nodeLids.size(), LAMBDA_EXPRESSION(int n) {
        for (int ii=0; ii<3; ++ii) 
          F[3*n+ii] = velocity(nodeLids[n],ii);
      }, "Velocity Check");
      constexpr Scalar x[3] = {-1.0, 0.0, 0.0};
      Omega_h::HostWrite<Scalar> H(F);
      for (int n=0; n<nodeLids.size(); ++n) 
        for (int ii=0; ii<3; ++ii) 
          TEST_FLOATING_EQUALITY(H[3*n+ii], x[ii], 1.0e-15);
    }
  }
  {
    const auto iter = mesh_sets[Omega_h::NODE_SET].find("ns_2");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::NODE_SET].end(), std::invalid_argument,
      "InitialConditionTests node set ns_2 doesn't exist!\n");
    auto nodeLids = iter->second;
    const geom_state_array_type velocity_state = Velocity<Fields>();
    for (int i = 0; i < Fields::NumStates; ++i) { 
      geom_array_type velocity(Fields::getGeomFromSA(velocity_state, i));
      Omega_h::Write<Scalar> F(3*nodeLids.size());
      Kokkos::parallel_for(nodeLids.size(), LAMBDA_EXPRESSION(int n) {
        for (int ii=0; ii<3; ++ii) 
          F[3*n+ii] = velocity(nodeLids[n],ii);
      }, "Velocity Check");
      constexpr Scalar x[3] = {0.0, 0.0, 0.0};
      Omega_h::HostWrite<Scalar> H(F);
      for (int n=0; n<nodeLids.size(); ++n) 
        for (int ii=0; ii<3; ++ii) 
          TEST_FLOATING_EQUALITY(H[3*n+ii], x[ii], 1.0e-15);
    }
  }
  {
    constexpr int N = Fields::FaceNodeCount;
    constexpr int D = Fields::SpaceDim;
    const Fields::node_coords_type coords = femesh.node_coords;
    const Fields::face_node_ids_type face_nodes = femesh.face_node_ids;
    const auto iter = mesh_sets[Omega_h::SIDE_SET].find("ss_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::NODE_SET].end(), std::invalid_argument,
      "InitialConditionTests node set ss_1 doesn't exist!\n");
    const auto faceLids   = iter->second;
    const auto nset_faces = faceLids.size();
    Omega_h::Write<double> Fx(nset_faces);
    Omega_h::Write<double> Fy(nset_faces);
    Omega_h::Write<double> Fz(nset_faces);

    auto side_vec = LAMBDA_EXPRESSION(int set_face) {
      const auto face = faceLids[set_face];
      int nodes[N];    
      for (int i = 0; i < N; ++i) 
        nodes[i] = face_nodes(face, i); 
      double X[D][N];
      for (int i = 0; i < N; ++i) 
        for (int j = 0; j < D; ++j) 
           X[i][j] = coords(nodes[i], j); 
      double A[3], B[3];
      for (int j = 0; j < D; ++j) {
        A[j] = X[1][j]-X[0][j];
        B[j] = X[2][j]-X[0][j];
      }
      const double x[3] = {A[1]*B[2] - A[2]*B[1],
                           A[2]*B[0] - A[0]*B[2],
                           A[0]*B[1] - A[1]*B[0]};
      Fx[set_face] = x[0]/2;
      Fy[set_face] = x[1]/2;
      Fz[set_face] = x[2]/2;
    };  
    Kokkos::parallel_for(nset_faces, side_vec);
    const array_type face_flux = MagneticFaceFlux<Fields>();
    Omega_h::Write<Scalar> F(faceLids.size());
    Kokkos::parallel_for(faceLids.size(), LAMBDA_EXPRESSION(int n) {
        F[n] = face_flux(faceLids[n]);
    }, "Face Flux Check");
    Omega_h::HostWrite<Scalar> H(F);
    Omega_h::HostWrite<Scalar> x(Fz);
    for (int n=0; n<faceLids.size(); ++n) 
      TEST_FLOATING_EQUALITY(H[n], x[n], 1.0e-15);
  }
  {
    constexpr int N = Fields::FaceNodeCount;
    constexpr int F = Fields::ElemFaceCount;
    constexpr int D = Fields::SpaceDim;
    const Fields::node_coords_type coords = femesh.node_coords;
    const Fields::face_node_ids_type face_nodes = femesh.face_node_ids;
    const Fields::elem_face_ids_type elem_faces = femesh.elem_face_ids;
    const auto iter = mesh_sets[Omega_h::ELEM_SET].find("eb_1");
    TEUCHOS_TEST_FOR_EXCEPTION(
      iter == mesh_sets[Omega_h::ELEM_SET].end(), std::invalid_argument,
      "InitialConditionTests node set eb_1 doesn't exist!\n");
    const auto elemLids   = iter->second;
    const auto nset_elems = elemLids.size();
    const auto nset_faces = F*nset_elems;
    Omega_h::Write<double> Fx(nset_faces);
    Omega_h::Write<double> Fy(nset_faces);
    Omega_h::Write<double> Fz(nset_faces);

    auto side_vec = LAMBDA_EXPRESSION(int set_face) {
      const int   el = elemLids[set_face/F];
      const int    f = set_face%F;
      const int face = elem_faces(el,f);  
      int nodes[N];    
      for (int i = 0; i < N; ++i) 
        nodes[i] = face_nodes(face, i); 
      double X[D][N];
      for (int i = 0; i < N; ++i) 
        for (int j = 0; j < D; ++j) 
           X[i][j] = coords(nodes[i], j); 
      double A[3], B[3];
      for (int j = 0; j < D; ++j) {
        A[j] = X[1][j]-X[0][j];
        B[j] = X[2][j]-X[0][j];
      }
      const double x[3] = {A[1]*B[2] - A[2]*B[1],
                           A[2]*B[0] - A[0]*B[2],
                           A[0]*B[1] - A[1]*B[0]};
      Fx[set_face] = x[0]/2;
      Fy[set_face] = x[1]/2;
      Fz[set_face] = x[2]/2;
    };  
    Kokkos::parallel_for(nset_faces, side_vec);
    const array_type face_flux = MagneticFaceFlux<Fields>();
    Omega_h::Write<Scalar> FF(nset_faces);
    Kokkos::parallel_for(nset_faces, LAMBDA_EXPRESSION(int n) {
        const int   el = elemLids[n/F];
        const int    f = n%F;
        const int face = elem_faces(el,f);  
        FF[n] = face_flux(face);
    }, "Face Flux Check");
    Omega_h::HostWrite<Scalar> H(FF);
    Omega_h::HostWrite<Scalar> x(Fz);
    for (int n=0; n<nset_faces; ++n) 
      TEST_FLOATING_EQUALITY(H[n], x[n], 1.0e-15);
  }
}


