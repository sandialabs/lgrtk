/*!
  These unit tests are for mhd.
*/

#include "LGRTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"
#include "Omega_h_qr.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include <Teuchos_YamlParameterListCoreHelpers.hpp>

#include "CrsMatrix.hpp"
#include "MeshFixture.hpp"
#include "StaticsTypes.hpp"

#include "PlatoTestHelpers.hpp"

#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>

#ifdef KOKKOS_HAVE_CUDA
#define R3D_USE_CUDA
#endif

#include <r3d.hpp>
#include <Fields.hpp>
#include <FieldDB.hpp>
#include <MeshIO.hpp>
#include <ParallelComm.hpp>
#include <InitialConditions.hpp>

#include "MagnetoHydroDynamics.hpp"
#include "ExplicitFunctors.cpp"
#include <ErrorHandling.hpp>


#include <impl/Kokkos_Timer.hpp>


using namespace lgr;

namespace Kokkos {
template<>
struct reduction_identity<Omega_h::Vector<3>> {
  static KOKKOS_FUNCTION Omega_h::Vector<3> sum() {Omega_h::Vector<3> Z; Z(0)=0; Z(1)=0; Z(2)=0; return Z;}
};
}


bool run_the_test(const std::array<Scalar,3> &perturbation);
/******************************************************************************/
TEUCHOS_UNIT_TEST( MHD, MatrixAssembly )
{
  if (1) {
    std::srand(0);
    const std::array<Scalar,3> perterb = {0,0,0};
    bool pass = run_the_test(perterb);
    TEST_ASSERT(pass);
  }

  if (1) {
    std::array<Scalar,3> perterb;
    perterb[0] = double(std::rand()%100)/1000.;
    perterb[1] = double(std::rand()%100)/1000.;
    perterb[2] = double(std::rand()%100)/1000.;
    printf ("Perterb mesh by:(%f, %f, %f)\n",perterb[0],perterb[1],perterb[2]);
    bool pass = run_the_test(perterb);
    TEST_ASSERT(pass);
  }
}

bool run_the_test(const std::array<Scalar,3> &perturbation) {

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
      Value: 'vector(2*z, 5*x, y)'
      Element Block: eb_1
...
)Love";

/*
  B = curl{z, x, 0.0} = {+1.0, +2.0, +5.0}
*/

  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromYamlString(yaml);

  Teuchos::ParameterList &assoc_pl =
    params->sublist("Associations");
  Teuchos::ParameterList &initialCond =
    params->sublist("Initial Conditions", false, "Initial conditions.");

  //using DeviceSpace = Kokkos::DefaultExecutionSpace;
  //using DynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
  //using DeviceTeamHandleType = Kokkos::TeamPolicy<DeviceSpace, DynamicScheduleType>::member_type;

  constexpr int spaceDim  = 3;
  using Fields = Fields<spaceDim>;
  constexpr int ElemNodeCount = Fields::ElemNodeCount;
  constexpr int ElemFaceCount = Fields::ElemFaceCount;

  constexpr int vert      = spaceDim+1;
  constexpr int moment    = 2;
  constexpr int meshWidth = 2;
  constexpr int maxElem   = 50;
  constexpr int maxFaceCount = maxElem*ElemFaceCount;

  typedef r3d::Few<r3d::Vector<spaceDim>,vert> Tet4;
  typedef r3d::Polytope<spaceDim>              Polytope;
  typedef r3d::Polynomial<spaceDim,moment>     Polynomial;
  typedef typename Fields::size_type size_type;

  constexpr Scalar aX_scaling = 1.0;
  constexpr Scalar aY_scaling = 1.0;
  constexpr Scalar aZ_scaling = 1.0;

  Teuchos::RCP<Omega_h::Mesh> meshOmegaH_src = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth, aX_scaling, aY_scaling, aZ_scaling);

  Teuchos::RCP<Omega_h::Mesh> meshOmegaH_trg = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth, aX_scaling, aY_scaling, aZ_scaling);

  Omega_h::Assoc  assoc;
  Omega_h::update_assoc(&assoc, assoc_pl);
  Omega_h::MeshSets mesh_sets_src = Omega_h::invert(meshOmegaH_src.get(), assoc);
  Omega_h::MeshSets mesh_sets_trg = Omega_h::invert(meshOmegaH_trg.get(), assoc);

  FEMesh<spaceDim> femesh_src = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH_src);
  FEMesh<spaceDim> femesh_trg = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH_trg);

  Teuchos::ParameterList paramList;
  auto fields_src = Teuchos::rcp(new Fields(femesh_src, paramList));
  auto fields_trg = Teuchos::rcp(new Fields(femesh_trg, paramList));

  Kokkos::realloc(FieldDB<Fields::array_type>::Self()["magnetic face flux source"], femesh_src.nfaces);
  Kokkos::realloc(FieldDB<Fields::array_type>::Self()["magnetic face flux target"], femesh_trg.nfaces);

  auto magneticFaceFlux_src = safeFieldLookup<Fields::array_type>("magnetic face flux source");
  auto magneticFaceFlux_trg = safeFieldLookup<Fields::array_type>("magnetic face flux target");

  InitialConditions<Fields> ic(initialCond);
  ic.set( mesh_sets_src[Omega_h::SIDE_SET],
          mesh_sets_src[Omega_h::ELEM_SET],
          magneticFaceFlux_src, 
          femesh_src );

  const auto iter_src = mesh_sets_src[Omega_h::ELEM_SET].find("eb_1");
  const auto iter_trg = mesh_sets_trg[Omega_h::ELEM_SET].find("eb_1");
  TEUCHOS_TEST_FOR_EXCEPTION(
    iter_src == mesh_sets_src[Omega_h::ELEM_SET].end(), std::invalid_argument,
    "mhdTests element set eb_1 doesn't exist!\n");
  TEUCHOS_TEST_FOR_EXCEPTION(
    iter_trg == mesh_sets_trg[Omega_h::ELEM_SET].end(), std::invalid_argument,
    "mhdTests element set eb_1 doesn't exist!\n");
  auto elemLids_src = iter_src->second;
  auto elemLids_trg = iter_trg->second;

  LGR_THROW_IF(maxElem<elemLids_src.size(), "Number of elements exceeds max.");
  LGR_THROW_IF(maxElem<elemLids_trg.size(), "Number of elements exceeds max.");

  const auto elem_node_ids_src = femesh_src.elem_node_ids;
  const auto node_coords_src = femesh_src.node_coords;
  const auto elemFaceIDs_src = femesh_src.elem_face_ids;
  const auto elemFaceOrientations_src = femesh_src.elem_face_orientations;

  const auto elem_node_ids_trg = femesh_trg.elem_node_ids;
  const auto node_coords_trg = femesh_trg.node_coords;
  const auto elemFaceIDs_trg = femesh_trg.elem_face_ids;
  const auto elemFaceOrientations_trg = femesh_trg.elem_face_orientations;

  Kokkos::parallel_for("Tweek Coordinates", Kokkos::RangePolicy<Kokkos::Serial>(0, 1), [&] (const int) {
      for (unsigned n = 0; n < node_coords_src.extent(0); ++n) {
         if (std::abs(node_coords_src(n, 0)-.5)<.00001 &&
             std::abs(node_coords_src(n, 1)-.5)<.00001 &&
             std::abs(node_coords_src(n, 2)-.5)<.00001) {
                node_coords_src(n, 0) += perturbation[0];
                node_coords_src(n, 1) += perturbation[1];
                node_coords_src(n, 2) += perturbation[2];
         }
      }
  }); //end Kokkos::parallel_for outer loop over target elements 

  // Can only do a single matrix for a thread, so no looping over elements.
  // Hopefully Omega_h will eventually be looping over cavities and there will
  // be lots of cavities.
  const size_type inull=std::numeric_limits<size_type>::max();
  const std::string debuggingName("mhdMatrixAssemblyTests");
  Kokkos::parallel_for(debuggingName, Kokkos::RangePolicy<Kokkos::Serial>(0, 1), [&] (const int) {

    size_type elemFaceLocalIDs_trg [maxElem][ElemFaceCount];
    size_type elemFaceIsSurface_trg[maxElem][ElemFaceCount];
    for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
      for (int face = 0; face < ElemFaceCount; ++face) {
        elemFaceLocalIDs_trg[elem][face] = inull;
        elemFaceIsSurface_trg[elem][face] = 0;
      }
    }
    int number_of_target_faces = 0;
    for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
      for (int face = 0; face < ElemFaceCount; ++face) {
        if (elemFaceLocalIDs_trg[elem][face] == inull) {
          int n = 0;
          for (int ielem = elem; ielem < elemLids_trg.size() && n < 2; ++ielem) {
            for (int iface = 0; iface < ElemFaceCount && n < 2; ++iface) {
              if (elemFaceIDs_trg(elem,face) == elemFaceIDs_trg(ielem,iface)) {
                if (elemFaceLocalIDs_trg[ielem][iface] == inull) {
                  elemFaceLocalIDs_trg[ielem][iface] = number_of_target_faces; 
                  ++n;
                }
              }
            }
          }
          ++number_of_target_faces;
          if (n==1) elemFaceIsSurface_trg[elem][face] = 1;
        }
      }
    }

    Omega_h::Matrix<maxFaceCount,maxFaceCount> M;
    Omega_h::Vector<maxFaceCount>              f;
    Omega_h::Matrix<maxElem,maxFaceCount>      Q;
    Omega_h::Vector<maxElem>                   q;
    
    for (int iface = 0; iface < maxFaceCount; ++iface) {
      f(iface) = 0;
      for (int jface = 0; jface < maxFaceCount; ++jface) {
        M(iface,jface) = 0;
      }
    }
    for (int elem = 0; elem < maxElem; ++elem) {
      q(elem) = 0;
      for (int face = 0; face < maxFaceCount; ++face) {
        Q(elem,face) = 0;
      }
    }

    for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
      q(elem) = 0;
      for (int face = 0; face < ElemFaceCount; ++face) {
        const int sign = elemFaceOrientations_trg(elem,face);
        const size_t iface = elemFaceLocalIDs_trg[elem][face];
        Q(elem,iface) = sign;
      }
    }

    for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
      Scalar moments[Polynomial::nterms]; 
      for (int i=0; i<Polynomial::nterms; ++i) moments[i]=0;
      Omega_h::Few<Omega_h::Vector<spaceDim>,vert> nodalCoordinates;

      {
        Tet4 nodalCoordinates_trg;
        for (int i = 0; i < ElemNodeCount; ++i) {
           const int n = elem_node_ids_trg(elem, i);
           r3d::Vector<spaceDim> &coord = nodalCoordinates_trg[i];
           coord[0] = node_coords_trg(n, 0);
           coord[1] = node_coords_trg(n, 1);
           coord[2] = node_coords_trg(n, 2);
        } 
        Polytope poly_trg;
        r3d::init(poly_trg, nodalCoordinates_trg);
        r3d::reduce<moment>(poly_trg, moments);

        for (int j=0; j<vert; ++j)
           for (int i=0; i<spaceDim; ++i)
              nodalCoordinates[j][i] = nodalCoordinates_trg[j][i];
      }

      for (int iface = 0; iface < ElemFaceCount; ++iface) {

        Omega_h::Few<Scalar,ElemFaceCount> ifacesomething;
        for (int i=0; i<ElemFaceCount; ++i) ifacesomething[i]=0;
        ifacesomething[iface]= elemFaceOrientations_trg(elem,iface);
        Omega_h::Vector<3> iconstantTerm;
        Scalar ilinearFactor;
        elementPhysicalFacePolynomial( /*input*/
                                       nodalCoordinates,
                                       ifacesomething,
                                       /*output*/
                                       iconstantTerm,
                                       ilinearFactor );
 
        for (int jface = 0; jface < ElemFaceCount; ++jface) {
          Omega_h::Few<Scalar,ElemFaceCount> jfacesomething;
          for (int i=0; i<ElemFaceCount; ++i) jfacesomething[i]=0;
          jfacesomething[jface]= elemFaceOrientations_trg(elem,jface);
          Omega_h::Vector<3> jconstantTerm;
          Scalar jlinearFactor;
          elementPhysicalFacePolynomial( /*input*/
                                       nodalCoordinates,
                                       jfacesomething,
                                       /*output*/
                                       jconstantTerm,
                                       jlinearFactor );
         
          Scalar src_poly[Polynomial::nterms];
          src_poly[0] = iconstantTerm*jconstantTerm; //inner_product(iconstantTerm,jconstantTerm);
          src_poly[1] = iconstantTerm[0]*jlinearFactor + jconstantTerm[0]*ilinearFactor;
          src_poly[2] = iconstantTerm[1]*jlinearFactor + jconstantTerm[1]*ilinearFactor;
          src_poly[3] = iconstantTerm[2]*jlinearFactor + jconstantTerm[2]*ilinearFactor;
          src_poly[4] = ilinearFactor*jlinearFactor;
          src_poly[5] = 0;
          src_poly[6] = 0;
          src_poly[7] = ilinearFactor*jlinearFactor;
          src_poly[8] = 0;
          src_poly[9] = ilinearFactor*jlinearFactor;

          Scalar integral = 0;
          for (int i=0; i<Polynomial::nterms; ++i) {
            integral += moments[i]*src_poly[i];
          }    
          const size_t i = elemFaceLocalIDs_trg[elem][iface];
          const size_t j = elemFaceLocalIDs_trg[elem][jface];
          M(i,j) += integral;
        }
      }
    }

    for (int elem_trg = 0; elem_trg < elemLids_trg.size(); ++elem_trg) {
      Tet4 nodalCoordinates_trg;
      for (int i = 0; i < ElemNodeCount; ++i) {
         const int n = elem_node_ids_trg(elem_trg, i);
         r3d::Vector<spaceDim> &coord = nodalCoordinates_trg[i];
         coord[0] = node_coords_trg(n, 0);
         coord[1] = node_coords_trg(n, 1);
         coord[2] = node_coords_trg(n, 2);
      } 

      for (int elem_src = 0; elem_src < elemLids_src.size(); ++elem_src) {
        Scalar moments[Polynomial::nterms] = {0.}; 
        for (int i=0; i<Polynomial::nterms; ++i) moments[i] = 0; 

        Tet4 nodalCoordinates_src;
        for (int i = 0; i < ElemNodeCount; ++i) {
           const int n = elem_node_ids_src(elem_src, i);
           r3d::Vector<spaceDim> &coord = nodalCoordinates_src[i];
           coord[0] = node_coords_src(n, 0);
           coord[1] = node_coords_src(n, 1);
           coord[2] = node_coords_src(n, 2);
        } 

        Polytope intersection;
        r3d::intersect_simplices(intersection, nodalCoordinates_src, nodalCoordinates_trg);
        r3d::reduce<moment>(intersection, moments);

        Omega_h::Vector<3> jconstantTerm;
        Scalar jlinearFactor;
        {
          Omega_h::Few<Omega_h::Vector<spaceDim>,vert> nodalCoordinates;
          for (int j=0; j<vert; ++j)
             for (int i=0; i<spaceDim; ++i)
                nodalCoordinates[j][i] = nodalCoordinates_src[j][i];

          Omega_h::Vector<ElemFaceCount> faceFluxSource;
          for (int face = 0; face < ElemFaceCount; ++face) {
            const int sign = elemFaceOrientations_src(elem_src,face);
            const size_t faceGID = elemFaceIDs_src(elem_src,face);
            faceFluxSource[face] = sign * magneticFaceFlux_src(faceGID);  
          }       
          elementPhysicalFacePolynomial( /*input*/
                                         nodalCoordinates,
                                         faceFluxSource,
                                         /*output*/
                                         jconstantTerm,
                                         jlinearFactor );
        }

        for (int iface = 0; iface < ElemFaceCount; ++iface) {

          Omega_h::Vector<3> iconstantTerm;
          Scalar ilinearFactor;
          {
            Omega_h::Few<Omega_h::Vector<spaceDim>,vert> nodalCoordinates;
            for (int j=0; j<vert; ++j)
               for (int i=0; i<spaceDim; ++i)
                  nodalCoordinates[j][i] = nodalCoordinates_trg[j][i];
            Omega_h::Few<Scalar,ElemFaceCount> ifacesomething;
            for (int i=0; i<ElemFaceCount; ++i) ifacesomething[i]=0;
            ifacesomething[iface] = elemFaceOrientations_trg(elem_trg,iface);
            elementPhysicalFacePolynomial( /*input*/
                                           nodalCoordinates,
                                           ifacesomething,
                                           /*output*/
                                           iconstantTerm,
                                           ilinearFactor );
          }

          Scalar src_poly[Polynomial::nterms];
          for (int i=0; i<Polynomial::nterms; ++i) src_poly[i]=0;
          src_poly[0] = iconstantTerm*jconstantTerm; //inner_product(iconstantTerm,jconstantTerm);
          src_poly[1] = iconstantTerm[0]*jlinearFactor + jconstantTerm[0]*ilinearFactor;
          src_poly[2] = iconstantTerm[1]*jlinearFactor + jconstantTerm[1]*ilinearFactor;
          src_poly[3] = iconstantTerm[2]*jlinearFactor + jconstantTerm[2]*ilinearFactor;
          src_poly[4] = ilinearFactor*jlinearFactor;
          src_poly[5] = 0;
          src_poly[6] = 0;
          src_poly[7] = ilinearFactor*jlinearFactor;
          src_poly[8] = 0;
          src_poly[9] = ilinearFactor*jlinearFactor;

          Scalar integral = 0;
          for (int i=0; i<Polynomial::nterms; ++i) {
            integral += moments[i]*src_poly[i];
          }    

          const size_t i = elemFaceLocalIDs_trg[elem_trg][iface];
          f(i) += integral;
        }
      }
    }

    const int nface = number_of_target_faces;
    const int nelem = elemLids_trg.size();
    int nsize = nface+nelem;

    constexpr int maxSize = maxFaceCount+maxElem;
    Omega_h::Matrix<maxSize,maxSize> A;
    Omega_h::Vector<maxSize>         b;
    for (int i=0; i<maxSize; ++i) {
      b(i) = 0;
      for (int j=0; j<maxSize; ++j) {
        A(i,j) = 0;
      }
    }
    for (int i=0; i<nface; ++i) {
      for (int j=0; j<nface; ++j) {
        A(i,j) = M(i,j);
      }
    }
    for (int i=0; i<nelem; ++i) {
      for (int j=0; j<nface; ++j) {
        A(nface+i,j) = Q(i,j);
        A(j,nface+i) = Q(i,j);
      }
    }

    for (int i=0; i<nface; ++i) b(i) = f(i);
    for (int i=0; i<nelem; ++i) b(i+nface) = q(i);

    for (int elem_trg = 0; elem_trg < elemLids_trg.size(); ++elem_trg) {
      for (int face_trg = 0; face_trg < ElemFaceCount; ++face_trg) {
        if (elemFaceIsSurface_trg[elem_trg][face_trg]) {  
          const size_t faceGID_trg = elemFaceIDs_trg(elem_trg,face_trg);
          Scalar faceFluxSource = 0;
          int sign = 0;
          bool found = false;
          for (int elem_src = 0; elem_src < elemLids_src.size() && !found; ++elem_src) {
            for (int face_src = 0; face_src < ElemFaceCount && !found; ++face_src) {
              const size_t faceGID_src = elemFaceIDs_src(elem_src,face_src);
              if (faceGID_trg == faceGID_src) {
                sign = elemFaceOrientations_trg(elem_trg,face_trg);
                faceFluxSource =  magneticFaceFlux_src(faceGID_src);  
                found = true;
              }
            }
          }
          LGR_THROW_IF(!found, "External face not found.");
          const int i = elemFaceLocalIDs_trg[elem_trg][face_trg];
          A(nsize,i) = sign;
          A(i,nsize) = sign;
          b(nsize)   = faceFluxSource;
          ++nsize;
        }
      }
    }
    LGR_THROW_IF(maxSize<nsize, "Maximum cavity size exceeded. Increase max number of elements.");

    Omega_h::Vector<maxSize> x = Omega_h::solve_using_qr(nsize, nsize, A, b);

    for (int elem_trg = 0; elem_trg < elemLids_trg.size(); ++elem_trg) {
      for (int face_trg = 0; face_trg < ElemFaceCount; ++face_trg) {
        const size_t faceID_trg = elemFaceLocalIDs_trg[elem_trg][face_trg];
        const size_t faceGID = elemFaceIDs_trg(elem_trg,face_trg);
        magneticFaceFlux_trg(faceGID) = x(faceID_trg);
      }
    }

  }); //end Kokkos::parallel_for outer loop over target elements 

  bool success = true;
  Scalar norm = 0; for (int i=0; i<3; ++i) norm += perturbation[i]*perturbation[i];
  const double tol = norm ? .1 : 1.0e-12;
  auto flux_trg = Kokkos::create_mirror_view(magneticFaceFlux_trg);
  auto flux_src = Kokkos::create_mirror_view(magneticFaceFlux_src);
  Kokkos::deep_copy(flux_trg, magneticFaceFlux_trg);
  Kokkos::deep_copy(flux_src, magneticFaceFlux_src);
  for (unsigned i=0; i<magneticFaceFlux_trg.size(); ++i) {
    const bool check = std::abs(flux_trg(i)-flux_src(i))<tol;
    if (!check) std::cout<<" Failed check. Target flux:"<<flux_trg(i)
      <<" Source Flux:"<<flux_src(i)
      <<" Diff:"<<std::abs(flux_trg(i)-flux_src(i))
      <<" Tol:"<<tol<<std::endl;
    success = success && check;
  }
  return success;
}


