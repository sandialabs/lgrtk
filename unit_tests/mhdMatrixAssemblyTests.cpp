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


/******************************************************************************/
TEUCHOS_UNIT_TEST( MHD, MatrixAssembly )
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
  constexpr int vert      = spaceDim+1;
  constexpr int moment    = 2;
  constexpr int meshWidth = 2;
  constexpr int maxElem   = 100;

  using Fields = Fields<spaceDim>;
  constexpr int ElemNodeCount = Fields::ElemNodeCount;
  constexpr int ElemFaceCount = Fields::ElemFaceCount;

  typedef r3d::Few<r3d::Vector<spaceDim>,vert> Tet4;
  typedef r3d::Polytope<spaceDim>              Polytope;
  typedef r3d::Polynomial<spaceDim,moment>     Polynomial;
  typedef typename Fields::size_type size_type;

  constexpr Scalar aX_scaling = 1.0;
  constexpr Scalar aY_scaling = 2.0;
  constexpr Scalar aZ_scaling = 3.0;

  Teuchos::RCP<Omega_h::Mesh> meshOmegaH_src = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth, aX_scaling, aY_scaling, aZ_scaling);

  Teuchos::RCP<Omega_h::Mesh> meshOmegaH_trg = 
    PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth+1, aX_scaling, aY_scaling, aZ_scaling);

  Omega_h::Assoc  assoc;
  Omega_h::update_assoc(&assoc, assoc_pl);
  Omega_h::MeshSets mesh_sets_src = Omega_h::invert(meshOmegaH_src.get(), assoc);
  Omega_h::MeshSets mesh_sets_trg = Omega_h::invert(meshOmegaH_trg.get(), assoc);

  FEMesh<spaceDim> femesh_src = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH_src);
  FEMesh<spaceDim> femesh_trg = PlatoUtestHelpers::createFEMesh<spaceDim>(meshOmegaH_trg);

  Teuchos::ParameterList paramList;
  auto fields_src = Teuchos::rcp(new Fields(femesh_src, paramList));
  auto fields_trg = Teuchos::rcp(new Fields(femesh_trg, paramList));

  auto magneticFaceFlux_src = MagneticFaceFlux<Fields>();

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

  Omega_h::Write<Scalar> errorPlusOneCell(elemLids_trg.size());

  const auto elem_node_ids_src = femesh_src.elem_node_ids;
  const auto node_coords_src = femesh_src.node_coords;
  const auto elemFaceIDs_src = femesh_src.elem_face_ids;
  const auto elemFaceOrientations_src = femesh_src.elem_face_orientations;

  const auto elem_node_ids_trg = femesh_trg.elem_node_ids;
  const auto node_coords_trg = femesh_trg.node_coords;
  const auto elemFaceIDs_trg = femesh_trg.elem_face_ids;
  const auto elemFaceOrientations_trg = femesh_trg.elem_face_orientations;

  const auto magneticFaceFlux = MagneticFaceFlux<Fields>();


  // Can only do a single matrix for a thread, so no looping over elements.
  // Hopefully Omega_h will eventually be looping over cavities and there will
  // be lots of cavities.
  const size_type inull=std::numeric_limits<size_type>::max();
  const std::string debuggingName("mhdMatrixAssemblyTests");
  Kokkos::parallel_for(debuggingName, Kokkos::RangePolicy<Kokkos::Serial>(0, 1), [&] (const int) {
      constexpr int maxFaceCount = maxElem*ElemFaceCount;
      size_type elemFaceLocalIDs_trg [maxElem][ElemFaceCount];
      size_type elemLocalIDsFace_trg [maxFaceCount][4];
      size_type elemFaceIsSurface_trg[maxElem][ElemFaceCount];
      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
        for (int face = 0; face < ElemFaceCount; ++face) {
          elemFaceLocalIDs_trg[elem][face] = inull;
          elemLocalIDsFace_trg[elem][face] = inull;
          elemFaceIsSurface_trg[elem][face] = 0;
        }
      }
      for (int elem = 0; elem < maxElem*ElemFaceCount; ++elem) {
        for (int face = 0; face < 4; ++face) {
          elemLocalIDsFace_trg[elem][face] = inull;
        }
      }
      int number_of_faces = 0;
      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
        for (int face = 0; face < ElemFaceCount; ++face) {
          if (elemFaceLocalIDs_trg[elem][face] == inull) {
            int n = 0;
            for (int ielem = elem; ielem < elemLids_trg.size() && n < 2; ++ielem) {
              for (int iface = face; iface < ElemFaceCount && n < 2; ++iface) {
                if (elemFaceIDs_trg(elem,face) == elemFaceIDs_trg(ielem,iface)) {
                  if (elemFaceLocalIDs_trg[ielem][iface] == inull) {
                    elemFaceLocalIDs_trg[ielem][iface] = number_of_faces; 
                    ++n;
                  }
                }
              }
            }
            ++number_of_faces;
            if (n==1) elemFaceIsSurface_trg[elem][face] = 1;
          }
        }
      }
      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
        for (int face = 0; face < ElemFaceCount; ++face) {
          const size_type id = elemFaceLocalIDs_trg[elem][face];
          const int i = elemLocalIDsFace_trg[id][0] == inull ? 0 : 2;
          elemLocalIDsFace_trg[id][i  ] = elem;
          elemLocalIDsFace_trg[id][i+1] = face;
        }
      }
      printf ("%d", (int)elemFaceIsSurface_trg[0][0]);
      printf ("%d", (int)elemLocalIDsFace_trg[0][0]);

      double faceFluxSource[maxFaceCount];
      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
        for (int face = 0; face < ElemFaceCount; ++face) {
          const int sign = elemFaceOrientations_trg  (elem,face);
          const size_t faceGID = elemFaceIDs_trg     (elem,face);
          const size_t faceLID = elemFaceLocalIDs_trg[elem][face];
          faceFluxSource[faceLID] = sign * magneticFaceFlux(faceGID);  
        }       
      }

      Omega_h::Matrix<maxFaceCount,maxFaceCount> M;
      Omega_h::Matrix<maxElem,maxFaceCount> Q;
      Omega_h::Matrix<maxElem,maxFaceCount> D;
      Omega_h::Vector<maxElem>              q;
      Omega_h::Vector<maxElem>              d;
     
      for (int iface = 0; iface < maxFaceCount; ++iface) {
        for (int jface = 0; jface < maxFaceCount; ++jface) {
          M(iface,jface) = 0;
        }
      }
      for (int elem = 0; elem < maxElem; ++elem) {
        q(elem) = 0;
        d(elem) = 0;
        for (int face = 0; face < maxFaceCount; ++face) {
          Q(elem,face) = 0;
          D(elem,face) = 0;
        }
      }

      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
        q(elem) = 0;
        for (int face = 0; face < ElemFaceCount; ++face) {
          const int sign = elemFaceOrientations_trg(elem,face);
          const size_t iface = elemLocalIDsFace_trg[elem][face];
          Q(elem,iface) = sign;
        }
      }

      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
        d(elem) = 1;
        for (int face = 0; face < ElemFaceCount; ++face) {
          if (1 == elemFaceIsSurface_trg[elem][face]) {
            const size_t face = elemLocalIDsFace_trg[elem][face];
            D(elem,face) = faceFluxSource[face];
          }
        }
      }

      for (int elem = 0; elem < elemLids_trg.size(); ++elem) {
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
        double moments[Polynomial::nterms] = {0.}; 
        r3d::reduce<moment>(poly_trg, moments);

        for (int iface = 0; iface < ElemFaceCount; ++iface) {
          Omega_h::Few<Omega_h::Vector<spaceDim>,vert> nodalCoordinates;
          for (int j=0; j<vert; ++j)
             for (int i=0; i<spaceDim; ++i)
                nodalCoordinates[j][i] = nodalCoordinates_trg[j][i];
          Omega_h::Few<double,4> ifacesomething={0};
          ifacesomething[iface]=1;
          Omega_h::Vector<3> iconstantTerm;
          lgr::Scalar ilinearFactor;
          lgr::elementPhysicalFacePolynomial( /*input*/
					     nodalCoordinates,
					     ifacesomething,
					     /*output*/
					     iconstantTerm,
					     ilinearFactor );
 
          for (int jface = 0; jface < ElemFaceCount; ++jface) {
            Omega_h::Few<double,4> jfacesomething={0};
            jfacesomething[jface]=1;
            Omega_h::Vector<3> jconstantTerm;
            lgr::Scalar jlinearFactor;
            lgr::elementPhysicalFacePolynomial( /*input*/
	  				       nodalCoordinates,
					       jfacesomething,
					       /*output*/
					       jconstantTerm,
					       jlinearFactor );
            
            double src_poly[Polynomial::nterms]={0.};
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

            double integral = 0;
            for (int i=0; i<Polynomial::nterms; ++i) {
              integral += moments[i]*src_poly[i];
            }    
            integral *= elemFaceOrientations_trg(elem,iface);
            integral *= elemFaceOrientations_trg(elem,jface);

            const size_t i = elemLocalIDsFace_trg[elem][iface];
            const size_t j = elemLocalIDsFace_trg[elem][jface];
            M(i,j) += integral;
          }
        }
      }
#if 0
      Tet4 nodalCoordinates_trg;
      const int e_trg = 1;
      const size_t ielem_trg = elemLids_trg[e_trg];
 
      for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_ids_trg(ielem_trg, i);
        r3d::Vector<spaceDim> &coord = nodalCoordinates_trg[i];
        coord[0] = node_coords_trg(n, 0);
        coord[1] = node_coords_trg(n, 1);
        coord[2] = node_coords_trg(n, 2);
      } 

      //integral of source B over a single target/source intersection
      auto L =  LAMBDA_EXPRESSION(int e_src, Omega_h::Vector<3> &integralB) {

          Tet4 nodalCoordinates_src;
          const size_t ielem_src = elemLids_src[e_src];

          for (int i = 0; i < ElemNodeCount; ++i) {
            const int n = elem_node_ids_src(ielem_src, i);
            r3d::Vector<spaceDim> &coord = nodalCoordinates_src[i];
            coord[0] = node_coords_src(n, 0); 
            coord[1] = node_coords_src(n, 1); 
            coord[2] = node_coords_src(n, 2);
          } 

          Omega_h::Vector<ElemFaceCount> faceFluxSource;
          for (int face = 0; face < ElemFaceCount; ++face) {
            const int sign = elemFaceOrientations_src(ielem_src,face);
            const size_t faceID = elemFaceIDs_src(ielem_src,face);
            faceFluxSource[face] = sign * magneticFaceFlux(faceID);  
          }       
       
          Omega_h::Few<Omega_h::Vector<spaceDim>,vert> nodalCoordinates;
          for (int j=0; j<vert; ++j)
             for (int i=0; i<spaceDim; ++i)
                nodalCoordinates[j][i] = nodalCoordinates_src[j][i];
          Omega_h::Vector<3> constantTerm;
          lgr::Scalar linearFactor;
          lgr::elementPhysicalFacePolynomial( /*input*/
					     nodalCoordinates,
					     faceFluxSource,
					     /*output*/
					     constantTerm,
					     linearFactor );
 
          double src_poly[3][Polynomial::nterms]={{0.}};
          src_poly[0][0] = constantTerm[0];
          src_poly[0][1] = linearFactor;
          src_poly[1][0] = constantTerm[1];
          src_poly[1][2] = linearFactor;
          src_poly[2][0] = constantTerm[2];
          src_poly[2][3] = linearFactor;
      
          Polytope intersection;
          r3d::intersect_simplices(intersection, nodalCoordinates_src, nodalCoordinates_trg);
          double moments[Polynomial::nterms] = {0.}; 
          r3d::reduce<moment>(intersection, moments);

          for (int j=0; j<3; ++j) {
             for (int i=0; i<Polynomial::nterms; ++i) {
                integralB(j) += moments[i]*src_poly[j][i];
             }    
          }
      };//end lambda L

      //magnetic data to be computed for one target element
      Omega_h::Vector<3> integralMagneticFluxDensity; 
      integralMagneticFluxDensity(0) = integralMagneticFluxDensity(1) = integralMagneticFluxDensity(2) = 0.;

      //for each target element, loop over source elements
      //Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, elemLids_src.size()), L, integralMagneticFluxDensity);

      //exact solution for a single target element
      double target_volume(0.);
      Polytope poly_trg;
      r3d::init(poly_trg, nodalCoordinates_trg);
      r3d::reduce<0>(poly_trg, &target_volume);
      Omega_h::Vector<3> Bexact;
      Bexact(0) = 1.; Bexact(1) = 2.; Bexact(2) = 5.; 
      Bexact*=target_volume;
      errorPlusOneCell[e_trg] = 1.0 + Omega_h::norm(Bexact-integralMagneticFluxDensity)/Omega_h::norm(Bexact);

#endif
  }); //end Kokkos::parallel_for outer loop over target elements 

  Omega_h::HostWrite<Scalar> errorPlusOne(errorPlusOneCell);
  for (int i=0; i<elemLids_trg.size(); ++i) 
    TEST_FLOATING_EQUALITY(errorPlusOne[i], 1.0, 2.0e-14);

}


