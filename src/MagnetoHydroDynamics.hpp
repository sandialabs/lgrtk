//@HEADER
// ************************************************************************
//
//                        lgr v. 1.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  Glen A. Hansen (gahanse@sandia.gov)
//
// ************************************************************************
//@HEADER

#ifndef LGR_MAGNETOHYDRODYNAMICS_HPP
#define LGR_MAGNETOHYDRODYNAMICS_HPP

#include "Cubature.hpp"
#include "ElementHelpers.hpp"
#include "FieldDB.hpp"
#include "PhysicalConstants.hpp"

namespace lgr{

  template<int SpatialDim>
  class MHD {

  public:

    typedef Omega_h::Vector<3> Vector;
    typedef Omega_h::Matrix<3,3> Tensor;

    typedef lgr::Fields<SpatialDim> Fields;
    typedef typename Fields::size_type size_type;
    
    const typename Fields::array_type magneticFaceFlux;
    const typename Fields::elem_face_ids_type elemFaceIDs;
    const typename Fields::elem_face_orient_type elemFaceOrientations;
    
    typename lgr::Cubature::RefPointsView points_;
    typename lgr::Cubature::WeightsView weights_;

    MHD( const Fields &arg_mesh_fields ) 
      : magneticFaceFlux(MagneticFaceFlux<Fields>())
      , elemFaceIDs(arg_mesh_fields.femesh.elem_face_ids)
      , elemFaceOrientations(arg_mesh_fields.femesh.elem_face_orientations)
      , points_("quadrature points", 4, SpatialDim)
      , weights_("quadrature weights", 4)
    {
      Cubature::getCubature( SpatialDim, 2, points_, weights_);
    }

    KOKKOS_INLINE_FUNCTION constexpr int numberGaussPoints() const {return 4;}

    KOKKOS_INLINE_FUNCTION constexpr Scalar sumWeights() const {return +1./6.;}

    KOKKOS_INLINE_FUNCTION 
    Scalar 
    elementMagneticEnergy( const int ielem, 
			   const Scalar * const x,
			   const Scalar * const y,
			   const Scalar * const z ) const
    {
      const Scalar volume = tet4Volume(x,y,z);
      const Scalar J = volume/sumWeights();
      
      constexpr int elemFaceCount = Fields::ElemFaceCount;
      Scalar faceFlux[elemFaceCount];
      for ( int face=0; face<elemFaceCount; ++face ) {
	const int sign = elemFaceOrientations(ielem,face);
	const size_type faceID = elemFaceIDs(ielem,face);
	faceFlux[face] = sign * magneticFaceFlux(faceID);  
      }	
      
      Scalar magneticEnergy(0.0);
      for (int gp=0; gp<numberGaussPoints(); ++gp) {
	const Scalar wt = weights_(gp);
	const Scalar xi[] = { points_(gp,0), points_(gp,1), points_(gp,2) };
	const auto faceBasis = comp_face_basis( x,y,z, xi );
	Vector B = Omega_h::zero_vector<3>();
	for (int face=0; face<elemFaceCount; ++face)
	  B += faceFlux[face]*faceBasis[face];
	const Scalar dV = J * wt;
	magneticEnergy += 0.5 * Omega_h::norm_squared(B) * dV;
      }
      constexpr Scalar moo = PhysicalConstants::VacuumMu;
      magneticEnergy /= moo;
      return magneticEnergy;
    }

    KOKKOS_INLINE_FUNCTION 
    Vector
    elementMagneticFluxDensity( const int ielem, 
				const Scalar * const x,
				const Scalar * const y,
				const Scalar * const z ) const
    {
      constexpr Scalar xi[] = {1./3.,1./3.,1./3.};
      constexpr int elemFaceCount = Fields::ElemFaceCount;
      const auto faceBasis = comp_face_basis( x,y,z, xi );
      Vector B = Omega_h::zero_vector<3>();
      for (int face=0; face<elemFaceCount; ++face) {
	const int sign = elemFaceOrientations(ielem,face);
	const size_type faceID = elemFaceIDs(ielem,face);
	const Scalar globalFaceFlux = magneticFaceFlux(faceID);  
	const Scalar localFaceFlux = sign * globalFaceFlux;
	B += localFaceFlux*faceBasis[face];
      }
      return B;
    }

    KOKKOS_INLINE_FUNCTION 
    Tensor
    elementMagneticStressTensor( const int ielem, 
				 const Scalar * const x,
				 const Scalar * const y,
				 const Scalar * const z ) const
    {
      const Scalar volume = tet4Volume(x,y,z);
      const Scalar J = volume/sumWeights();
      constexpr int elemFaceCount = Fields::ElemFaceCount;
      Scalar faceFlux[elemFaceCount];
      for ( int face=0; face<elemFaceCount; ++face ) {
	const int sign = elemFaceOrientations(ielem,face);
	const size_type faceID = elemFaceIDs(ielem,face);
	faceFlux[face] = sign * magneticFaceFlux(faceID);  
      }	
      
      Tensor sigma = Omega_h::zero_matrix<3,3>();
      for (int gp=0; gp<numberGaussPoints(); ++gp) {
	Scalar wt = weights_(gp);
	const Scalar xi[] = { points_(gp,0), points_(gp,1), points_(gp,2) };
	const auto faceBasis = comp_face_basis( x,y,z, xi );
	Vector B = Omega_h::zero_vector<3>();
	for (int face=0; face<elemFaceCount; ++face)
	  B += faceFlux[face]*faceBasis[face];
	const Scalar dV = J * wt;
	sigma += dV * Omega_h::outer_product(B,B);
	sigma -= dV * ( 0.5 * Omega_h::norm_squared(B) ) * (Omega_h::identity_matrix<3,3>());	
      }
      sigma /= volume;
      constexpr Scalar moo = PhysicalConstants::VacuumMu;
      sigma /= moo;
      return sigma;
    }

    KOKKOS_INLINE_FUNCTION 
    Scalar
    elementAlfvenWaveModulus( const int ielem, 
			      const Scalar * const x,
			      const Scalar * const y,
			      const Scalar * const z ) const
    {
      const Vector B = elementMagneticFluxDensity(ielem, x,y,z);
      Scalar modulus = Omega_h::norm_squared(B);
      constexpr Scalar moo = PhysicalConstants::VacuumMu;
      modulus /= moo;
      return modulus;
    }

    MHD() = delete;
    MHD(const MHD &) = default;
    MHD& operator=(const MHD &) = delete;

  };//end class MHD

}//end namespace lgr
#endif
