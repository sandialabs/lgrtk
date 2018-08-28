/*
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
*/
#if !defined(CONTACT_FORCE_HPP)
#define CONTACT_FORCE_HPP

#include <VectorContribution.hpp>

namespace lgr {

template <int SpatialDim>
void compute_contact_forces(VectorContributions<SpatialDim>& forces,
                            Teuchos::ParameterList&          params);

template <int SpatialDim>
class ContactForce : public VectorContribution<SpatialDim> {
 public:
  using typename VectorContribution<SpatialDim>::geom_array_type;
  using typename VectorContribution<SpatialDim>::node_coords_type;

  std::string name_{""};

  NodeSet node_set_;

  Kokkos::View<const Scalar*[SpatialDim], MemSpace> value_;

  Scalar gap_length_{0.0};

  ContactForce(std::string const& name, Teuchos::ParameterList& params);

  virtual void add_to(geom_array_type field) const override final;

  virtual ~ContactForce();

};  // class ContactForce

template <int SpatialDim>
class PenaltyContactForce : public ContactForce<SpatialDim> {
 public:
  using typename ContactForce<SpatialDim>::geom_array_type;
  using typename ContactForce<SpatialDim>::node_coords_type;

  Scalar penalty_coefficient_{0.0};

  PenaltyContactForce(std::string const& name, Teuchos::ParameterList& params);

  virtual void update(Omega_h::MeshSets const&,
                      Scalar const           time,
                      node_coords_type const coords) override final;

  virtual ~PenaltyContactForce();

};  // class ContactForce

}  // namespace lgr

#endif  // CONTACT_FORCE_HPP
