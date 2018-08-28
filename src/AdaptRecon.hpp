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

#ifndef ADAPT_RECON_HPP
#define ADAPT_RECON_HPP

#include <Teuchos_ParameterList.hpp>
#include <Omega_h_adapt.hpp>

#include "Fields.hpp"
#include "MeshIO.hpp"

namespace lgr {

template <int SpatialDim>
class AdaptRecon {
  using DefaultFields = Fields<SpatialDim>;
  static constexpr int default_dim = SpatialDim;
  static constexpr int elem_node_count = DefaultFields::ElemNodeCount;

 private:
  bool   is_adaptive;
  bool   always_adapt;
  double trigger_quality;
  double trigger_length;
  bool                 should_debug_momentum;
  bool                 should_advect_original_implied;
  bool                 always_recompute_metric;
  DefaultFields&       mesh_fields;
  comm::Machine        machine;
  Omega_h::AdaptOpts   adapt_opts;
  Omega_h::MetricInput metric_input;
  Omega_h::TagSet      metric_tags;
  Omega_h::TagSet      adapt_tags;
  Omega_h::TagSet      restart_tags;

  void preAdapt(int next_state);
  void postAdapt(MeshIO& out_obj, int next_state);
  void cleanupRestart();
  bool shouldAdapt(Omega_h::AdaptOpts const& opts) const;

 public:
  AdaptRecon(
      Teuchos::ParameterList& problem_,
      DefaultFields&          mesh_fields_,
      comm::Machine           machine_);


  bool   isAdaptive() const { return is_adaptive; }

  void computeMetric(int current_state, int next_state);
  bool adaptMeshAndRemapFields(
      MeshIO& out_obj, const int current_state, const int next_state);

  void writeRestart(std::string const& path, const int next_state);
  void loadRestart(MeshIO&   out_obj);
};
extern template class AdaptRecon<3>;
extern template class AdaptRecon<2>;
}  // namespace lgr

#endif
