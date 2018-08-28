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

#ifndef LGR_TIME_INTEGRATION_HPP
#define LGR_TIME_INTEGRATION_HPP

#include "MeshFixture.hpp"
#include "Fields.hpp"
#include "MaterialModels.hpp"
#include "VectorContribution.hpp"
#include <list>

namespace lgr {

struct PerformanceData {
  double mesh_time;
  double init_time;
  double internal_force_time;
  double midpoint;
  double comm_time;
  size_t number_of_steps;

  PerformanceData();

  void best(const PerformanceData &rhs);
};  //end struct PerformanceData

template <int SpatialDim>
class LagrangianStep {
 public:
  using FixtureType = MeshFixture<SpatialDim>;
  typedef typename FixtureType::execution_space              execution_space;
  typedef lgr::Fields<SpatialDim> Fields;

 private:
  std::list<
      std::shared_ptr<MaterialModelBase<SpatialDim>>>
      &          theMaterialModels_;
  Fields &       meshFields_;
  comm::Machine  machine_;
  Omega_h::Mesh *mesh_;

 public:
  LagrangianStep(
      std::list<std::shared_ptr<
          MaterialModelBase<SpatialDim>>>
          &          material_models,
      Fields &       mesh_fields,
      comm::Machine  machine,
      Omega_h::Mesh *mesh);

  PerformanceData advanceTime(
      const VectorContributions<SpatialDim>& accel_contribs,
      const VectorContributions<SpatialDim>& internal_force_contribs,
      const Scalar                      simtime,
      const Scalar                      dt,
      const int                         current_state,
      const int                         next_state) const;

};  //end class LagrangianStep

extern template class LagrangianStep<3>;
extern template class LagrangianStep<2>;

}  //end namespace lgr

#endif
