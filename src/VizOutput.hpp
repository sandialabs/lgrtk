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

#ifndef LGR_VIZ_OUTPUT_HPP
#define LGR_VIZ_OUTPUT_HPP

#include <Teuchos_ParameterList.hpp>
#include <Omega_h_file.hpp>
#include <string>

namespace lgr {

class VizOutput {
 public:
  VizOutput(
      Omega_h::Mesh*          mesh,
      std::string const&      output_path,
      Teuchos::ParameterList& viz_pl,
      double                    restart_time);

  template <typename fields_type>
  void writeOutputFile(
      const fields_type& mesh_fields,
      int                step,
      int                state,
      double             time);

  Omega_h::vtk::Writer writer;
  Omega_h::TagSet      viz_tags;
};

template <class fields_type>
void VizOutput::writeOutputFile(
    const fields_type &mesh_fields,
    int                cycle,
    int                state,
    double             time) {

  mesh_fields.copyTagsToMesh(viz_tags, state);

  writer.write(Omega_h::Int(cycle), Omega_h::Real(time), viz_tags);

  if (mesh_fields.femesh.omega_h_mesh->comm()->rank() == 0) {
    std::cout << "Plot Dump | cycle: " << cycle << " time: " << time << '\n';
  }

  mesh_fields.cleanTagsFromMesh(viz_tags);
}

}  // end namespace lgr

#endif
