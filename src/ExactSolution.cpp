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

// because of vagaries of C++ compilers, it is important to include Omega_h_expr.hpp
// prior to things that include Teuchos_any.hpp (which, eventually, ExactSolution.hpp does)
#include <Omega_h_expr.hpp>

#include "ExactSolution.hpp"
#include "LGRLambda.hpp"
#include "FieldDB.hpp"

#include <Kokkos_Core.hpp>
#include <Omega_h_teuchos.hpp>
#include <MeshFixture.hpp>
#include <Omega_h_array_ops.hpp>

namespace lgr {

template<int SpatialDim >
void ExactSolution(const Teuchos::ParameterList& param_list,
                   const Teuchos::RCP<Fields<SpatialDim>> mesh_fields,
                   const int next_state,
                   const comm::Machine &machine) {
  typedef Fields<SpatialDim> Fields ;
  typedef typename Fields::FEMesh FEMesh;

  const FEMesh &femesh = mesh_fields->femesh;
  typename Fields::node_coords_type coords          = femesh.node_coords; 
  typename Fields::elem_node_ids_type elem_node_ids = femesh.elem_node_ids; 
  Omega_h::Mesh *mesh = femesh.omega_h_mesh;
  const int ent_dim =  Omega_h::get_ent_dim_by_name(mesh, "Cell");
  const std::string field_name  = "mass_density"; //param_list.get<std::string>("Field");
  const std::string expr_string = param_list.get<std::string>("Value");
  Omega_h::TagSet tags;
  tags[std::size_t(ent_dim)].insert(field_name);
  mesh_fields->copyTagsToMesh(tags, next_state);

  Omega_h::Read<double> actual = mesh->get_array<double>(ent_dim, field_name);

  const int dim = mesh->dim();

  const int num_elem     = elem_node_ids.extent(0);
  const int num_node_ids = elem_node_ids.size();
  Omega_h::Write<double> x_w(num_node_ids);
  Omega_h::Write<double> y_w(num_node_ids);
  Omega_h::Write<double> z_w(num_node_ids);
  auto prepare = OMEGA_H_LAMBDA(int elem) {
    x_w[elem] = 0;
    y_w[elem] = 0;
    z_w[elem] = 0;
    for (int i = 0; i < Fields::ElemNodeCount; ++i) {
      auto node = elem_node_ids(elem, i); 
      x_w[elem] += coords(node, 0); 
      if (1<dim) y_w[elem] += coords(node, 1); 
      if (2<dim) z_w[elem] += coords(node, 2); 
    }   
    x_w[elem] /= Fields::ElemNodeCount;
    y_w[elem] /= Fields::ElemNodeCount;
    z_w[elem] /= Fields::ElemNodeCount;
  };  
  Kokkos::parallel_for(num_elem, prepare);
  Omega_h::ExprReader reader(num_elem, dim); 
  reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_w)));
  reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_w)));
  reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_w)));
  Teuchos::any result;
  reader.read_string(result, expr_string, field_name);
  reader.repeat(result);
  auto exact = Teuchos::any_cast<Omega_h::Reals>(result);
  long double diff = 0;
  
  const typename Fields::array_type vol = ElementVolume<Fields>();
  Omega_h::Write<double> diffs_w(num_elem);
  auto get_diffs = OMEGA_H_LAMBDA(int i) {
    diffs_w[i] = std::abs(actual[i] - exact[i])*vol[i];
  };
  Kokkos::parallel_for(num_elem, get_diffs);
  diff = Omega_h::get_sum(Omega_h::Reals(diffs_w));
  diff = comm::sum(machine, diff);
  if (!comm::rank(machine)) 
    std::cout<<" L1 difference with Exact Soluion of "<<expr_string<<" is:"<<diff<<std::endl;
  mesh_fields->cleanTagsFromMesh(tags);
}

#define LGR_EXPL_INST(SpatialDim) \
template \
void ExactSolution(const Teuchos::ParameterList& param_list, \
                   const Teuchos::RCP<Fields<SpatialDim>> mesh_fields, \
                   const int next_state, \
                   const comm::Machine &machine);
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

}
