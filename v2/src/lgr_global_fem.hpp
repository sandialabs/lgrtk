#ifndef LGR_GLOBAL_FEM_HPP
#define LGR_GLOBAL_FEM_HPP

#include <lgr_simulation.hpp>
#include <lgr_linear_algebra.hpp>
#include <lgr_for.hpp>
#include <lgr_field_access.hpp>
#include <lgr_field_index.hpp>
#include <lgr_element_functions.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_array_ops.hpp>

namespace lgr {

template <typename Elem>
class Global_FEM {
private:
  Simulation& sim;
public:
  Global_FEM(Simulation& sim_in) :
      sim(sim_in) {
  }
  GlobalMatrix learn_disc() {
     GlobalMatrix matrix;
     auto const verts_to_other_verts = sim.disc.mesh.ask_star(0);
     auto const verts_to_selves =
         Omega_h::identity_graph(sim.disc.mesh.nverts());
     auto const verts_to_verts =
         Omega_h::add_edges(verts_to_selves, verts_to_other_verts);
     matrix.rows_to_columns = verts_to_verts;
     auto const nnz = verts_to_verts.a2ab.last();
     matrix.entries = Omega_h::Write<double>(nnz, "matrix entries");
     return matrix;
  }
  Omega_h::Read<double> lumped_mass() {
     // Create lumped mass matrix
     auto const points_to_w = sim.get(sim.weight);
     auto const nodes_to_elems = sim.nodes_to_elems();
     auto const nnodes = sim.disc.mesh.nverts();
     Omega_h::Write<double> vector = Omega_h::Write<double>(nnodes, 0.0);
     auto functor = OMEGA_H_LAMBDA(int node) {
       double node_weight = 0.0;
       auto const begin = nodes_to_elems.a2ab[node];
       auto const end = nodes_to_elems.a2ab[node + 1];
       for (auto node_elem = begin; node_elem < end; ++node_elem) {
         auto const elem = nodes_to_elems.ab2b[node_elem];
         auto const code = nodes_to_elems.codes[node_elem];
         auto const elem_node = Omega_h::code_which_down(code);
         double elem_weight = 0.0;
         for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
           auto const point = elem * Elem::points + elem_pt;
           auto const w = points_to_w[point];
           elem_weight += w;
         }
         node_weight += elem_weight * Elem::lumping_factor(elem_node);
       }
       vector[node] = node_weight;
     };
     parallel_for(nnodes, std::move(functor));

     return read(vector);
  }
  Omega_h::Read<double> inv_lumped_mass() {
     // Inverse of lumped mass matrix is 1 / diagonal entries
     return Omega_h::invert_each( lumped_mass() );
  }
  GlobalMatrix stiffness(Omega_h::Write<double> points_to_field) {
     // Create matrix
     GlobalMatrix const matrix = learn_disc();

     // Assembly with special tricks so no atomics needed
     constexpr int edges_per_elem = Omega_h::simplex_degree(Elem::dim, 1);
     constexpr int verts_per_elem = Omega_h::simplex_degree(Elem::dim, 0);
     Omega_h::Write<double> elems_to_vert_contribs(
         sim.disc.mesh.nelems() * verts_per_elem);
     Omega_h::Write<double> elems_to_edge_contribs(
         sim.disc.mesh.nelems() * edges_per_elem);
     auto const points_to_grad = sim.fields.get(sim.gradient);
     auto const points_to_weight = sim.fields.get(sim.weight);
     auto elem_functor = OMEGA_H_LAMBDA(int const elem) {
       for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
         auto const point = elem * Elem::points + elem_pt;
         auto const weight = points_to_weight[point];
         auto const f = points_to_field[point];
         auto const grads = getgrads<Elem>(points_to_grad, point);
         for (int elem_vert = 0; elem_vert < verts_per_elem; ++elem_vert) {
           auto const contrib =
               weight * f * (grads[elem_vert] * grads[elem_vert]);
           elems_to_vert_contribs[elem * verts_per_elem + elem_vert] = contrib;
         }
         for (int elem_edge = 0; elem_edge < edges_per_elem; ++elem_edge) {
           auto const elem_vert0 =
               Omega_h::simplex_down_template(Elem::dim, 1, elem_edge, 0);
           auto const elem_vert1 =
               Omega_h::simplex_down_template(Elem::dim, 1, elem_edge, 1);
           auto const contrib =
               weight * f * (grads[elem_vert0] * grads[elem_vert1]);
           elems_to_edge_contribs[elem * edges_per_elem + elem_edge] = contrib;
         }
       }
     };
     parallel_for(sim.disc.mesh.nelems(), std::move(elem_functor));
     Omega_h::Write<double> edges_to_value(sim.disc.mesh.nedges());
     auto const edges_to_elems = sim.disc.mesh.ask_up(1, Elem::dim);
     auto edge_functor = OMEGA_H_LAMBDA(int const edge) {
       auto const begin = edges_to_elems.a2ab[edge];
       auto const end = edges_to_elems.a2ab[edge + 1];
       double edge_value = 0.0;
       for (auto edge_elem = begin; edge_elem < end; ++edge_elem) {
         auto const elem = edges_to_elems.ab2b[edge_elem];
         auto const code = edges_to_elems.codes[edge_elem];
         auto const elem_edge = Omega_h::code_which_down(code);
         auto const contrib =
             elems_to_edge_contribs[elem * edges_per_elem + elem_edge];
         edge_value += contrib;
       }
       edges_to_value[edge] = edge_value;
     };
     parallel_for(sim.disc.mesh.nedges(), std::move(edge_functor));
     Omega_h::Write<double> verts_to_value(sim.disc.mesh.nverts());
     auto const verts_to_elems = sim.disc.mesh.ask_up(0, Elem::dim);
     auto vert_functor = OMEGA_H_LAMBDA(int const vert) {
       auto const begin = verts_to_elems.a2ab[vert];
       auto const end = verts_to_elems.a2ab[vert + 1];
       double vert_value = 0.0;
       for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
         auto const elem = verts_to_elems.ab2b[vert_elem];
         auto const code = verts_to_elems.codes[vert_elem];
         auto const elem_vert = Omega_h::code_which_down(code);
         auto const contrib =
             elems_to_vert_contribs[elem * verts_per_elem + elem_vert];
         vert_value += contrib;
       }
       verts_to_value[vert] = vert_value;
     };
     parallel_for(sim.disc.mesh.nverts(), std::move(vert_functor));
     auto const verts_to_edges = sim.disc.mesh.ask_up(0, 1);
     auto row_functor = OMEGA_H_LAMBDA(int const row) {
       auto const row_begin = matrix.rows_to_columns.a2ab[row];
       auto row_col = row_begin;
       matrix.entries[row_col++] = verts_to_value[row];
       auto const edge_begin = verts_to_edges.a2ab[row];
       auto const edge_end = verts_to_edges.a2ab[row + 1];
       for (auto vert_edge = edge_begin; vert_edge < edge_end; ++vert_edge) {
         auto const edge = verts_to_edges.ab2b[vert_edge];
         matrix.entries[row_col++] = edges_to_value[edge];
       }
     };
     parallel_for(sim.disc.mesh.nverts(), std::move(row_functor));
     return matrix;
  }
  GlobalMatrix stiffness(FieldIndex fi) {
     Omega_h::Write<double> points_to_field = sim.fields.set(fi);
     return stiffness(points_to_field);
  }
};

}  // namespace lgr

#endif
