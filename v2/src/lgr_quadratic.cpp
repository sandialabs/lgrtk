#include <lgr_quadratic.hpp>
#include <Omega_h_align.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_simplex.hpp>

namespace lgr {

static Omega_h::LOs count_nodes_by_verts(Omega_h::Mesh& mesh) {
  auto verts2edges = mesh.ask_up(0, 1);
  Omega_h::Write<Omega_h::LO> counts(mesh.nverts(), "node_counts_by_vert");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO v) {
    Omega_h::LO count = 1;
    auto adj_edge_begin = verts2edges.a2ab[v];
    auto adj_edge_end = verts2edges.a2ab[v + 1];
    for (auto e = adj_edge_begin; e < adj_edge_end; ++e) {
      auto edge_code = verts2edges.codes[e];
      if (Omega_h::code_which_down(edge_code) == 0) ++count;
    }
    counts[v] = count;
  };
  Omega_h::parallel_for("count nodes by vert", mesh.nverts(), std::move(functor));
  return counts;
}

Omega_h::Few<Omega_h::LOs, 2> number_p2_nodes(Omega_h::Mesh& mesh) {
  auto verts2edges = mesh.ask_up(0, 1);
  Omega_h::Write<Omega_h::LO> vtx_nodes(mesh.nverts(), "vtx_node_nmbr");
  Omega_h::Write<Omega_h::LO> edge_nodes(mesh.nedges(), "edge_node_nmbr");
  auto counts = count_nodes_by_verts(mesh);
  auto offsets = Omega_h::offset_scan(counts, "node_number_by_vert");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO v) {
    auto node = offsets[v];
    vtx_nodes[v] = node;
    auto adj_edge_begin = verts2edges.a2ab[v];
    auto adj_edge_end = verts2edges.a2ab[v + 1];
    for (auto e = adj_edge_begin; e < adj_edge_end; ++e) {
      auto edge = verts2edges.ab2b[e];
      auto edge_code = verts2edges.codes[e];
      if (Omega_h::code_which_down(edge_code) == 0) edge_nodes[edge] = ++node;
    }
    OMEGA_H_CHECK(offsets[v + 1] == (node + 1));
  };
  Omega_h::parallel_for("number p2 nodes", mesh.nverts(), std::move(functor));
  Omega_h::Few<Omega_h::LOs, 2> nodes{vtx_nodes, edge_nodes};
  return nodes;
}

Omega_h::LOs build_p2_elems2nodes(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto elem_dim = mesh.dim();
  auto ndown_verts = Omega_h::simplex_degree(elem_dim, 0);
  auto ndown_edges = Omega_h::simplex_degree(elem_dim, 1);
  auto nnodes = nodes[0].size() + nodes[1].size();
  auto nnodes_per_elem = ndown_verts + ndown_edges;
  OMEGA_H_CHECK(nnodes == (mesh.nverts() + mesh.nedges()));
  auto elems2verts = mesh.ask_down(elem_dim, 0).ab2b;
  auto elems2edges = mesh.ask_down(elem_dim, 1).ab2b;
  auto e2n_size = mesh.nelems() * nnodes_per_elem;
  Omega_h::Write<Omega_h::LO> elems2nodes(e2n_size, "elems2nodes");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO elem) {
    int node_ctr = 0;
    for (auto v = 0; v < ndown_verts; ++v) {
      auto vtx = elems2verts[elem * ndown_verts + v];
      auto node = nodes[0][vtx];
      elems2nodes[elem * nnodes_per_elem + node_ctr] = node;
      ++node_ctr;
    }
    for (auto e = 0; e < ndown_edges; ++e) {
      auto edge = elems2edges[elem * ndown_edges + e];
      auto node = nodes[1][edge];
      elems2nodes[elem * nnodes_per_elem + node_ctr] = node;
      ++node_ctr;
    }
  };
  Omega_h::parallel_for("build p2 elems2nodes", mesh.nelems(), std::move(functor));
  return elems2nodes;
}

Omega_h::Adj build_p2_nodes2elems(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  (void)mesh;
  (void)nodes;
  return Omega_h::Adj{};
}

}
