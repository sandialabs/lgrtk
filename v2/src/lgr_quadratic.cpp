#include <lgr_quadratic.hpp>
#include <Omega_h_align.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_simplex.hpp>

namespace lgr {

// we number edge nodes by which vertex 'owns' the edge
// (the first downward vertex of the edge is the edge 'owner')
// this ensures that we can index into edge node arrays without
// race conditions when we loop over vertices

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
  auto nnodes_per_elem = ndown_verts + ndown_edges;
  auto nnodes = nodes[0].size() + nodes[1].size();
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

static Omega_h::LOs count_nodes2elems(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto elem_dim = mesh.dim();
  auto nnodes = nodes[0].size() + nodes[1].size();
  OMEGA_H_CHECK(nnodes == (mesh.nverts() + mesh.nedges()));
  auto verts2edges = mesh.ask_up(0, 1);
  auto verts2elems = mesh.ask_up(0, elem_dim);
  auto edges2elems = mesh.ask_up(1, elem_dim);
  Omega_h::Write<Omega_h::LO> node_counts(nnodes, "node_counts");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO v) {
    auto vtx_node = nodes[0][v];
    auto nvtx_adj_elems = verts2elems.a2ab[v + 1] - verts2elems.a2ab[v];
    node_counts[vtx_node] = nvtx_adj_elems;
    auto adj_edge_begin = verts2edges.a2ab[v];
    auto adj_edge_end = verts2edges.a2ab[v + 1];
    for (auto e = adj_edge_begin; e < adj_edge_end; ++e) {
      auto edge = verts2edges.ab2b[e];
      auto edge_code = verts2edges.codes[e];
      if (Omega_h::code_which_down(edge_code) == 0) {
        auto edge_node = nodes[1][edge];
        auto nedge_adj_elems = edges2elems.a2ab[edge + 1] - edges2elems.a2ab[edge];
        node_counts[edge_node] = nedge_adj_elems;
      }
    }
  };
  Omega_h::parallel_for("count nodes2elems", mesh.nverts(), std::move(functor));
  return node_counts;
}

Omega_h::Adj build_p2_nodes2elems(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto elem_dim = mesh.dim();
  auto verts2edges = mesh.ask_up(0, 1);
  auto verts2elems = mesh.ask_up(0, elem_dim);
  auto edges2elems = mesh.ask_up(1, elem_dim);
  auto counts = count_nodes2elems(mesh, nodes);
  auto a2ab = Omega_h::offset_scan(counts, "nodes2elems_a2ab");
  Omega_h::Write<Omega_h::LO> ab2b(a2ab.last(), "nodes2elems_ab2b");
  Omega_h::Write<Omega_h::I8> codes(a2ab.last(), "nodes2elems_codes");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO v) {
    auto vtx_node = nodes[0][v];
    auto ab = a2ab[vtx_node];
    auto vtx_adj_elem_begin = verts2elems.a2ab[v];
    auto vtx_adj_elem_end = verts2elems.a2ab[v + 1];
    for (auto elem_idx = vtx_adj_elem_begin; elem_idx < vtx_adj_elem_end; ++elem_idx) {
      auto elem = verts2elems.ab2b[elem_idx];
      auto code = verts2elems.codes[elem_idx];
      ab2b[ab] = elem;
      codes[ab] = code;
      ab++;
    }
    auto adj_edge_begin = verts2edges.a2ab[v];
    auto adj_edge_end = verts2edges.a2ab[v + 1];
    for (auto edge_idx = adj_edge_begin; edge_idx < adj_edge_end; ++edge_idx) {
      auto edge = verts2edges.ab2b[edge_idx];
      auto edge_code = verts2edges.codes[edge_idx];
      if (Omega_h::code_which_down(edge_code) == 0) {
        auto edge_adj_elem_begin = edges2elems.a2ab[edge];
        auto edge_adj_elem_end = edges2elems.a2ab[edge + 1];
        for (auto elem_idx = edge_adj_elem_begin; elem_idx < edge_adj_elem_end; ++elem_idx) {
          auto elem = edges2elems.ab2b[elem_idx];
          auto code = edges2elems.codes[elem_idx];
          ab2b[ab] = elem;
          codes[ab] = code;
          ab++;
        }
      }
    }
  };
  Omega_h::parallel_for("build p2 nodes2elems", mesh.nverts(), std::move(functor));
  return Omega_h::Adj{a2ab, ab2b, codes};
}

Omega_h::Reals build_p2_node_coords(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto elem_dim = mesh.dim();
  auto vtx_coords = mesh.coords();
  auto verts2edges = mesh.ask_up(0, 1);
  auto edge_coords = Omega_h::average_field(&mesh, 1, elem_dim, vtx_coords);
  auto nnodes = nodes[0].size() + nodes[1].size();
  OMEGA_H_CHECK(nnodes == (mesh.nverts() + mesh.nedges()));
  Omega_h::Write<Omega_h::Real> node_coords(nnodes * elem_dim, "node_coords");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO v) {
    auto vtx_node = nodes[0][v];
    for (int dim = 0; dim < elem_dim; ++dim) {
      auto coord = vtx_coords[v * elem_dim + dim];
      node_coords[vtx_node * elem_dim + dim] = coord;
    }
    auto adj_edge_begin = verts2edges.a2ab[v];
    auto adj_edge_end = verts2edges.a2ab[v + 1];
    for (auto e = adj_edge_begin; e < adj_edge_end; ++e) {
      auto edge = verts2edges.ab2b[e];
      auto code = verts2edges.codes[e];
      if (Omega_h::code_which_down(code) == 0) {
        auto edge_node = nodes[1][edge];
        for (int dim = 0; dim < elem_dim; ++dim) {
          auto coord = edge_coords[edge * elem_dim + dim];
          node_coords[edge_node * elem_dim + dim] = coord;
        }
      }
    }
  };
  Omega_h::parallel_for("build p2 node_coords", mesh.nverts(), std::move(functor));
  return node_coords;
}

}
