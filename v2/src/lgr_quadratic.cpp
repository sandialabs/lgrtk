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

Omega_h::LOs build_p2_ents2nodes(Omega_h::Mesh& mesh, int ent_dim,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto ndown_verts = Omega_h::simplex_degree(ent_dim, 0);
  auto ndown_edges = Omega_h::simplex_degree(ent_dim, 1);
  auto nnodes_per_ent = ndown_verts + ndown_edges;
  auto nnodes = nodes[0].size() + nodes[1].size();
  OMEGA_H_CHECK(nnodes == (mesh.nverts() + mesh.nedges()));
  auto ents2verts = mesh.ask_down(ent_dim, 0).ab2b;
  auto ents2edges = mesh.ask_down(ent_dim, 1).ab2b;
  auto e2n_size = mesh.nents(ent_dim) * nnodes_per_ent;
  Omega_h::Write<Omega_h::LO> ents2nodes(e2n_size, "ents2nodes");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO ent) {
    int node_ctr = 0;
    for (auto v = 0; v < ndown_verts; ++v) {
      auto vtx = ents2verts[ent * ndown_verts + v];
      auto node = nodes[0][vtx];
      ents2nodes[ent * nnodes_per_ent + node_ctr] = node;
      ++node_ctr;
    }
    for (auto e = 0; e < ndown_edges; ++e) {
      auto edge = ents2edges[ent * ndown_edges + e];
      auto node = nodes[1][edge];
      ents2nodes[ent * nnodes_per_ent + node_ctr] = node;
      ++node_ctr;
    }
  };
  Omega_h::parallel_for("build p2 ents2nodes",
      mesh.nents(ent_dim), std::move(functor));
  return ents2nodes;
}

static void count_ent_nodes2ents(Omega_h::Mesh& mesh, int node_dim,
    int ent_dim, Omega_h::LOs nodes, Omega_h::Write<Omega_h::LO> counts) {
  auto node_ents2ents = mesh.ask_up(node_dim, ent_dim);
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO ent) {
    auto node = nodes[ent];
    auto nadj_ents = node_ents2ents.a2ab[ent + 1] - node_ents2ents.a2ab[ent];
    counts[node] = nadj_ents;
  };
  Omega_h::parallel_for("count ent_nodes 2 ents",
      mesh.nents(node_dim), std::move(functor));
}

static Omega_h::LOs count_nodes2ents(Omega_h::Mesh& mesh, int ent_dim,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto nnodes = nodes[0].size() + nodes[1].size();
  OMEGA_H_CHECK(nnodes == (mesh.nverts() + mesh.nedges()));
  Omega_h::Write<Omega_h::LO> node_counts(nnodes, "node_counts");
  count_ent_nodes2ents(mesh, 0, ent_dim, nodes[0], node_counts);
  count_ent_nodes2ents(mesh, 1, ent_dim, nodes[1], node_counts);
  return node_counts;
}

static void build_ent_nodes2ents(Omega_h::Mesh& mesh, int node_dim,
    int ent_dim, Omega_h::LOs nodes, Omega_h::LOs a2ab,
    Omega_h::Write<Omega_h::LO> ab2b, Omega_h::Write<Omega_h::I8> codes) {
  auto node_ents2ents = mesh.ask_up(node_dim, ent_dim);
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO ent) {
    auto node = nodes[ent];
    auto ab = a2ab[node];
    auto adj_ent_begin = node_ents2ents.a2ab[ent];
    auto adj_ent_end = node_ents2ents.a2ab[ent + 1];
    for (int idx = adj_ent_begin; idx < adj_ent_end; ++idx) {
      auto adj_ent = node_ents2ents.ab2b[idx];
      auto adj_code = node_ents2ents.codes[idx];
      ab2b[ab] = adj_ent;
      codes[ab] = adj_code;
      ab++;
    }
  };
  Omega_h::parallel_for("build ent_nodes 2 ents",
      mesh.nents(node_dim), std::move(functor));
}

Omega_h::Adj build_p2_nodes2ents(Omega_h::Mesh& mesh, int ent_dim,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto counts = count_nodes2ents(mesh, ent_dim, nodes);
  auto a2ab = Omega_h::offset_scan(counts, "nodes2ents_a2ab");
  Omega_h::Write<Omega_h::LO> ab2b(a2ab.last(), "nodes2ents_ab2b");
  Omega_h::Write<Omega_h::I8> codes(a2ab.last(), "nodes2ents_codes");
  build_ent_nodes2ents(mesh, 0, ent_dim, nodes[0], a2ab, ab2b, codes);
  build_ent_nodes2ents(mesh, 1, ent_dim, nodes[1], a2ab, ab2b, codes);
  return Omega_h::Adj{a2ab, ab2b, codes};
}

static void build_ent_node_coords(Omega_h::Mesh& mesh, int node_dim,
    Omega_h::LOs nodes, Omega_h::Reals ent_coords,
    Omega_h::Write<Omega_h::Real> coords) {
  auto elem_dim = mesh.dim();
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO ent) {
    auto node = nodes[ent];
    for (int dim = 0; dim < elem_dim; ++dim) {
      auto c = ent_coords[ent * elem_dim + dim];
      coords[node * elem_dim + dim] = c;
    }
  };
  Omega_h::parallel_for("build ent_node coords",
      mesh.nents(node_dim), std::move(functor));
}

Omega_h::Reals build_p2_node_coords(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes) {
  auto elem_dim = mesh.dim();
  auto vtx_coords = mesh.coords();
  auto edge_coords = Omega_h::average_field(&mesh, 1, elem_dim, vtx_coords);
  auto nnodes = nodes[0].size() + nodes[1].size();
  OMEGA_H_CHECK(nnodes == (mesh.nverts() + mesh.nedges()));
  Omega_h::Write<Omega_h::Real> node_coords(nnodes * elem_dim, "node_coords");
  build_ent_node_coords(mesh, 0, nodes[0], vtx_coords, node_coords);
  build_ent_node_coords(mesh, 1, nodes[1], edge_coords, node_coords);
  return node_coords;
}

}
