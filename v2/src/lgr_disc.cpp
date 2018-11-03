#include <lgr_disc.hpp>
#include <lgr_config.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_expr.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_reduce.hpp>
#include <Omega_h_int_iterator.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_align.hpp>
#include <Omega_h_int_scan.hpp>
#include <fstream>
#include <sstream>
#include <limits>

namespace lgr {

template <int dim>
static void mark_closest_vertex_dim(Omega_h::Mesh& mesh, std::string const& set_name, Omega_h::any const& pos_any) {
  auto const pos = Omega_h::any_cast<Vector<dim>>(pos_any);
  auto const coords = mesh.coords();
  auto min_dist_functor = OMEGA_H_LAMBDA(int vertex) -> double {
    auto vertex_pos = Omega_h::get_vector<dim>(coords, vertex);
    return Omega_h::norm(vertex_pos - pos);
  };
  double const min_dist = Omega_h::transform_reduce(
      Omega_h::IntIterator(0),
      Omega_h::IntIterator(mesh.nverts()),
      std::numeric_limits<double>::max(),
      Omega_h::minimum<double>(),
      std::move(min_dist_functor));
  auto globals = mesh.globals(Omega_h::VERT);
  // tiebreaker pass
  auto min_global_functor = OMEGA_H_LAMBDA(int vertex) -> std::int64_t {
    auto vertex_pos = Omega_h::get_vector<dim>(coords, vertex);
    auto dist = Omega_h::norm(vertex_pos - pos);
    if (dist == min_dist) return globals[vertex];
    else return Omega_h::ArithTraits<std::int64_t>::max();
  };
  auto const min_global = Omega_h::transform_reduce(
      Omega_h::IntIterator(0),
      Omega_h::IntIterator(mesh.nverts()),
      std::numeric_limits<std::int64_t>::max(),
      Omega_h::minimum<std::int64_t>(),
      std::move(min_global_functor));
  auto class_ids = Omega_h::deep_copy(mesh.get_array<Omega_h::ClassId>(Omega_h::VERT, "class_id"));
  auto class_dims = Omega_h::deep_copy(mesh.get_array<std::int8_t>(Omega_h::VERT, "class_dim"));
  auto max_class_id_functor = OMEGA_H_LAMBDA(int vertex) -> Omega_h::ClassId {
    if (class_dims[vertex] == std::int8_t(0)) return class_ids[vertex];
    else return Omega_h::ClassId(-1);
  };
  auto const max_class_id = Omega_h::transform_reduce(
      Omega_h::IntIterator(0),
      Omega_h::IntIterator(mesh.nverts()),
      Omega_h::ClassId(-1),
      Omega_h::maximum<Omega_h::ClassId>(),
      std::move(max_class_id_functor));
  auto const new_class_id = max_class_id + 1;
  auto set_functor = OMEGA_H_LAMBDA(int vertex) {
    if (globals[vertex] == min_global) {
      class_ids[vertex] = new_class_id;
      class_dims[vertex] = std::int8_t(0);
    }
  };
  Omega_h::parallel_for("set", mesh.nverts(), std::move(set_functor));
  mesh.set_tag(Omega_h::VERT, "class_id", Omega_h::read(class_ids));
  mesh.set_tag(Omega_h::VERT, "class_dim", Omega_h::read(class_dims));
  mesh.class_sets[set_name].push_back({std::int8_t(0), new_class_id});
}

static void mark_closest_vertex(Omega_h::Mesh& mesh, std::string const& set_name, std::string const& pos_expr) {
  Omega_h::ExprOpsReader reader;
  auto pos_op = reader.read_ops(pos_expr);
  Omega_h::ExprEnv pos_env(1, mesh.dim());
  if (mesh.dim() == 1) mark_closest_vertex_dim<1>(mesh, set_name, pos_op->eval(pos_env));
  if (mesh.dim() == 2) mark_closest_vertex_dim<2>(mesh, set_name, pos_op->eval(pos_env));
  return mark_closest_vertex_dim<3>(mesh, set_name, pos_op->eval(pos_env));
}

static void change_element_count(Omega_h::Mesh& mesh, double desired_nelems) {
  auto min_error = std::numeric_limits<double>::max();
  int min_error_i = 0;
  for (int i = 0; true; ++i) {
    auto current_nelems = double(mesh.nglobal_ents(mesh.dim()));
    auto new_error = std::abs(current_nelems - desired_nelems);
    if (new_error < min_error) {
      min_error = new_error;
      min_error_i = i;
    } else if (i - min_error_i > 5) {
      break;
    }
    mesh.set_parting(OMEGA_H_GHOSTED);
    if (!mesh.has_tag(0, "metric")) Omega_h::add_implied_isos_tag(&mesh);
    auto metrics = mesh.get_array<double>(0, "metric");
    auto scalar = Omega_h::get_metric_scalar_for_nelems(
        mesh.dim(), current_nelems, desired_nelems);
    metrics = Omega_h::multiply_each_by(metrics, scalar);
    mesh.add_tag(Omega_h::VERT, "target_metric", 1, metrics);
    auto opts = Omega_h::AdaptOpts(&mesh);
    opts.verbosity = Omega_h::SILENT;
    while (Omega_h::approach_metric(&mesh, opts)) {
      Omega_h::adapt(&mesh, opts);
    }
    Omega_h::adapt(&mesh, opts);
  }
}

static Omega_h::Read<int> count_nodes_by_vert(Omega_h::Mesh& mesh) {
  auto verts2edges = mesh.ask_up(0, 1);
  Omega_h::Write<int> counts(mesh.nverts(), "node_counts_by_vert");
  auto functor = OMEGA_H_LAMBDA(Omega_h::LO v) {
    int count = 1;
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

static Omega_h::Few<Omega_h::LOs, 2> number_p2_nodes(Omega_h::Mesh& mesh) {
  auto verts2edges = mesh.ask_up(0, 1);
  Omega_h::Write<Omega_h::LO> vtx_nodes(mesh.nverts(), "vtx_node_nmbr");
  Omega_h::Write<Omega_h::LO> edge_nodes(mesh.nedges(), "edge_node_nmbr");
  auto counts = count_nodes_by_vert(mesh);
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

void Disc::setup(Omega_h::CommPtr comm, Omega_h::InputMap& pl) {
  if (pl.is<std::string>("file")) {
    mesh = Omega_h::read_mesh_file(pl.get<std::string>("file"), comm);
  } else if (pl.is_map("box")) {
    auto& box_pl = pl.get_map("box");
    int x_elements = box_pl.get<int>("x elements");
    int y_elements = box_pl.get<int>("y elements", (dim_ > 1) ? "1" : "0");
    int z_elements = box_pl.get<int>("z elements", (dim_ > 2) ? "1" : "0");
    double x_size = box_pl.get<double>("x size", "1.0");
    double y_size = box_pl.get<double>("y size", "1.0");
    double z_size = box_pl.get<double>("z size", "1.0");
    bool symmetric = box_pl.get<bool>("symmetric", "false");
    mesh = Omega_h::build_box(comm, (is_simplex_ ? OMEGA_H_SIMPLEX : OMEGA_H_HYPERCUBE),
        x_size, y_size, z_size, x_elements, y_elements, z_elements, symmetric);
  } else if (pl.is_map("CUBIT")) {
#ifdef LGR_USE_CUBIT
    auto& cubit_pl = pl.get_map("CUBIT");
    auto cubit_path = LGR_CUBIT;
    std::string journal_path;
    if (cubit_pl.is<std::string>("commands")) {
      auto commands = cubit_pl.get<std::string>("commands");
      journal_path = cubit_pl.get<std::string>("journal file", "lgr_temporary.jou");
      if (comm->rank() == 0) {
        std::ofstream journal_stream(journal_path.c_str());
        journal_stream << commands << '\n';
        journal_stream.close();
      }
    } else {
      journal_path = cubit_pl.get<std::string>("journal file");
    }
    OMEGA_H_CHECK(Omega_h::ends_with(journal_path, ".jou"));
    auto default_exodus_path =
      journal_path.substr(0, journal_path.length() - 3) + "exo";
    auto exodus_path =
      cubit_pl.get<std::string>("Exodus file", default_exodus_path.c_str());
    std::stringstream system_stream;
    system_stream << cubit_path << ' ';
    system_stream << "-nobanner ";
    system_stream << "-batch ";
    system_stream << "-noecho ";
    system_stream << "-nographics ";
    system_stream << "-nojournal ";
    system_stream << "-nooverwritecheck ";
    system_stream << "-input " << journal_path << '\n';
    auto system_cmd = system_stream.str();
    if (comm->rank() == 0) {
      auto ret = std::system(system_cmd.c_str());
      if (ret != 0) Omega_h_fail("Running CUBIT failed!\n");
    }
    mesh = Omega_h::read_mesh_file(exodus_path, comm);
#else
    Omega_h_fail("CUBIT mesh requested but LGRTK not compiled with CUBIT support!\n");
#endif
  } else {
    Omega_h_fail("no input mesh!\n");
  }
  OMEGA_H_CHECK(mesh.dim() == dim_);
  OMEGA_H_CHECK(mesh.family() == (is_simplex_ ? OMEGA_H_SIMPLEX : OMEGA_H_HYPERCUBE));
  if (pl.is_map("sets")) {
    Omega_h::update_class_sets(&mesh.class_sets, pl.get_map("sets"));
  }
  if (pl.is<std::string>("transform")) {
    Omega_h::ExprReader reader(mesh.nverts(), mesh.dim());
    reader.register_variable("x", Omega_h::any(mesh.coords()));
    auto result = reader.read_string(pl.get<std::string>("transform"), "mesh transform");
    reader.repeat(result);
    mesh.set_coords(Omega_h::any_cast<Omega_h::Reals>(result));
  }
  if (pl.is<double>("element count")) {
    change_element_count(mesh, pl.get<double>("element count"));
  }
  if (pl.get<bool>("reorder", "false")) {
    Omega_h::reorder_by_hilbert(&mesh);
  }
  if (pl.is_list("mark closest nodes")) {
    auto& markings = pl.get_list("mark closest nodes");
    for (int i = 0; i < markings.size(); ++i) {
      auto& marking = markings.get_list(i);
      auto set_name = marking.get<std::string>(0);
      auto pos_expr = marking.get<std::string>(1);
      mark_closest_vertex(mesh, set_name, pos_expr);
    }
  }
  if (pl.get<bool>("add mid edge nodes", "false")) {
    auto nodes = number_p2_nodes(mesh);
  } else {
    elems2nodes_ = mesh.ask_elem_verts();
    nodes2elems_ = mesh.ask_up(0, mesh.dim());
  }
  std::set<int> volume_ids;
  for (auto& s : mesh.class_sets) {
    for (auto& cp : s.second) {
      if (cp.dim == dim()) {
        auto it = volume_ids.lower_bound(cp.id);
        if (it == volume_ids.end() || *it != cp.id) {
          covering_class_names_.insert(s.first);
          volume_ids.insert(it, cp.id);
        }
      }
    }
  }
}

int Disc::dim() { return mesh.dim(); }

int Disc::count(EntityType type) {
  if (type == ELEMS) return mesh.nelems();
  if (type == NODES) return mesh.nverts(); // linear specific!
  OMEGA_H_NORETURN(-1);
}

Omega_h::LOs Disc::ents_to_nodes(EntityType type) {
  OMEGA_H_CHECK(type == ELEMS);
  return elems2nodes_;
}

Omega_h::Adj Disc::nodes_to_ents(EntityType type) {
  OMEGA_H_CHECK(type == ELEMS);
  return nodes2elems_;
}

Omega_h::LOs Disc::ents_on_closure(
    std::set<std::string> const& class_names,
    EntityType type) {
  int ent_dim = -1;
  if (type == NODES) ent_dim = 0; // linear specific!
  else if (type == ELEMS) ent_dim = dim();
  return Omega_h::ents_on_closure(&mesh, class_names, ent_dim);
}

ClassNames const& Disc::covering_class_names() {
  return covering_class_names_;
}

int Disc::points_per_ent(EntityType type) {
  return points_per_ent_[type];
}

int Disc::nodes_per_ent(EntityType type) {
  return nodes_per_ent_[type];
}

template <class Elem>
void Disc::set_elem() {
  dim_ = Elem::dim;
  is_simplex_ = Elem::is_simplex;
  points_per_ent_[ELEMS] = Elem::points;
  points_per_ent_[SIDES] = Elem::side::points;
  points_per_ent_[EDGES] = -1;
  points_per_ent_[NODES] = -1;
  nodes_per_ent_[ELEMS] = Elem::nodes;
  nodes_per_ent_[SIDES] = Elem::side::nodes;
  nodes_per_ent_[EDGES] = -1;
  nodes_per_ent_[NODES] = 1;
}

Omega_h::Reals Disc::node_coords() {
  //linear specific!
  return mesh.coords();
}

#define LGR_EXPL_INST(Elem) \
template void Disc::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
