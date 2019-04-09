#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_expr.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_int_iterator.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_reduce.hpp>
#include <fstream>
#include <lgr_config.hpp>
#include <lgr_disc.hpp>
#include <lgr_quadratic.hpp>
#include <limits>
#include <sstream>

namespace lgr {

template <int dim>
static void mark_closest_vertex_dim(Omega_h::Mesh& mesh,
    std::string const& set_name, Omega_h::any const& pos_any) {
  auto const pos = Omega_h::any_cast<Vector<dim>>(pos_any);
  auto const coords = mesh.coords();
  auto min_dist_functor = OMEGA_H_LAMBDA(int vertex)->double {
    auto vertex_pos = Omega_h::get_vector<dim>(coords, vertex);
    return Omega_h::norm(vertex_pos - pos);
  };
  double const min_dist = Omega_h::transform_reduce(Omega_h::IntIterator(0),
      Omega_h::IntIterator(mesh.nverts()), std::numeric_limits<double>::max(),
      Omega_h::minimum<double>(), std::move(min_dist_functor));
  auto globals = mesh.globals(Omega_h::VERT);
  // tiebreaker pass
  auto min_global_functor = OMEGA_H_LAMBDA(int vertex)->std::int64_t {
    auto vertex_pos = Omega_h::get_vector<dim>(coords, vertex);
    auto dist = Omega_h::norm(vertex_pos - pos);
    if (dist == min_dist)
      return globals[vertex];
    else
      return Omega_h::ArithTraits<std::int64_t>::max();
  };
  auto const min_global = Omega_h::transform_reduce(Omega_h::IntIterator(0),
      Omega_h::IntIterator(mesh.nverts()),
      std::numeric_limits<std::int64_t>::max(),
      Omega_h::minimum<std::int64_t>(), std::move(min_global_functor));
  auto class_ids = Omega_h::deep_copy(
      mesh.get_array<Omega_h::ClassId>(Omega_h::VERT, "class_id"));
  auto class_dims = Omega_h::deep_copy(
      mesh.get_array<std::int8_t>(Omega_h::VERT, "class_dim"));
  auto max_class_id_functor = OMEGA_H_LAMBDA(int vertex)->Omega_h::ClassId {
    if (class_dims[vertex] == std::int8_t(0))
      return class_ids[vertex];
    else
      return Omega_h::ClassId(-1);
  };
  auto const max_class_id = Omega_h::transform_reduce(Omega_h::IntIterator(0),
      Omega_h::IntIterator(mesh.nverts()), Omega_h::ClassId(-1),
      Omega_h::maximum<Omega_h::ClassId>(), std::move(max_class_id_functor));
  auto const new_class_id = max_class_id + 1;
  auto set_functor = OMEGA_H_LAMBDA(int vertex) {
    if (globals[vertex] == min_global) {
      class_ids[vertex] = new_class_id;
      class_dims[vertex] = std::int8_t(0);
    }
  };
  Omega_h::parallel_for(mesh.nverts(), std::move(set_functor));
  mesh.set_tag(Omega_h::VERT, "class_id", Omega_h::read(class_ids));
  mesh.set_tag(Omega_h::VERT, "class_dim", Omega_h::read(class_dims));
  mesh.class_sets[set_name].push_back({std::int8_t(0), new_class_id});
}

static void mark_closest_vertex(Omega_h::Mesh& mesh,
    std::string const& set_name, std::string const& pos_expr) {
  Omega_h::ExprOpsReader reader;
  auto const pos_op = reader.read_ops(pos_expr);
  Omega_h::ExprEnv pos_env(1, mesh.dim());
  if (mesh.dim() == 1)
    mark_closest_vertex_dim<1>(mesh, set_name, pos_op->eval(pos_env));
  else if (mesh.dim() == 2)
    mark_closest_vertex_dim<2>(mesh, set_name, pos_op->eval(pos_env));
  else
    mark_closest_vertex_dim<3>(mesh, set_name, pos_op->eval(pos_env));
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

void Disc::setup(Omega_h::CommPtr comm, Omega_h::InputMap& pl) {
  if (pl.is<std::string>("file")) {
    mesh = Omega_h::read_mesh_file(pl.get<std::string>("file"), comm);
  } else if (pl.is_map("box")) {
    auto& box_pl = pl.get_map("box");
    int x_elements = box_pl.get<int>("x elements");
    int y_elements = box_pl.get<int>("y elements", "0");
    int z_elements = box_pl.get<int>("z elements", "0");
    double x_size = box_pl.get<double>("x size", "1.0");
    double y_size = box_pl.get<double>("y size", "1.0");
    double z_size = box_pl.get<double>("z size", "1.0");
    bool symmetric = box_pl.get<bool>("symmetric", "false");
    mesh = Omega_h::build_box(comm,
        (is_simplex_ ? OMEGA_H_SIMPLEX : OMEGA_H_HYPERCUBE), x_size, y_size,
        z_size, x_elements, y_elements, z_elements, symmetric);
  } else if (pl.is_map("CUBIT")) {
#ifdef LGR_USE_CUBIT
    auto& cubit_pl = pl.get_map("CUBIT");
    auto cubit_path = LGR_CUBIT;
    Omega_h::filesystem::path journal_path;
    if (cubit_pl.is<std::string>("commands")) {
      auto commands = cubit_pl.get<std::string>("commands");
      journal_path =
          cubit_pl.get<std::string>("journal file", "lgr_temporary.jou");
      if (comm->rank() == 0) {
        std::ofstream journal_stream(journal_path.c_str());
        journal_stream << commands << '\n';
        journal_stream.close();
      }
    } else {
      journal_path = cubit_pl.get<std::string>("journal file");
    }
    OMEGA_H_CHECK(journal_path.extension().string() == ".jou");
    auto default_exodus_path = journal_path.parent_path();
    default_exodus_path /= journal_path.stem();
    default_exodus_path += ".exo";
    Omega_h::filesystem::path exodus_path =
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
    if (!cubit_pl.get<bool>("keep files", "false")) {
      if (comm->rank() == 0) {
        Omega_h::filesystem::remove(journal_path);
        Omega_h::filesystem::remove(exodus_path);
      }
    }
#else
    Omega_h_fail(
        "CUBIT mesh requested but LGRTK not compiled with CUBIT support!\n");
#endif
  } else {
    Omega_h_fail("no input mesh!\n");
  }
  if (!(mesh.dim() == dim_)) {
    Omega_h_fail("element dimension %d doesn't match mesh dimension %d\n", dim_,
        mesh.dim());
  }
  OMEGA_H_CHECK(
      mesh.family() == (is_simplex_ ? OMEGA_H_SIMPLEX : OMEGA_H_HYPERCUBE));
  if (pl.is_map("sets")) {
    Omega_h::update_class_sets(&mesh.class_sets, pl.get_map("sets"));
  }
  if (pl.is<std::string>("transform")) {
    Omega_h::ExprReader reader(mesh.nverts(), mesh.dim());
    reader.register_variable("x", Omega_h::any(mesh.coords()));
    auto result =
        reader.read_string(pl.get<std::string>("transform"), "mesh transform");
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
  this->is_second_order_ = pl.get<bool>("add mid edge nodes", "false");
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
  this->update_from_mesh();
}

void Disc::update_from_mesh() {
  if (is_second_order_) {
    OMEGA_H_CHECK(is_simplex_);
    p2_nodes = number_p2_nodes(mesh);
    ents2nodes_[dim_] = build_p2_ents2nodes(mesh, dim_, p2_nodes);
    nodes2ents_[dim_] = build_p2_nodes2ents(mesh, dim_, p2_nodes);
    nodes2ents_[dim_ - 1] = build_p2_nodes2ents(mesh, dim_ - 1, p2_nodes);
    node_coords_ = build_p2_node_coords(mesh, p2_nodes);
  } else {
    ents2nodes_[dim_] = mesh.ask_elem_verts();
    nodes2ents_[dim_] = mesh.ask_up(0, mesh.dim());
    node_coords_ = mesh.coords();
  }
}

int Disc::dim() { return mesh.dim(); }

int Disc::count(EntityType type) {
  if (type == ELEMS) return mesh.nelems();
  if (type == NODES) return nodes2ents_[dim_].a2ab.size() - 1;
  OMEGA_H_NORETURN(-1);
}

Omega_h::LOs Disc::ents_to_nodes(EntityType type) {
  OMEGA_H_CHECK(type == ELEMS);
  return ents2nodes_[dim_];
}

Omega_h::Adj Disc::nodes_to_ents(EntityType type) {
  OMEGA_H_CHECK(type == ELEMS);
  return nodes2ents_[dim_];
}

Omega_h::LOs Disc::ents_on_closure(
    std::set<std::string> const& class_names, EntityType type) {
  if (class_names.empty()) return Omega_h::LOs({});
  int ent_dim = -1;
  if (type == NODES) {
    if (is_second_order_) {
      Omega_h::Graph nodes2ents[4];
      for (int i = 0; i < 4; ++i) nodes2ents[i] = nodes2ents_[i];
      return Omega_h::nodes_on_closure(&mesh, class_names, nodes2ents);
    } else {
      ent_dim = 0;
    }
  } else if (type == ELEMS)
    ent_dim = dim();
  else if (type == SIDES)
    ent_dim = dim() - 1;
  else
    Omega_h_fail("unsupported EntityType in Disc::ents_on_closure\n");
  return Omega_h::ents_on_closure(&mesh, class_names, ent_dim);
}

ClassNames const& Disc::covering_class_names() { return covering_class_names_; }

int Disc::points_per_ent(EntityType type) { return points_per_ent_[type]; }

int Disc::nodes_per_ent(EntityType type) { return nodes_per_ent_[type]; }

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

Omega_h::Reals Disc::get_node_coords() { return node_coords_; }
void Disc::set_node_coords(Omega_h::Reals input) { node_coords_ = input; }

#define LGR_EXPL_INST(Elem) template void Disc::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
