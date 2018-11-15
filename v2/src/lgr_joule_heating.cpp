#include <lgr_electrostatic.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <Omega_h_array_ops.hpp>
#include <lgr_linear_algebra.hpp>

namespace lgr {

template <class Elem>
struct JouleHeating : public Model<Elem> {
  FieldIndex conductivity;
  FieldIndex normalized_voltage;
  FieldIndex conductance;
  GlobalMatrix matrix;
  GlobalVector rhs;
  Subset* anode_subset;
  Subset* cathode_subset;
  double normalized_anode_voltage;
  double normalized_cathode_voltage;
  double conductance_multiplier;
  double tolerance;
  double anode_voltage;
  double cathode_voltage;
  double integrated_conductance;
  JouleHeating(Simulation& sim_in, Omega_h::InputMap& pl):Model<Elem>(sim_in, pl) {
    this->conductivity =
      this->point_define("sigma", "conductivity", 1,
          RemapType::NONE, pl, "");
    this->normalized_voltage =
      sim.fields.define("phi", "normalized voltage",
          1, NODES, false, sim.disc.covering_class_names());
    sim.fields[this->normalized].remap_type = RemapType::NODAL;
    this->conductance =
      sim.fields.define("G", "conductance",
          1, NODES, false, sim.disc.covering_class_names());
    auto& anode_pl = pl.get_list("anode");
    ClassNames anode_class_names;
    for (int i = 0; i < anode_pl.size(); ++i) {
      anode_class_names.insert(anode_pl.get<std::string>(i));
    }
    anode_subset = sim.subsets.get_subset(NODES, anode_class_names);
    auto& cathode_pl = pl.get_list("cathode");
    ClassNames cathode_class_names;
    for (int i = 0; i < cathode_pl.size(); ++i) {
      cathode_class_names.insert(cathode_pl.get<std::string>(i));
    }
    cathode_subset = sim.subsets.get_subset(NODES, cathode_class_names);
    normalized_anode_voltage = pl.get<double>("normalized anode voltage", "1.0");
    normalized_cathode_voltage = pl.get<double>("normalized cathode voltage", "0.0");
    tolerance = pl.get<double>("tolerance", "1.0e-6");
    conductance_multiplier = pl.get<double>("conductance multiplier", "1.0");
  }
  void learn_disc() override final {
    // linear specific!
    auto const verts_to_other_verts = sim.disc.mesh.ask_star(0);
    auto const verts_to_selves = Omega_h::identity_graph(sim.disc.mesh.nverts());
    auto const verts_to_verts = Omega_h::add_edges(verts_to_selves, verts_to_other_verts);
    matrix.rows_to_columns = verts_to_verts;
    auto const nnz = verts_to_verts.a2ab.last();
    matrix.entries = Omega_h::Write<double>(nnz, "conductance matrix entries");
  }
  std::uint64_t exec_stages() override final { return BEFORE_POSITION_UPDATE | AFTER_CORRECTION; }
  char const* name() override final { return "electrostatic"; }
  void assemble_normalized_voltage_system() {
    OMEGA_H_TIME_FUNCTION;
    constexpr int edges_per_elem = Omega_h::simplex_degree(Elem::dim, 1);
    constexpr int verts_per_elem = Omega_h::simplex_degree(Elem::dim, 0);
    Omega_h::Write<double> elems_to_vert_contribs(sim.disc.mesh.nelems() * verts_per_elem);
    Omega_h::Write<double> elems_to_edge_contribs(sim.disc.mesh.nelems() * edges_per_elem);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_conductivity = this->points_get(this->conductivity);
    auto const points_to_weight = sim.set(sim.weight);
    auto elem_functor = OMEGA_H_LAMBDA(int const elem) {
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const weight = points_to_weight[point];
        auto const conductivity = points_to_conductivity[point];
        auto const grads = getgrads<Elem>(points_to_grad, point);
        for (int elem_vert = 0; elem_vert < verts_per_elem; ++elem_vert) {
          auto const contrib = weight * conductivity * (grads[elem_vert] * grads[elem_vert]);
          elems_to_vert_contribs[elem * verts_per_elem + elem_vert] = contrib;
        }
        for (int elem_edge = 0; elem_edge < verts_per_elem; ++elem_edge) {
          auto const elem_vert0 = Omega_h::simplex_down_template(Elem::dim, 1, elem_edge, 0);
          auto const elem_vert1 = Omega_h::simplex_down_template(Elem::dim, 1, elem_edge, 1);
          auto const contrib = weight * conductivity * (grads[elem_vert0] * grads[elem_vert1]);
          elems_to_edge_contribs[elem * edges_per_elem + elem_edge] = contrib;
        }
      }
    };
    parallel_for(sim.disc.mesh.nelems(), std::move(elem_functor));
    Omega_h::Write<double> edges_to_value(sim.disc.mesh.nedges());
    auto const edges_to_elems = sim.disc.mesh.ask_up(1, Elem::dim);
    auto edge_functor = OMEGA_H_LAMBDA(int const edge) {
      auto const begin = edges_to_elems.a2ab[edge];
      auto const end = edges_to_elems.a2ab[edge];
      double edge_value = 0.0;
      for (auto edge_elem = begin; edge_elem < end; ++edge_elem) {
        auto const elem = edges_to_elems.ab2b[edge_elem];
        auto const code = edges_to_elems.codes[edge_elem];
        auto const elem_edge = Omega_h::code_which_down(code);
        auto const contrib = elems_to_edge_contribs[elem * edges_per_elem + elem_edge];
        edge_value += contrib;
      }
      edges_to_value[edge] = edge_value;
    };
    parallel_for(sim.disc.mesh.nedges(), std::move(edge_functor));
    Omega_h::Write<double> verts_to_value(sim.disc.mesh.nverts());
    auto const verts_to_elems = sim.disc.mesh.ask_up(0, Elem::dim);
    auto vert_functor = OMEGA_H_LAMBDA(int const vert) {
      auto const begin = verts_to_elems.a2ab[vert];
      auto const end = verts_to_elems.a2ab[vert];
      double vert_value = 0.0;
      for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
        auto const elem = verts_to_elems.ab2b[vert_elem];
        auto const code = verts_to_elems.codes[vert_elem];
        auto const elem_vert = Omega_h::code_which_down(code);
        auto const contrib = elems_to_vert_contribs[elem * verts_per_elem + elem_vert];
        vert_value += contrib;
      }
      verts_to_value[vert] = vert_value;
    };
    parallel_for(sim.disc.mesh.nverts(), std::move(vert_functor));
    auto const A = this->matrix;
    auto const verts_to_edges = sim.disc.mesh.ask_up(0, 1);
    auto row_functor = OMEGA_H_LAMBDA(int const row) {
      auto const row_begin = A.rows_to_columns.a2ab[row];
      auto const row_col = row_begin;
      A.entries[row_col] = verts_to_value[row];
      ++row_col;
      auto const edge_begin = verts_to_edges.a2ab[row];
      auto const edge_end = verts_to_edges.a2ab[row + 1];
      for (auto const vert_edge = edge_begin; vert_edge < edge_end; ++vert_edge, ++row_col) {
        auto const edge = verts_to_edges.ab2b[vert_edge];
        A.entries[row_col] = edges_to_value[edge];
      }
    };
    parallel_for(sim.disc.mesh.nverts(), std::move(row_functor));
    auto const nnodes = sim.disc.mesh.nverts();
    auto const nodes_to_phi = sim.getset(this->normalized_voltage);
    rhs = Omega_h::Write<double>(nnodes, 0.0);
    {
    auto const anode_nodes_to_nodes = anode_subset->mapping.things;
    Omega_h::map_value_into(normalized_anode_voltage, anode_nodes_to_nodes, nodes_to_phi);
    OMEGA_H_CHECK(anode_nodes_to_nodes.size() != 0);
    auto const nodes_to_anode_nodes = sim.subsets.acquire_inverse(anode_nodes_to_nodes, nnodes);
    set_boundary_conditions(A, nodes_to_phi, rhs, nodes_to_anode_nodes); 
    sim.subsets.release_inverse(anode_nodes_to_nodes);
    }
    {
    auto const cathode_nodes_to_nodes = cathode_subset->mapping.things;
    Omega_h::map_value_into(normalized_cathode_voltage, cathode_nodes_to_nodes, nodes_to_phi);
    OMEGA_H_CHECK(cathode_nodes_to_nodes.size() != 0);
    auto const nodes_to_cathode_nodes = sim.subsets.acquire_inverse(cathode_nodes_to_nodes, nnodes);
    set_boundary_conditions(A, nodes_to_phi, rhs, nodes_to_cathode_nodes); 
    sim.subsets.release_inverse(cathode_nodes_to_nodes);
    }
  }
  void solve_normalized_voltage_system() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_phi = sim.getset(this->normalized_voltage);
    auto const niter = conjugate_gradient(matrix, rhs, nodes_to_phi, tolerance);
    OMEGA_H_CHECK(niter <= nodes_to_phi.size());
    std::cout << "phi solve took " << niter << " iterations\n";
  }
  void compute_conductance() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_phi = sim.get(this->normalized_voltage);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_conductivity = this->points_get(this->conductivity);
    auto const points_to_weight = sim.set(sim.weight);
    auto const points_to_G = this->points_set(this->conductance);
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point / Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const phi = getscals<Elem>(nodes_to_phi, elem_nodes);
      auto const weight = points_to_weight[point];
      auto const conductivity = points_to_conductivity[point];
      auto const grads = getgrads<Elem>(points_to_grad, point);
      auto const grad_phi = grad<Elem>(grads, phi);
      auto const integral += weight * conductivity * (grad_phi * grad_phi);
      points_to_conductance[point] = integral;
    };
    parallel_for(points(), std::move(functor));
  }
  void integrate_conductance() {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_G = this->points_get(this->conductance);
    integrated_conductance = repro_sum(read(points_to_G));
  }
  void compute_electrode_voltages() {
    OMEGA_H_TIME_FUNCTION;
    // TODO: put a circuit here!
    anode_voltage = 1.0;
    cathode_voltage = 0.0;
  }
  void compute_joule_heating() {
    OMEGA_H_TIME_FUNCTION;
    auto const voltage_difference = anode_voltage - cathode_voltage;
  }
};

template <class Elem>
ModelBase* joule_heating_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new JouleHeating<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* joule_heating_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

