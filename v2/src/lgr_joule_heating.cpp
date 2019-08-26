#include <Omega_h_align.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_simplex.hpp>
#include <lgr_for.hpp>
#include <lgr_joule_heating.hpp>
#include <lgr_linear_algebra.hpp>
#include <lgr_simulation.hpp>
#include <lgr_circuit.hpp>
#include <lgr_scalar.hpp>
//#include <iostream>

namespace lgr {

template <class Elem>
struct JouleHeating : public Model<Elem> {
  using Model<Elem>::sim;
  FieldIndex conductivity;
  FieldIndex normalized_voltage;
  FieldIndex conductance;
  FieldIndex specific_internal_energy_rate;
  GlobalMatrix matrix;
  GlobalVector rhs;
  Subset* anode_subset;
  Subset* cathode_subset;
  double normalized_anode_voltage;
  double normalized_cathode_voltage;
  double conductance_multiplier;
  double relative_tolerance;
  double absolute_tolerance;
  double anode_voltage;
  double cathode_voltage;
  double integrated_conductance;
  int cg_iterations;
  JouleHeating(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->conductivity =
        this->point_define("sigma", "conductivity", 1, RemapType::PER_UNIT_MASS, pl, "");
    this->normalized_voltage = sim.fields.define("phi", "normalized voltage", 1,
        NODES, false, sim.disc.covering_class_names());
    sim.fields[this->normalized_voltage].remap_type = RemapType::NODAL;
    sim.fields[this->normalized_voltage].default_value =
        pl.get<std::string>("normalized voltage", "0.0");
    this->conductance =
        this->point_define("G", "conductance", 1, RemapType::NONE, "");
    this->specific_internal_energy_rate = this->point_define(
        "e_dot", "specific internal energy rate", 1, RemapType::NONE, "0.0");
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
    normalized_anode_voltage =
        pl.get<double>("normalized anode voltage", "1.0");
    normalized_cathode_voltage =
        pl.get<double>("normalized cathode voltage", "0.0");
    relative_tolerance = pl.get<double>("relative tolerance", "1.0e-6");
    absolute_tolerance = pl.get<double>("absolute tolerance", "1.0e-10");
    conductance_multiplier = pl.get<double>("conductance multiplier", "1.0");
    anode_voltage = pl.get<double>("anode voltage", "1.0");
    cathode_voltage = pl.get<double>("cathode voltage", "0.0");
    cg_iterations = pl.get<int>("cg iterations", "0");
    JouleHeating::learn_disc();
  }
  void learn_disc() override final {
    // linear specific!
    auto const verts_to_other_verts = sim.disc.mesh.ask_star(0);
    auto const verts_to_selves =
        Omega_h::identity_graph(sim.disc.mesh.nverts());
    auto const verts_to_verts =
        Omega_h::add_edges(verts_to_selves, verts_to_other_verts);
    matrix.rows_to_columns = verts_to_verts;
    auto const nnz = verts_to_verts.a2ab.last();
    matrix.entries = Omega_h::Write<double>(nnz, "conductance matrix entries");
  }
  std::uint64_t exec_stages() override final { return AT_SECONDARIES; }
  char const* name() override final { return "electrostatic"; }
  void at_secondaries() override final {
    Omega_h::ScopedTimer timer("JouleHeating::at_secondaries");
    assemble_normalized_voltage_system();
    solve_normalized_voltage_system();
    compute_conductance();
    integrate_conductance();
    compute_electrode_voltages();
    contribute_joule_heating();
    if (this->sim.circuit.usingMesh) {
       auto dt   = this->sim.dt;
       auto time = this->sim.time;
       this->sim.circuit.Solve(dt,time);
    }
  }
  void assemble_normalized_voltage_system() {
//  std::cerr << "assembling normalized voltage system\n";
    OMEGA_H_TIME_FUNCTION;
    constexpr int edges_per_elem = Omega_h::simplex_degree(Elem::dim, 1);
    constexpr int verts_per_elem = Omega_h::simplex_degree(Elem::dim, 0);
    Omega_h::Write<double> elems_to_vert_contribs(
        sim.disc.mesh.nelems() * verts_per_elem);
    Omega_h::Write<double> elems_to_edge_contribs(
        sim.disc.mesh.nelems() * edges_per_elem);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_conductivity = this->points_get(this->conductivity);
    auto const points_to_weight = sim.set(sim.weight);
    auto elem_functor = OMEGA_H_LAMBDA(int const elem) {
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const weight = points_to_weight[point];
        auto const G = points_to_conductivity[point];
        auto const grads = getgrads<Elem>(points_to_grad, point);
        for (int elem_vert = 0; elem_vert < verts_per_elem; ++elem_vert) {
          auto const contrib =
              weight * G * (grads[elem_vert] * grads[elem_vert]);
          elems_to_vert_contribs[elem * verts_per_elem + elem_vert] = contrib;
        }
        for (int elem_edge = 0; elem_edge < edges_per_elem; ++elem_edge) {
          auto const elem_vert0 =
              Omega_h::simplex_down_template(Elem::dim, 1, elem_edge, 0);
          auto const elem_vert1 =
              Omega_h::simplex_down_template(Elem::dim, 1, elem_edge, 1);
          auto const contrib =
              weight * G * (grads[elem_vert0] * grads[elem_vert1]);
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
    auto const A = this->matrix;
    auto const verts_to_edges = sim.disc.mesh.ask_up(0, 1);
    auto row_functor = OMEGA_H_LAMBDA(int const row) {
      auto const row_begin = A.rows_to_columns.a2ab[row];
      auto row_col = row_begin;
      A.entries[row_col++] = verts_to_value[row];
      auto const edge_begin = verts_to_edges.a2ab[row];
      auto const edge_end = verts_to_edges.a2ab[row + 1];
      for (auto vert_edge = edge_begin; vert_edge < edge_end; ++vert_edge) {
        auto const edge = verts_to_edges.ab2b[vert_edge];
        A.entries[row_col++] = edges_to_value[edge];
      }
    };
    parallel_for(sim.disc.mesh.nverts(), std::move(row_functor));
    auto const nnodes = sim.disc.mesh.nverts();
    auto const nodes_to_phi = sim.getset(this->normalized_voltage);
    rhs = Omega_h::Write<double>(nnodes, 0.0);
    {
      auto const anode_nodes_to_nodes = anode_subset->mapping.things;
      Omega_h::map_value_into(
          normalized_anode_voltage, anode_nodes_to_nodes, nodes_to_phi);
      OMEGA_H_CHECK(anode_nodes_to_nodes.size() != 0);
      auto const nodes_to_anode_nodes =
          sim.subsets.acquire_inverse(anode_nodes_to_nodes, nnodes);
      set_boundary_conditions(A, nodes_to_phi, rhs, nodes_to_anode_nodes);
      sim.subsets.release_inverse(anode_nodes_to_nodes);
    }
    {
      auto const cathode_nodes_to_nodes = cathode_subset->mapping.things;
      Omega_h::map_value_into(
          normalized_cathode_voltage, cathode_nodes_to_nodes, nodes_to_phi);
      OMEGA_H_CHECK(cathode_nodes_to_nodes.size() != 0);
      auto const nodes_to_cathode_nodes =
          sim.subsets.acquire_inverse(cathode_nodes_to_nodes, nnodes);
      set_boundary_conditions(A, nodes_to_phi, rhs, nodes_to_cathode_nodes);
      sim.subsets.release_inverse(cathode_nodes_to_nodes);
    }
  }
  void solve_normalized_voltage_system() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_phi = sim.getset(this->normalized_voltage);
    auto const cg_it = (sim.step == 0) ? nodes_to_phi.size() : cg_iterations;
    double relative_tol, absolute_tol;
    auto const niter = diagonal_preconditioned_conjugate_gradient(
        matrix, rhs, nodes_to_phi, relative_tolerance, absolute_tolerance, cg_it,
        relative_tol, absolute_tol);
    sim.circuit.jh_rel_tolerance = relative_tol;
    sim.circuit.jh_abs_tolerance = absolute_tol;
    sim.circuit.jh_iterations = niter;
    OMEGA_H_CHECK(niter <= nodes_to_phi.size());
//  std::cout << "phi solve took " << niter << " iterations\n";
  }
  void compute_conductance() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_phi = sim.get(this->normalized_voltage);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_conductivity = this->points_get(this->conductivity);
    auto const points_to_weight = sim.set(sim.weight);
    auto const points_to_G = this->points_set(this->conductance);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point / Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const phi = getscals<Elem>(nodes_to_phi, elem_nodes);
      auto const weight = points_to_weight[point];
      auto const G = points_to_conductivity[point];
      auto const grads = getgrads<Elem>(points_to_grad, point);
      auto const grad_phi = grad<Elem>(grads, phi);
      auto const integral = weight * G * (grad_phi * grad_phi);
      points_to_G[point] = integral;
    };
    parallel_for(this->points(), std::move(functor));
  }
  void integrate_conductance() {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_G = sim.get(this->conductance);
    integrated_conductance = repro_sum(points_to_G);
  }
  void compute_electrode_voltages() {
    OMEGA_H_TIME_FUNCTION;
    if (this->sim.circuit.usingMesh) {
       auto& myCircuit = this->sim.circuit;
       auto mesh_conductance = conductance_multiplier*
                               integrated_conductance;
       myCircuit.SetMeshConductance(mesh_conductance);
       // Below: 
       //   overwrites default/specified values in YAML 
       //   modifiers for joule heating when using circuit
       anode_voltage = myCircuit.GetMeshAnodeVoltage();
       cathode_voltage = myCircuit.GetMeshCathodeVoltage();
     }
  }
  void contribute_joule_heating() {
    OMEGA_H_TIME_FUNCTION;
    auto const V = anode_voltage - cathode_voltage;
    auto const Vsq = square(V);
    auto const points_to_G = this->points_get(this->conductance);
    auto const points_to_rho = this->points_get(sim.density);
    auto const points_to_volume = this->points_get(sim.weight);
    auto const points_to_e_dot = this->points_set(this->specific_internal_energy_rate);
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const G = points_to_G[point];
      auto const P = Vsq * G;
      auto const rho = points_to_rho[point];
      auto const volume = points_to_volume[point];
      auto const mass = rho * volume;
      auto const e_dot = P / mass;
      points_to_e_dot[point] += e_dot;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

void setup_joule_heating(Simulation& sim, Omega_h::InputMap& pl) {
  auto& models_pl = pl.get_list("modifiers");
  for (int i = 0; i < models_pl.size(); ++i) {
    auto& model_pl = models_pl.get_map(i);
    if (model_pl.get<std::string>("type") == "Joule heating") {
#define LGR_EXPL_INST(Elem) \
      if (sim.elem_name == Elem::name()) { \
        sim.models.add(new JouleHeating<Elem>(sim, model_pl)); \
      }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    }
  }
}

}  // namespace lgr
