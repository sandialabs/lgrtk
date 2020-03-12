#include <Omega_h_align.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_simplex.hpp>
#include <Omega_h_fail.hpp>
#include <lgr_for.hpp>
#include <lgr_joule_heating.hpp>
#include <lgr_linear_algebra.hpp>
#include <lgr_simulation.hpp>
#include <lgr_circuit.hpp>
#include <lgr_model.hpp>
#include <lgr_global_fem.hpp>
#include <map>

namespace lgr {

template <class Elem>
struct JouleHeating : public Model<Elem> {
  using Model<Elem>::sim;
  FieldIndex conductivity;
  FieldIndex diffused_conductivity;
  FieldIndex nodal_conductivity;
  FieldIndex electric_field;
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
  bool constant_voltage_field;
  double z_thickness;
  double z_total;
  bool diffuse_conductivity;
  double diffusion_length;
  JouleHeating(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->conductivity =
        this->point_define("sigma", "conductivity", 1, RemapType::PER_UNIT_MASS, pl, "");
    this->diffused_conductivity =
        this->point_define("sigmad", "diffused conductivity", 1, RemapType::NONE, pl, "");
    this->normalized_voltage = sim.fields.define("phi", "normalized voltage", 1,
        NODES, false, sim.disc.covering_class_names());
    sim.fields[this->normalized_voltage].remap_type = RemapType::NODAL;
    sim.fields[this->normalized_voltage].default_value =
        pl.get<std::string>("normalized voltage", "0.0");
    this->nodal_conductivity = sim.fields.define("nodal_sigma", "nodal conductivity", 1,
        NODES, false, sim.disc.covering_class_names());
    sim.fields[this->nodal_conductivity].remap_type = RemapType::NODAL;
    sim.fields[this->nodal_conductivity].default_value =
        pl.get<std::string>("nodal conductivity", "0.0");

    this->electric_field = sim.fields.define("efield", "electric field", Elem::dim,
        NODES, false, sim.disc.covering_class_names());
    sim.fields[this->electric_field].remap_type = RemapType::NONE;
    sim.fields[this->electric_field].default_value =
        pl.get<std::string>("electric field", "");

//  this->electric_field =
//      this->point_define("efield", "electric field", Elem::dim, RemapType::NONE, pl, "");

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
    diffuse_conductivity = pl.get<bool>("diffuse conductivity","false");
    if (diffuse_conductivity) {
        diffusion_length = pl.get<double>("diffusion length");
    }
    constant_voltage_field = pl.get<bool>("constant voltage field","false");
    relative_tolerance = pl.get<double>("relative tolerance", "1.0e-6");
    absolute_tolerance = pl.get<double>("absolute tolerance", "1.0e-10");
    anode_voltage = pl.get<double>("anode voltage", "1.0");
    cathode_voltage = pl.get<double>("cathode voltage", "0.0");
    cg_iterations = pl.get<int>("iterations", "0");
    normalized_anode_voltage = sim.input_variables.get_double(pl,"normalized anode voltage","1.0");
    normalized_cathode_voltage = sim.input_variables.get_double(pl,"normalized cathode voltage","0.0");
    conductance_multiplier = sim.input_variables.get_double(pl,"conductance multiplier","1.0");
    JouleHeating::learn_disc();
    // Initially set global outputs for case where constant voltage field is used
    sim.globals.set("Joule heating relative tolerance", std::nan("1"));
    sim.globals.set("Joule heating absolute tolerance", std::nan("1"));
    sim.globals.set("Joule heating iterations", 0);
    sim.globals.set("mesh voltage", sim.circuit.GetMeshVoltageDrop());
    sim.globals.set("mesh current", sim.circuit.GetMeshCurrent());
    sim.globals.set("mesh conductance", sim.circuit.GetMeshConductance());
    // Also set voltage/current elements as a possible variable
    auto element_voltages = sim.circuit.element_voltages;
    auto element_currents = sim.circuit.element_currents;
    for (auto it = element_voltages.begin(); it != element_voltages.end(); ++it) {
        std::string name = "circuit_v" + std::to_string(it->first);
        double voltage = it->second;
        sim.globals.set(name, voltage);
    }
    for (auto it = element_currents.begin(); it != element_currents.end(); ++it) {
        std::string name = "circuit_i" + std::to_string(it->first);
        double current = it->second;
        sim.globals.set(name, current);
    }
    // 2D thickness
    z_thickness = sim.input_variables.get_double(pl,"z thickness","-1.0");
    if (z_thickness > 0) {
       if (Elem::dim != 2)
           Omega_h_fail("Only 2D simulations can use z thickness\n");
       // Divide: recover conductance integral with symmetry
       // Multiply: 3D to 2D volume integration
       // conductance_multiplier*=z_thickness/z_thickness;
    }
    z_total = sim.input_variables.get_double(pl,"z total","-1.0");
    if (z_total > 0) {
       if (Elem::dim != 2)
           Omega_h_fail("Only 2D simulations can use z total\n");
       // Multiply: recover conductance integral with symmetry
       conductance_multiplier*=z_total;
    }
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
    if (!this->constant_voltage_field) {
       if (this->diffuse_conductivity) {
          compute_diffused_conductivity();
       }
       assemble_normalized_voltage_system();
       solve_normalized_voltage_system();
    }
    compute_electric_field();
    compute_conductance();
    integrate_conductance();
    compute_electrode_voltages();
    contribute_joule_heating();
    if (this->sim.circuit.usingMesh) {
       auto dt   = this->sim.dt;
       auto time = this->sim.time;
       if (dt > 1e-14)
          this->sim.circuit.Solve(dt,time);
    }
    sim.globals.set("Joule heating CPU time", timer.total_runtime());
  }
  void compute_electric_field() {
    auto const nodes_to_phi = sim.get(this->normalized_voltage);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const verts_to_elems = sim.disc.mesh.ask_up(0, Elem::dim);
    auto const nodes_to_efield = sim.set(this->electric_field);
    auto v_functor = OMEGA_H_LAMBDA(int const vert) {
      auto const begin = verts_to_elems.a2ab[vert];
      auto const end = verts_to_elems.a2ab[vert + 1];
      int const e_count = end-begin;
      Omega_h::Vector<Elem::dim> g_count;
      for (int i=0; i < Elem::dim; ++i) {
         g_count[i] = 0.0;
      }
      for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
        auto const elem = verts_to_elems.ab2b[vert_elem];
        auto const point = elem;
        auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
        auto const phi = getscals<Elem>(nodes_to_phi, elem_nodes);
        auto const grads = getgrads<Elem>(points_to_grad, point);
        auto const grad_phi = grad<Elem>(grads, phi);
        for (int i=0; i < Elem::dim; ++i) {
           g_count[i] += -grad_phi[i]/e_count;
        }
      }
      setvec<Elem>(nodes_to_efield, vert, g_count);
    };
    parallel_for(sim.disc.mesh.nverts(), std::move(v_functor));
  }
  void compute_diffused_conductivity() {
    OMEGA_H_TIME_FUNCTION;
    // Compute nodal conductivity
    auto const points_to_conductivity = this->points_get(this->conductivity);
    Omega_h::Write<double> nodes_to_initial_nodal_conductivity(sim.disc.mesh.nverts());

    auto const verts_to_elems = sim.disc.mesh.ask_up(0, Elem::dim);
    auto v_functor = OMEGA_H_LAMBDA(int const vert) {
      auto const begin = verts_to_elems.a2ab[vert];
      auto const end = verts_to_elems.a2ab[vert + 1];
      int const e_count = end-begin;
      double g_count = 0.0;
      for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
        auto const elem = verts_to_elems.ab2b[vert_elem];
        auto const point = elem;
        g_count += points_to_conductivity[point];
      }
      nodes_to_initial_nodal_conductivity[vert] = g_count/e_count;
    };
    parallel_for(sim.disc.mesh.nverts(), std::move(v_functor));

    // Setup diffusion number for each element
    auto const lscale = this->diffusion_length;
    auto const nsteps = 1;
    auto const dtau = 1.0;
    auto const dnumber = lscale*lscale/(4.0*dtau*nsteps);
    Omega_h::Write<double> diffno(sim.disc.mesh.nelems(), dnumber);

    // Compute matrix:
    //    matrix = I + inv(M)*S
    Global_FEM<Elem> gfem(sim);
    matrix = gfem.stiffness(diffno);
    auto const inv_mass = gfem.inv_lumped_mass();

    auto const verts_to_edges = sim.disc.mesh.ask_up(0, 1);
    auto row_functor = OMEGA_H_LAMBDA(int const row) {
      auto const row_begin = matrix.rows_to_columns.a2ab[row];
      auto row_col = row_begin;
      auto const factor = dtau*inv_mass[row];
      // Diagonal terms
      matrix.entries[row_col] = 1.0 + factor*matrix.entries[row_col];
      row_col++;
      auto const edge_begin = verts_to_edges.a2ab[row];
      auto const edge_end = verts_to_edges.a2ab[row + 1];
      for (auto vert_edge = edge_begin; vert_edge < edge_end; ++vert_edge) {
        // Off diagonal terms
        matrix.entries[row_col] = factor*matrix.entries[row_col];
        row_col++;
      }
    };
    parallel_for(sim.disc.mesh.nverts(), std::move(row_functor));

    // Solve system:
    //    matrix * x = rhs,
    //
    //    where:
    //       x = diffused nodal conductivity
    auto const nodes_to_nodal_conductivity = sim.getset(this->nodal_conductivity);
    double relative_out, absolute_out;
    auto const cg_it = (sim.step == 0) ? sim.disc.mesh.nverts() : cg_iterations;
    auto const niter = diagonal_preconditioned_conjugate_gradient(
        matrix, nodes_to_initial_nodal_conductivity, nodes_to_nodal_conductivity, 
        relative_tolerance, absolute_tolerance, cg_it,
        relative_out, absolute_out);
    OMEGA_H_CHECK(niter <= nodes_to_nodal_conductivity.size());

    // Convert diffused nodal conductivity to diffused element conductivity
    // through averaging the nodes of each element
    auto const elems_to_nodes = sim.disc.ents_to_nodes(ELEMS);
    auto const points_to_diffused_conductivity = 
        this->points_set(this->diffused_conductivity);
    auto end_functor = OMEGA_H_LAMBDA(int const elem) {
       double value = 0.0;
       int count = 0;
       auto const elem_nodes = getnodes<Elem>(elems_to_nodes,elem);
       for (int elem_node = 0; elem_node < Elem::nodes; ++elem_node) {
          auto const node = elem_nodes[elem_node];
          value += nodes_to_nodal_conductivity[node];
          count += 1;
       }
       points_to_diffused_conductivity[elem] = value/count;
    };
    parallel_for(sim.disc.mesh.nelems(), std::move(end_functor));
  }
  void assemble_normalized_voltage_system() {
//  std::cerr << "assembling normalized voltage system\n";
    OMEGA_H_TIME_FUNCTION;
    // Get stiffness matrix
    Global_FEM<Elem> gfem(sim);
    if (diffuse_conductivity) {
       matrix = gfem.stiffness(this->diffused_conductivity);
    } else {
       matrix = gfem.stiffness(this->conductivity);
    }
    // Apply Dirichlet bc's
    auto const A = matrix;
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
    double relative_out, absolute_out;
    auto const niter = diagonal_preconditioned_conjugate_gradient(
        matrix, rhs, nodes_to_phi, relative_tolerance, absolute_tolerance, cg_it,
        relative_out, absolute_out);
    sim.globals.set("Joule heating relative tolerance", relative_out);
    sim.globals.set("Joule heating absolute tolerance", absolute_out);
    sim.globals.set("Joule heating iterations", niter);
    OMEGA_H_CHECK(niter <= nodes_to_phi.size());
//  std::cout << "phi solve took " << niter << " iterations\n";
  }
  void compute_conductance() {
    OMEGA_H_TIME_FUNCTION;
    // 2D thickness factors
    auto const grad_mult = (z_thickness > 0.0) ? 0.0 : 1.0;
    auto const grad_add = (z_thickness > 0.0) ? (normalized_anode_voltage -
                                                 normalized_cathode_voltage ) / z_total
                                              : 0.0;
    auto const grad_add_sq = grad_add * grad_add;
    // Voltage contribution
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
      auto const sig = points_to_conductivity[point];
      auto const grads = getgrads<Elem>(points_to_grad, point);
      auto const grad_phi = grad<Elem>(grads, phi);
      auto const grad_phi_sq = grad_phi * grad_phi * grad_mult + grad_add_sq;
      auto const integral = weight * sig * (grad_phi_sq);
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
       // Below: 
       //   overwrites default/specified values in YAML 
       //   modifiers for joule heating when using circuit
       anode_voltage = sim.circuit.GetMeshAnodeVoltage();
       cathode_voltage = sim.circuit.GetMeshCathodeVoltage();
       sim.globals.set("mesh voltage", sim.circuit.GetMeshVoltageDrop());
       sim.globals.set("mesh current", sim.circuit.GetMeshCurrent());
       sim.globals.set("mesh conductance", sim.circuit.GetMeshConductance());
       auto element_voltages = sim.circuit.element_voltages;
       auto element_currents = sim.circuit.element_currents;
       for (auto it = element_voltages.begin(); it != element_voltages.end(); ++it) {
           std::string name = "circuit_v" + std::to_string(it->first);
           double voltage = it->second;
           sim.globals.set(name, voltage);
       }
       for (auto it = element_currents.begin(); it != element_currents.end(); ++it) {
           std::string name = "circuit_i" + std::to_string(it->first);
           double current = it->second;
           sim.globals.set(name, current);
       }
       // Update mesh values for next solve
       auto mesh_conductance = conductance_multiplier*
                               integrated_conductance;
       sim.circuit.SetMeshConductance(mesh_conductance);
     }
  }
  void contribute_joule_heating() {
    OMEGA_H_TIME_FUNCTION;
    auto const V = anode_voltage - cathode_voltage;
    auto const Vsq = square(V);
    auto const points_to_G = this->points_get(this->conductance);
    auto const points_to_rho = this->points_get(sim.density);
    auto const points_to_weight = this->points_get(sim.weight);
    auto const points_to_e_dot = this->points_set(this->specific_internal_energy_rate);
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const G = points_to_G[point];
      auto const P = Vsq * G;
      auto const rho = points_to_rho[point];
      auto const weight = points_to_weight[point];
      auto const mass = rho * weight;
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
