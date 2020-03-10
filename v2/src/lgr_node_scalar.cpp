#include <lgr_node_scalar.hpp>
#include <lgr_scalar.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct NodeScalar : public Scalar {
  FieldIndex fi;
  int comp;
  Subset* subset;
  NodeScalar(
      Simulation& sim_in, std::string const& name_in, Omega_h::InputMap& pl)
      : Scalar(sim_in, name_in) {
    auto set_name = pl.get<std::string>("set");
    auto field_name = pl.get<std::string>("field");
    fi = sim.fields.find(field_name);
    comp = pl.get<int>("component", "0");
    if (!(sim.fields[fi].support->subset->mapping.is_identity)) {
      Omega_h_fail(
          "NodeScalar doesn't yet support fields (%s) on a subset of nodes!\n",
          field_name.c_str());
    }
    subset = sim.subsets.get_subset(NODES, {set_name});
    if (subset->count() != 1) {
      Omega_h_fail(
          "The subset for NodeScalar (%s) must contain only one node!\n",
          set_name.c_str());
    }
  }
  void out_of_line_virtual_method() override;
  double compute_value() override {
    auto nodes_to_data = sim.fields.get(fi);
    auto node = subset->mapping.things.get(0);
    auto ncomps = sim.fields[fi].ncomps;
    double value = 0.0;
    if (sim.fields[fi].on_points) {
       auto const verts_to_elems = sim.disc.mesh.ask_up(0, sim.disc.dim());
       auto const begin = verts_to_elems.a2ab.get(node);
       auto const end = verts_to_elems.a2ab.get(node + 1);
       auto const count = end-begin;
       for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
          auto const elem = verts_to_elems.ab2b.get(vert_elem);
          auto const point = elem; // Assumes only one point per element
          value += nodes_to_data.get(point);
       }
       value /= count;
    } else {
       value = nodes_to_data.get(node * ncomps + comp);
    }
    return value;
  }
};

void NodeScalar::out_of_line_virtual_method() {}

Scalar* node_scalar_factory(
    Simulation& sim, std::string const& name, Omega_h::InputMap& pl) {
  return new NodeScalar(sim, name, pl);
}

}  // namespace lgr
