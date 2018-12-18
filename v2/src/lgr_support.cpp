#include <Omega_h_map.hpp>
#include <lgr_disc.hpp>
#include <lgr_element_functions.hpp>
#include <lgr_for.hpp>
#include <lgr_subset.hpp>
#include <lgr_support.hpp>

namespace lgr {

Support::Support(Disc& disc_in, Subset* subset_in)
    : disc(disc_in),
      subset(subset_in),
      cached_time(std::numeric_limits<double>::quiet_NaN()) {}

void Support::out_of_line_virtual_method() {}

Omega_h::Read<double> Support::ask_coords(
    double time, Omega_h::Read<double> node_coords) {
  if ((time != cached_time) || (!cached_coords.exists())) {
    cached_coords = this->interpolate_nodal(disc.dim(), node_coords);
  }
  return cached_coords;
}

struct EntitySupport : public Support {
  EntitySupport(Disc& disc_in, Subset* subset_in)
      : Support(disc_in, subset_in) {}
  int count() override final;
  Omega_h::Read<double> interpolate_nodal(int, Omega_h::Read<double>) override {
    Omega_h_fail(
        "Only nodes and integration points can interpolate a nodal field\n");
  }
  bool on_points() override final { return false; }
};

int EntitySupport::count() { return subset->count(); }

struct NodeSupport : public EntitySupport {
  NodeSupport(Disc& disc_in, Subset* subset_in)
      : EntitySupport(disc_in, subset_in) {}
  Omega_h::Read<double> interpolate_nodal(
      int ncomps, Omega_h::Read<double> node_values) override final;
};

Omega_h::Read<double> NodeSupport::interpolate_nodal(
    int ncomps, Omega_h::Read<double> node_values) {
  if (subset->mapping.is_identity) return node_values;
  return Omega_h::unmap(subset->mapping.things, node_values, ncomps);
}

template <class Elem>
Omega_h::Read<double> interpolate_nodal(MappedElemsToNodes elems_to_nodes,
    int ncomps, Omega_h::Read<double> node_values) {
  int nelems = count_elements<Elem>(elems_to_nodes);
  auto out = Omega_h::Write<double>(nelems * Elem::points * Elem::dim);
  auto functor = OMEGA_H_LAMBDA(int elem) {
    auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto Na = Elem::basis_values();
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto point = elem * Elem::points + elem_pt;
      for (int comp = 0; comp < ncomps; ++comp) {
        double val = 0.0;
        for (int elem_node = 0; elem_node < Elem::nodes; ++elem_node) {
          auto node = elem_nodes[elem_node];
          val += node_values[node * ncomps + comp] * Na[elem_pt][elem_node];
        }
        out[point * ncomps + comp] = val;
      }
    }
  };
  parallel_for(nelems, functor, "interpolation kernel");
  return out;
}

template <class Elem>
struct PointSupport : public Support {
  PointSupport(Disc& disc_in, Subset* subset_in)
      : Support(disc_in, subset_in) {}
  int count() override final { return subset->count() * Elem::points; }
  Omega_h::Read<double> interpolate_nodal(
      int ncomps, Omega_h::Read<double> node_values) override final {
    return ::lgr::interpolate_nodal<Elem>(
        subset->ents_to_nodes(), ncomps, node_values);
  }
  bool on_points() override final { return true; }
};

Support* entity_support_factory(Disc& disc, Subset* subset) {
  if (subset->entity_type == NODES) return new NodeSupport(disc, subset);
  return new EntitySupport(disc, subset);
}

template <class Elem>
Support* point_support_factory(Disc& disc, Subset* subset) {
  if (subset->entity_type == ELEMS) {
    return new PointSupport<Elem>(disc, subset);
  }
  if (subset->entity_type == SIDES) {
    return new PointSupport<typename Elem::side>(disc, subset);
  }
  OMEGA_H_NORETURN(nullptr);
}

#define LGR_EXPL_INST(Elem) template struct PointSupport<Elem>;
LGR_EXPL_INST_ELEMS_AND_SIDES
#undef LGR_EXPL_INST

#define LGR_EXPL_INST(Elem)                                                    \
  template Support* point_support_factory<Elem>(Disc&, Subset*);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
