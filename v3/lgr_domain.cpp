#include <lgr_domain.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>

namespace lgr {

domain::~domain() {}

void union_domain::add(std::unique_ptr<domain>&& uptr) {
  m_domains.push_back(std::move(uptr));
}

void union_domain::mark(
    hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
    int const marker,
    hpc::device_vector<int, node_index>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

void union_domain::mark(
    hpc::device_array_vector<hpc::vector3<double>, element_index> const& points,
    material_index const marker,
    hpc::device_vector<material_index, element_index>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

void union_domain::mark(
    hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
    material_index const marker,
    hpc::device_vector<material_set, node_index>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

template <class Index, class IsInFunctor>
static void collect_set(
  hpc::counting_range<Index> const range,
  IsInFunctor is_in_functor,
  hpc::device_vector<Index, int>& set_items
  )
{
  hpc::device_vector<int, Index> offsets(range.size());
  hpc::transform_inclusive_scan(hpc::device_policy(), range, offsets, hpc::plus<int>(), is_in_functor);
  int const set_size = hpc::transform_reduce(hpc::device_policy(), range, int(0), hpc::plus<int>(), is_in_functor);
  set_items.resize(set_size);
  auto const set_items_to_items = set_items.begin();
  auto const items_to_offsets = offsets.cbegin();
  auto functor = [=] HPC_DEVICE (Index const item) {
    if (is_in_functor(item) != 0) {
      set_items_to_items[items_to_offsets[item] - 1] = item;
    }
  };
  hpc::for_each(hpc::device_policy(), range, functor);
}

void assign_element_materials(input const& in, state& s) {
  hpc::fill(hpc::device_policy(), s.material, material_index(0));
  hpc::device_array_vector<hpc::vector3<double>, element_index> centroid_vector(s.elements.size());
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  double const N = 1.0 / double(s.nodes_in_element.size().get());
  auto const elements_to_centroids = centroid_vector.begin();
  auto centroid_functor = [=] HPC_DEVICE (element_index const element) {
    auto centroid = hpc::vector3<double>::zero();
    for (auto const element_node : elements_to_element_nodes[element]) {
      node_index const node = element_nodes_to_nodes[element_node];
      auto const x = nodes_to_x[node].load();
      centroid = centroid + x;
    }
    centroid = centroid * N;
    elements_to_centroids[element] = centroid;
  };
  hpc::for_each(hpc::device_policy(), s.elements, centroid_functor);
  for (auto const material : in.materials) {
    auto const& domain = in.domains[material];
    if (domain) {
      domain->mark(centroid_vector, material, &s.material);
    }
  }
}

void compute_nodal_materials(input const& in, state& s) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_materials = s.material.cbegin();
  s.nodal_materials.resize(s.nodes.size());
  auto const nodes_to_materials = s.nodal_materials.begin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const node_elements = nodes_to_node_elements[node];
    auto node_materials = material_set::none();
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_materials[element];
      node_materials = node_materials | material_set(element_material);
    }
    nodes_to_materials[node] = node_materials;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
  for (auto const boundary : in.boundaries) {
    auto const& domain = in.domains[boundary];
    domain->mark(s.x, boundary, &s.nodal_materials);
  }
}

void collect_node_sets(input const& in, state& s) {
  hpc::counting_range<material_index> const all_materials(in.materials.size() + in.boundaries.size());
  s.node_sets.resize(all_materials.size());
  assert(s.nodal_materials.size() == s.nodes.size());
  auto const nodes_to_materials = s.nodal_materials.cbegin();
  for (auto const material : all_materials) {
    auto is_in_functor = [=] HPC_DEVICE (node_index const node) -> int {
      material_set const materials = nodes_to_materials[node];
      return (materials.contains(material_set(material))) ? 1 : 0;
    };
    collect_set(s.nodes, is_in_functor, s.node_sets[material]);
  }
}

void collect_element_sets(input const& in, state& s)
{
  s.element_sets.resize(in.materials.size());
  auto const elements_to_material = s.material.cbegin();
  for (auto const material : in.materials) {
    auto is_in_functor = [=] HPC_DEVICE (element_index const element) -> int {
      material_index const element_material = elements_to_material[element];
      return (element_material == material) ? 1 : 0;
    };
    collect_set(s.elements, is_in_functor, s.element_sets[material]);
  }
}

std::unique_ptr<domain> epsilon_around_plane_domain(plane const& p, double eps) {
  auto out = std::make_unique<clipped_domain<all_space>>(all_space{});
  out->clip({p.normal, p.origin - eps});
  out->clip({-p.normal, -p.origin - eps});
  return out;
}

std::unique_ptr<domain> sphere_domain(hpc::vector3<double> const origin, double
    const radius) {
  lgr::sphere const s{origin, radius};
  auto out = std::make_unique<clipped_domain<lgr::sphere>>(s);
  return out;
}

std::unique_ptr<domain> half_space_domain(plane const& p) {
  auto out = std::make_unique<clipped_domain<all_space>>(all_space{});
  out->clip(p);
  return out;
}

std::unique_ptr<domain> box_domain(hpc::vector3<double> const lower_left,
    hpc::vector3<double> const upper_right) {
  auto out = std::make_unique<clipped_domain<all_space>>(all_space{});
  out->clip({hpc::vector3<double>::x_axis(), lower_left(0)});
  out->clip({-hpc::vector3<double>::x_axis(), -upper_right(0)});
  out->clip({hpc::vector3<double>::y_axis(), lower_left(1)});
  out->clip({-hpc::vector3<double>::y_axis(), -upper_right(1)});
  out->clip({hpc::vector3<double>::z_axis(), lower_left(2)});
  out->clip({-hpc::vector3<double>::z_axis(), -upper_right(2)});
  return out;
}

}
