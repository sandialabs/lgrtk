#include <lgr_domain.hpp>
#include <lgr_fill.hpp>
#include <lgr_scan.hpp>
#include <lgr_reduce.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>

namespace lgr {

domain::~domain() {}

void union_domain::add(std::unique_ptr<domain>&& uptr) {
  m_domains.push_back(std::move(uptr));
}

void union_domain::mark(
    device_vector<vector3<double>,
    node_index> const& points,
    int const marker,
    device_vector<int, node_index>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

void union_domain::mark(
    device_vector<vector3<double>,
    element_index> const& points,
    material_index const marker,
    device_vector<material_index, element_index>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

void union_domain::mark(
    device_vector<vector3<double>,
    node_index> const& points,
    material_index const marker,
    device_vector<material_set, node_index>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

template <class Index, class IsInFunctor>
static void collect_set(
  counting_range<Index> const range,
  IsInFunctor is_in_functor,
  device_vector<Index, int>& set_items
  )
{
  device_vector<int, Index> offsets(range.size(), set_items.get_allocator());
  transform_inclusive_scan(range, offsets, lgr::plus<int>(), is_in_functor);
  int const set_size = transform_reduce(range, int(0), lgr::plus<int>(), is_in_functor);
  set_items.resize(set_size);
  auto const set_items_to_items = set_items.begin();
  auto const items_to_offsets = offsets.cbegin();
  auto functor = [=] (Index const item) {
    if (is_in_functor(item) != 0) {
      set_items_to_items[items_to_offsets[item] - 1] = item;
    }
  };
  lgr::for_each(range, functor);
}

void assign_element_materials(input const& in, state& s) {
  lgr::fill(s.material, material_index(0));
  device_vector<vector3<double>, element_index>
    centroid_vector(s.elements.size(), s.devpool);
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const elements_to_centroids = centroid_vector.begin();
  auto centroid_functor = [=] (element_index const element) {
    auto centroid = vector3<double>::zero();
    for (auto const element_node : elements_to_element_nodes[element]) {
      node_index const node = element_nodes_to_nodes[element_node];
      vector3<double> const x = nodes_to_x[node];
      centroid = centroid + x;
    }
    centroid = centroid * N;
    elements_to_centroids[element] = centroid;
  };
  lgr::for_each(s.elements, centroid_functor);
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
  int dimension = -1;
  switch (in.element) {
    case BAR: dimension = 1; break;
    case TRIANGLE: dimension = 2; break;
    case TETRAHEDRON: dimension = 3; break;
    case COMPOSITE_TETRAHEDRON: dimension = 3; break;
  }
  auto functor = [=](node_index const node) {
    auto const node_elements = nodes_to_node_elements[node];
    auto node_materials = material_set::none();
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_materials[element];
      node_materials = node_materials | material_set(element_material);
    }
    nodes_to_materials[node] = node_materials;
  };
  for_each(s.nodes, functor);
  for (auto const boundary : in.boundaries) {
    auto const& domain = in.domains[boundary];
    domain->mark(s.x, boundary, &s.nodal_materials);
  }
}

void collect_node_sets(input const& in, state& s) {
  s.node_sets.resize(in.materials.size() + in.boundaries.size(), s.devpool);
  assert(s.nodal_materials.size() == s.nodes.size());
  auto const nodes_to_materials = s.nodal_materials.cbegin();
  for (auto const boundary : in.boundaries) {
    auto is_in_functor = [=](node_index const node) -> int {
      material_set const materials = nodes_to_materials[node];
      return (materials.contains(material_set(boundary))) ? 1 : 0;
    };
    collect_set(s.nodes, is_in_functor, s.node_sets[boundary]);
  }
}

void collect_element_sets(input const& in, state& s)
{
  s.element_sets.resize(in.materials.size(), s.devpool);
  auto const elements_to_material = s.material.cbegin();
  for (auto const material : in.materials) {
    auto is_in_functor = [=](element_index const element) -> int {
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

std::unique_ptr<domain> sphere_domain(vector3<double> const origin, double const radius) {
  lgr::sphere const s{origin, radius};
  auto out = std::make_unique<clipped_domain<lgr::sphere>>(s);
  return out;
}

std::unique_ptr<domain> half_space_domain(plane const& p) {
  auto out = std::make_unique<clipped_domain<all_space>>(all_space{});
  out->clip(p);
  return out;
}

std::unique_ptr<domain> box_domain(vector3<double> const lower_left, vector3<double> const upper_right) {
  auto out = std::make_unique<clipped_domain<all_space>>(all_space{});
  out->clip({vector3<double>::x_axis(), lower_left(0)});
  out->clip({-vector3<double>::x_axis(), -upper_right(0)});
  out->clip({vector3<double>::y_axis(), lower_left(1)});
  out->clip({-vector3<double>::y_axis(), -upper_right(1)});
  out->clip({vector3<double>::z_axis(), lower_left(2)});
  out->clip({-vector3<double>::z_axis(), -upper_right(2)});
  return out;
}

}
