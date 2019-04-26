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

template <class Index>
static void collect_set(
  counting_range<Index> const range,
  device_vector<int, Index> const& is_in,
  device_vector<Index, int>& set_items
  )
{
  device_vector<int, Index> offsets(range.size(), is_in.get_allocator());
  lgr::transform_inclusive_scan(is_in, offsets, lgr::plus<int>(), lgr::identity<int>());
  int const set_size = lgr::transform_reduce(is_in, int(0), lgr::plus<int>(), lgr::identity<int>());
  set_items.resize(set_size);
  auto const set_items_to_items = set_items.begin();
  auto const items_to_offsets = offsets.cbegin();
  auto const items_are_in = is_in.cbegin();
  auto functor2 = [=] (Index const item) {
    if (items_are_in[item]) {
      set_items_to_items[items_to_offsets[item] - 1] = item;
    }
  };
  lgr::for_each(range, functor2);
}

void collect_node_set(
    counting_range<node_index> const nodes,
    domain const& domain,
    device_vector<vector3<double>, node_index> const& x_vector,
    device_vector<node_index, int>* node_set_nodes)
{
  device_allocator<int> alloc(x_vector.get_allocator());
  device_vector<int, node_index> is_in(nodes.size(), alloc);
  lgr::fill(is_in, int(0));
  domain.mark(x_vector, int(1), &is_in);
  collect_set(nodes, is_in, *node_set_nodes);
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
  for (auto const& pair : in.material_domains) {
    auto const material = pair.first;
    assert(default_material <= material);
    assert(material <= max_user_material);
    auto const& domain = pair.second;
    domain->mark(centroid_vector, material, &s.material);
  }
}

void compute_nodal_materials(input const& in, state& s) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_materials = s.material.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  s.nodal_materials.resize(s.nodes.size());
  auto const nodes_to_materials = s.nodal_materials.begin();
  vector3<double> const x_maxima(in.x_domain_size, in.y_domain_size, in.z_domain_size);
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
    vector3<double> const x = nodes_to_x[node];
    constexpr double tol = 1.0e-10;
    if ((std::abs(x(0)) < tol) || (std::abs(x(0) - x_maxima(0)) < tol)) {
      node_materials = node_materials | material_set(x_boundary_material);
    }
    if ((std::abs(x(1)) < tol) || (std::abs(x(1) - x_maxima(1)) < tol)) {
      node_materials = node_materials | material_set(y_boundary_material);
    }
    if ((std::abs(x(2)) < tol) || (std::abs(x(2) - x_maxima(2)) < tol)) {
      node_materials = node_materials | material_set(z_boundary_material);
    }
    nodes_to_materials[node] = node_materials;
  };
  for_each(s.nodes, functor);
}

void collect_node_sets(input const& in, state& s) {
  for (auto const& pair : in.node_sets) {
    auto const& domain_name = pair.first;
    auto const& domain_ptr = pair.second;
    s.node_sets.emplace(domain_name, s.devpool);
    collect_node_set(s.nodes, *domain_ptr, s.x, &(s.node_sets.find(domain_name)->second));
  }
}

void collect_element_sets(input const& in, state& s)
{
  s.element_sets.resize(in.material_count, s.devpool);
  device_vector<int, element_index> is_in(s.elements.size(), s.devpool);
  auto const elements_to_material = s.material.cbegin();
  auto const elements_are_in = is_in.begin();
  for (material_index material(0); material < in.material_count; ++material) {
    auto functor = [=](element_index const element) {
      material_index const element_material = elements_to_material[element];
      elements_are_in[element] = int(element_material == material);
    };
    for_each(s.elements, functor);
    collect_set(s.elements, is_in, s.element_sets[material]);
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
