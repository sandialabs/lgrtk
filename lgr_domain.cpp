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

void collect_node_set(
    counting_range<node_index> const nodes,
    domain const& domain,
    device_vector<vector3<double>, node_index> const& x_vector,
    device_vector<node_index, int>* node_set_nodes)
{
  device_allocator<int> alloc(x_vector.get_allocator());
  device_vector<int, node_index> is_on(nodes.size(), alloc);
  lgr::fill(is_on, int(0));
  domain.mark(x_vector, int(1), &is_on);
  device_vector<int, node_index> offsets(nodes.size(), alloc);
  lgr::transform_inclusive_scan(is_on, offsets, lgr::plus<int>(), lgr::identity<int>());
  int const domain_size = lgr::transform_reduce(is_on, int(0), lgr::plus<int>(), lgr::identity<int>());
  node_set_nodes->resize(domain_size);
  auto const domain_nodes_to_nodes = node_set_nodes->begin();
  auto const nodes_to_offsets = offsets.cbegin();
  auto const nodes_are_on = is_on.cbegin();
  auto functor2 = [=] (node_index const node) {
    if (nodes_are_on[node]) {
      domain_nodes_to_nodes[nodes_to_offsets[node] - 1] = node;
    }
  };
  lgr::for_each(nodes, functor2);
}

void set_materials(input const& in, state& s) {
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
    auto const& domain = pair.second;
    domain->mark(centroid_vector, material, &s.material);
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

}
