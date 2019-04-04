#include <lgr_domain.hpp>
#include <lgr_fill.hpp>
#include <lgr_scan.hpp>
#include <lgr_reduce.hpp>

namespace lgr {

domain::~domain() {}

void union_domain::add(std::unique_ptr<domain>&& uptr) {
  m_domains.push_back(std::move(uptr));
}

void union_domain::mark(device_vector<vector3<double>, node_index> const& points, int const marker, device_vector<int, int>* markers) const {
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
  device_vector<int, int> is_on(int(nodes.size()), alloc);
  lgr::fill(is_on, int(0));
  domain.mark(x_vector, int(1), &is_on);
  device_vector<int, int> offsets(int(nodes.size()), alloc);
  lgr::transform_inclusive_scan(is_on, offsets, lgr::plus<int>(), lgr::identity<int>());
  int const domain_size = lgr::transform_reduce(is_on, int(0), lgr::plus<int>(), lgr::identity<int>());
  node_set_nodes->resize(domain_size);
  auto const domain_nodes_to_nodes = node_set_nodes->begin();
  auto const nodes_to_offsets = offsets.cbegin();
  auto const nodes_are_on = is_on.cbegin();
  auto functor2 = [=] (node_index const node) {
    if (nodes_are_on[int(node)]) {
      domain_nodes_to_nodes[nodes_to_offsets[int(node)] - 1] = node;
    }
  };
  lgr::for_each(nodes, functor2);
}

}
