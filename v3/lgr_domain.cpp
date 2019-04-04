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

template <class Index>
static void LGR_NOINLINE collect_entities(
    counting_range<Index> const all_entities,
    domain const& domain,
    device_vector<vector3<double>, Index> const& x_vector,
    device_vector<Index, int>* domain_entities)
{
  device_allocator<int> alloc(x_vector.get_allocator());
  device_vector<int, int> is_on(int(all_entities.size()), alloc);
  lgr::fill(is_on, int(0));
  domain.mark(x_vector, int(1), &is_on);
  device_vector<int, int> offsets(int(all_entities.size()), alloc);
  lgr::transform_inclusive_scan(is_on, offsets, lgr::plus<int>(), lgr::identity<int>());
  int const domain_size = lgr::transform_reduce(is_on, int(0), lgr::plus<int>(), lgr::identity<int>());
  domain_entities->resize(domain_size);
  auto const domain_entities_to_entities = domain_entities->begin();
  auto const entities_to_offsets = offsets.cbegin();
  auto const entities_are_on = is_on.cbegin();
  auto functor = [=] (Index const e) {
    if (entities_are_on[int(e)]) {
      domain_entities_to_entities[entities_to_offsets[int(e)] - 1] = e;
    }
  };
  lgr::for_each(all_entities, functor);
}

void collect_node_set(
    counting_range<node_index> const nodes,
    domain const& domain,
    device_vector<vector3<double>, node_index> const& x_vector,
    device_vector<node_index, int>* node_set_nodes)
{
  collect_entities<node_index>(nodes, domain, x_vector, node_set_nodes);
}

}
