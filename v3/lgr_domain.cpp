#include <lgr_domain.hpp>
#include <lgr_fill.hpp>
#include <lgr_inclusive_scan.hpp>

// TEMPORARY
#include <numeric>

namespace lgr {

domain::~domain() {}

void union_domain::add(std::unique_ptr<domain>&& uptr) {
  m_domains.push_back(std::move(uptr));
}

void union_domain::mark(host_vector<vector3<double>> const& points, int const marker, host_vector<int>* markers) const {
  for (auto const& uptr : m_domains) {
    uptr->mark(points, marker, markers);
  }
}

void collect_domain_entities(
    int_range const nodes,
    domain const& domain,
    host_vector<vector3<double>> const& x_vector,
    host_vector<int>* entities)
{
  host_vector<int> is_on(nodes.size());
  lgr::fill(is_on, int(0));
  domain.mark(x_vector, int(1), &is_on);
  host_vector<int> offsets(nodes.size());
  lgr::inclusive_scan(is_on, offsets);
  int const domain_size = std::accumulate(is_on.cbegin(), is_on.cend(), 0);
  entities->resize(domain_size);
  auto const domain_ents_to_ents = entities->begin();
  auto const nodes_to_offsets = offsets.cbegin();
  auto const nodes_are_on = is_on.cbegin();
  auto functor2 = [=] (int const node) {
    if (nodes_are_on[node]) {
      domain_ents_to_ents[nodes_to_offsets[node] - 1] = node;
    }
  };
  lgr::for_each(nodes, functor2);
}

}
