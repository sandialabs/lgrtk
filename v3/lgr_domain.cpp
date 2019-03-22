#include <lgr_domain.hpp>

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

}
