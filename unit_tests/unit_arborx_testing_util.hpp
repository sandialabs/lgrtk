#pragma once

#include <otm_search.hpp>

namespace lgr_unit {

using namespace lgr;

struct arborx_testing_singleton
{
 public:
  static arborx_testing_singleton&
  instance()
  {
    static arborx_testing_singleton s;
    return s;
  }

  ~arborx_testing_singleton() { search::finalize_otm_search(); }

 private:
  arborx_testing_singleton() { search::initialize_otm_search(); }
};

}  // namespace lgr_unit
