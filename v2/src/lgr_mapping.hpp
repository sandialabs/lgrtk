#ifndef LGR_MAPPING_HPP
#define LGR_MAPPING_HPP

#include <Omega_h_array.hpp>

namespace lgr {

struct Mapping {
  bool is_identity;
  Mapping() : is_identity(false) {}
  Omega_h::LOs things;
  OMEGA_H_DEVICE int operator[](int const i) const {
    if (is_identity) return i;
    return things[i];
  }
};

}  // namespace lgr

#endif
