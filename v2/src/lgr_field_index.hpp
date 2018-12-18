#ifndef LGR_FIELD_INDEX_HPP
#define LGR_FIELD_INDEX_HPP

#include <cstddef>

namespace lgr {

struct FieldIndex {
  FieldIndex() : storage_index(~std::size_t(0)) {}
  std::size_t storage_index;
  bool is_valid() const { return storage_index != (~std::size_t(0)); }
};

}  // namespace lgr

#endif
