#ifndef LGR_REMAP_TYPE_HPP
#define LGR_REMAP_TYPE_HPP

namespace lgr {

enum class RemapType {
  NONE,
  NODAL,
  SHAPE,
  PER_UNIT_VOLUME,
  PER_UNIT_MASS,
  POLAR_REMAP,
};

}

#endif
