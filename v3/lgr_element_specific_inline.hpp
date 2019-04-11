#pragma once

namespace lgr {

inline double tetrahedron_volume(array<vector3<double>, 4> const x) {
  return (1.0 / 6.0) * (cross((x[1] - x[0]), (x[2] - x[0])) * (x[3] - x[0]));
}

}
