#ifndef LGR_FIELDS_ENUM_HPP
#define LGR_FIELDS_ENUM_HPP

namespace lgr {
template <int SpatialDim>
struct FieldsEnum;

template <>
struct FieldsEnum<1> {
  // indices for full tensor:
  enum { K_F_XX = 0 };

  // indices for symmetric tensor:
  enum { K_S_XX = 0 };
};

template <>
struct FieldsEnum<2> {
  // indices for full tensor:
  enum { K_F_XX = 0, K_F_YY, K_F_XY, K_F_YX };

  // indices for symmetric tensor:
  enum { K_S_XX = 0, K_S_YY = 1, K_S_XY = 2, K_S_YX = 2 };
};

template <>
struct FieldsEnum<3> {
  // Indices for full 3x3 tensor:
  enum {
    K_F_XX = 0,
    K_F_YY,
    K_F_ZZ,
    K_F_XY,
    K_F_YZ,
    K_F_ZX,
    K_F_YX,
    K_F_ZY,
    K_F_XZ
  };

  //  Indexes into a 3 by 3 symmetric tensor stored as a length 6 vector
  enum {
    K_S_XX = 0,
    K_S_YY = 1,
    K_S_ZZ = 2,
    K_S_XY = 3,
    K_S_YZ = 4,
    K_S_ZX = 5,
    K_S_YX = 3,
    K_S_ZY = 4,
    K_S_XZ = 5
  };

  //  Indexes into a 3 by 3 skew symmetric tensor stored as a length 3 vector
  enum { K_V_XY = 0, K_V_YZ, K_V_ZX };
};

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION int TensorIndices(int ordinal);

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION int SymTensorDiagonalIndices(int ordinal);

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION int SymTensorIndices(int ordinal);

template <>
KOKKOS_INLINE_FUNCTION int TensorIndices<1>(int ordinal) {
  using FieldsEnum = FieldsEnum<1>;
  int indices[1 * 1] = {FieldsEnum::K_F_XX};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int SymTensorIndices<1>(int ordinal) {
  using FieldsEnum = FieldsEnum<1>;
  int indices[1 * 1] = {FieldsEnum::K_S_XX};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int SymTensorDiagonalIndices<1>(int ordinal) {
  using FieldsEnum = FieldsEnum<1>;
  int indices[1] = {FieldsEnum::K_S_XX};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int TensorIndices<2>(int ordinal) {
  using FieldsEnum = FieldsEnum<2>;
  int indices[2 * 2] = {FieldsEnum::K_F_XX, FieldsEnum::K_F_YY,
                        FieldsEnum::K_F_XY, FieldsEnum::K_F_YX};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int SymTensorIndices<2>(int ordinal) {
  using FieldsEnum = FieldsEnum<2>;
  int indices[2 * 2] = {FieldsEnum::K_S_XX, FieldsEnum::K_S_YY,
                        FieldsEnum::K_S_XY, FieldsEnum::K_S_YX};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int SymTensorDiagonalIndices<2>(int ordinal) {
  using FieldsEnum = FieldsEnum<2>;
  int indices[2] = {FieldsEnum::K_S_XX, FieldsEnum::K_S_YY};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int TensorIndices<3>(int ordinal) {
  using FieldsEnum = FieldsEnum<3>;
  int indices[3 * 3] = {
/* TODO: completely wrong! */
      FieldsEnum::K_F_XX, FieldsEnum::K_F_YY, FieldsEnum::K_F_ZZ,
      FieldsEnum::K_F_XY, FieldsEnum::K_F_YZ, FieldsEnum::K_F_ZX,
      FieldsEnum::K_F_YX, FieldsEnum::K_F_ZY, FieldsEnum::K_F_XZ};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int SymTensorIndices<3>(int ordinal) {
  using FieldsEnum = FieldsEnum<3>;
  int indices[3 * 3] = {
/* TODO: completely wrong! */
      FieldsEnum::K_S_XX, FieldsEnum::K_S_YY, FieldsEnum::K_S_ZZ,
      FieldsEnum::K_S_XY, FieldsEnum::K_S_YZ, FieldsEnum::K_S_ZX,
      FieldsEnum::K_S_YX, FieldsEnum::K_S_ZY, FieldsEnum::K_S_XZ};
  return indices[ordinal];
}

template <>
KOKKOS_INLINE_FUNCTION int SymTensorDiagonalIndices<3>(int ordinal) {
  using FieldsEnum = FieldsEnum<3>;
  int indices[3] = {FieldsEnum::K_S_XX, FieldsEnum::K_S_YY, FieldsEnum::K_S_ZZ};
  return indices[ordinal];
}
}  // namespace lgr

#endif
