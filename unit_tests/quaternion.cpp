#include <gtest/gtest.h>
#include <hpc_matrix3x3.hpp>
#include <hpc_quaternion.hpp>
#include <otm_util.hpp>

TEST(quaternion, zero_rotation)
{
  auto const eps = hpc::machine_epsilon<double>();
  auto const q = hpc::quaternion<double>(1.0, 0.0, 0.0, 0.0);
  auto const R = hpc::rotation_tensor_from_quaternion(q);
  auto const I = hpc::matrix3x3<double>::identity();
  auto const e1 = hpc::norm(R - I);
  ASSERT_LE(e1, eps);
  auto const w = hpc::rotation_vector_from_quaternion(q);
  auto const z = hpc::vector3<double>::zero();
  auto const e2 = hpc::norm(w - z);
  ASSERT_LE(e2, eps);
  auto const p = hpc::quaternion_from_rotation_tensor(I);
  auto const e3 = hpc::norm(p - q);
  ASSERT_LE(e3, eps);
  auto const r = hpc::quaternion_from_rotation_vector(z);
  auto const e4 = hpc::norm(r - q);
  ASSERT_LE(e4, eps);
  auto const i = hpc::rotation_vector_from_rotation_tensor(I);
  auto const e5 = hpc::norm(i - z);
  ASSERT_LE(e5, eps);
  auto const Z = hpc::rotation_tensor_from_rotation_vector(z);
  auto const e6 = hpc::norm(Z - I);
  ASSERT_LE(e6, eps);
}
