#include <gtest/gtest.h>

#include <hpc_matrix3x3.hpp>
#include <hpc_quaternion.hpp>
#include <hpc_vector3.hpp>
#include <otm_util.hpp>

TEST(quaternion, zero_rotation)
{
  auto const eps = hpc::machine_epsilon<double>();
  auto const z   = hpc::vector3<double>::zero();
  auto const q   = hpc::quaternion<double>(1, 0, 0, 0);
  auto const I   = hpc::matrix3x3<double>::identity();
  auto const R   = hpc::rotation_tensor_from_quaternion(q);
  auto const e1  = hpc::norm(R - I);
  ASSERT_LE(e1, eps);
  auto const w  = hpc::rotation_vector_from_quaternion(q);
  auto const e2 = hpc::norm(w - z);
  ASSERT_LE(e2, eps);
  auto const p  = hpc::quaternion_from_rotation_tensor(I);
  auto const e3 = hpc::norm(p - q);
  ASSERT_LE(e3, eps);
  auto const r  = hpc::quaternion_from_rotation_vector(z);
  auto const e4 = hpc::norm(r - q);
  ASSERT_LE(e4, eps);
  auto const i  = hpc::rotation_vector_from_rotation_tensor(I);
  auto const e5 = hpc::norm(i - z);
  ASSERT_LE(e5, eps);
  auto const Z  = hpc::rotation_tensor_from_rotation_vector(z);
  auto const e6 = hpc::norm(Z - I);
  ASSERT_LE(e6, eps);
}

TEST(quaternion, pi_rotation)
{
  auto const eps = hpc::machine_epsilon<double>();
  auto const pi  = 3.14159265358979323846;
  auto const z   = hpc::vector3<double>(pi, 0, 0);
  auto const q   = hpc::quaternion<double>(0, 1, 0, 0);
  auto const I   = hpc::matrix3x3<double>(1, 0, 0, 0, -1, 0, 0, 0, -1);
  auto const R   = hpc::rotation_tensor_from_quaternion(q);
  auto const e1  = hpc::norm(R - I);
  ASSERT_LE(e1, eps);
  auto const w  = hpc::rotation_vector_from_quaternion(q);
  auto const e2 = hpc::norm(w - z);
  ASSERT_LE(e2, eps);
  auto const p  = hpc::quaternion_from_rotation_tensor(I);
  auto const e3 = hpc::norm(p - q);
  ASSERT_LE(e3, eps);
  auto const r  = hpc::quaternion_from_rotation_vector(z);
  auto const e4 = hpc::norm(r - q);
  ASSERT_LE(e4, eps);
  auto const i  = hpc::rotation_vector_from_rotation_tensor(I);
  auto const e5 = hpc::norm(i - z);
  ASSERT_LE(e5, eps);
  auto const Z  = hpc::rotation_tensor_from_rotation_vector(z);
  auto const e6 = hpc::norm(Z - I);
  ASSERT_LE(e6, eps);
}

TEST(quaternion, minus_pi_rotation)
{
  auto const eps = hpc::machine_epsilon<double>();
  auto const pi  = 3.14159265358979323846;
  auto const z   = hpc::vector3<double>(0, -pi, 0);
  auto const q   = hpc::quaternion<double>(0, 0, -1, 0);
  auto const I   = hpc::matrix3x3<double>(-1, 0, 0, 0, 1, 0, 0, 0, -1);
  auto const R   = hpc::rotation_tensor_from_quaternion(q);
  auto const e1  = hpc::norm(R - I);
  ASSERT_LE(e1, eps);
  auto const w  = hpc::rotation_vector_from_quaternion(q);
  auto const e2 = hpc::norm(w - z);
  ASSERT_LE(e2, eps);
  auto const p  = hpc::quaternion_from_rotation_tensor(I);
  auto const e3 = hpc::norm(p + q);
  ASSERT_LE(e3, eps);
  auto const r  = hpc::quaternion_from_rotation_vector(z);
  auto const e4 = hpc::norm(r - q);
  ASSERT_LE(e4, eps);
  auto const i  = hpc::rotation_vector_from_rotation_tensor(I);
  auto const e5 = hpc::norm(i + z);
  ASSERT_LE(e5, eps);
  auto const Z  = hpc::rotation_tensor_from_rotation_vector(z);
  auto const e6 = hpc::norm(Z - I);
  ASSERT_LE(e6, eps);
}

TEST(quaternion, tau_rotation)
{
  auto const eps = hpc::machine_epsilon<double>();
  auto const tau = 6.28318530717958647693;
  auto const z   = hpc::vector3<double>(0, 0, tau);
  auto const q   = hpc::quaternion<double>(-1, 0, 0, 0);
  auto const I   = hpc::matrix3x3<double>::identity();
  auto const R   = hpc::rotation_tensor_from_quaternion(q);
  auto const e1  = hpc::norm(R - I);
  ASSERT_LE(e1, eps);
  auto const w  = hpc::rotation_vector_from_quaternion(q);
  auto const e2 = hpc::norm(w);
  ASSERT_LE(e2, eps);
  auto const p  = hpc::quaternion_from_rotation_tensor(I);
  auto const e3 = hpc::norm(p + q);
  ASSERT_LE(e3, eps);
  auto const r  = hpc::quaternion_from_rotation_vector(z);
  auto const e4 = hpc::norm(r - q);
  ASSERT_LE(e4, eps);
  auto const i  = hpc::rotation_vector_from_rotation_tensor(I);
  auto const e5 = hpc::norm(i);
  ASSERT_LE(e5, eps);
  auto const Z  = hpc::rotation_tensor_from_rotation_vector(z);
  auto const e6 = hpc::norm(Z - I);
  ASSERT_LE(e6, 2 * eps);
}

TEST(quaternion, x_to_111_rotation)
{
  auto const eps    = hpc::machine_epsilon<double>();
  auto const cosine = std::sqrt(3.0) / 3.0;
  auto const theta  = std::acos(cosine);
  auto const a      = std::sqrt(2.0) / 2.0;
  auto const v      = theta * hpc::vector3<double>(0, -a, a);
  auto const R      = hpc::rotation_tensor_from_rotation_vector(v);
  auto const e1     = std::abs(R(0, 0) - cosine);
  auto const e2     = std::abs(R(1, 0) - cosine);
  auto const e3     = std::abs(R(2, 0) - cosine);
  ASSERT_LE(e1, eps);
  ASSERT_LE(e2, eps);
  ASSERT_LE(e3, eps);
  auto const e4 = std::abs(R(0, 1) + cosine);
  auto const e5 = std::abs(R(0, 2) + cosine);
  ASSERT_LE(e4, eps);
  ASSERT_LE(e5, eps);
  auto const e6 = std::abs(R(1, 1) * R(1, 1) + R(1, 2) * R(1, 2) - 2.0 / 3.0);
  auto const e7 = std::abs(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2) - 2.0 / 3.0);
  ASSERT_LE(e6, eps);
  ASSERT_LE(e7, eps);
}
