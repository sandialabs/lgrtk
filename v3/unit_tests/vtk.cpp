#include <gtest/gtest.h>
#include <lgr_input.hpp>
#include <lgr_state.hpp>
#include <otm_meshless.hpp>
#include <otm_vtk.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

TEST(vtk, canPrintOtmStateToFile) {
  lgr::state s;
  using MI = lgr::material_index;
  lgr::input in(MI(0), MI(0));

  tetrahedron_single_point(s);
  lgr::otm_update_nodal_mass(s);
  lgr::otm_initialize_state(in, s);

  lgr::otm_file_writer writer("tetrahedron_single_point");

  writer.capture(s);

  ASSERT_EQ(writer.host_s.nodes.size(), s.nodes.size());
  ASSERT_EQ(writer.host_s.points.size(), s.points.size());

  ASSERT_EQ(writer.host_s.x.size(), s.x.size());
  ASSERT_EQ(writer.host_s.u.size(), s.u.size());
  ASSERT_EQ(writer.host_s.v.size(), s.v.size());
  ASSERT_EQ(writer.host_s.mass.size(), s.mass.size());

  ASSERT_EQ(writer.host_s.xp.size(), s.xp.size());
  ASSERT_EQ(writer.host_s.rho.size(), s.V.size());
  ASSERT_EQ(writer.host_s.sigma.size(), s.sigma_full.size());
  ASSERT_EQ(writer.host_s.F_total.size(), s.F_total.size());
  ASSERT_EQ(writer.host_s.G.size(), s.G.size());
  ASSERT_EQ(writer.host_s.K.size(), s.K.size());

  writer.write(0);
}
