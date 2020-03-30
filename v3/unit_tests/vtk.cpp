#include <gtest/gtest.h>
#include <lgr_state.hpp>
#include <otm_meshless.hpp>
#include <otm_vtk.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

TEST(vtk, canPrintOtmStateToFile) {
  lgr::state s;

  tetrahedron_single_point(s);
  lgr::otm_update_nodal_mass(s);

  lgr::otm_file_writer writer("tetrahedron_single_point");

  writer.capture(s);

  ASSERT_EQ(writer.host_s.nodes.size(), s.nodes.size());
  ASSERT_EQ(writer.host_s.points.size(), s.points.size());

  ASSERT_EQ(writer.host_s.x.size(), s.x.size());
  ASSERT_EQ(writer.host_s.xp.size(), s.xp.size());

  writer.write(0);
}
