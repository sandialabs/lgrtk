#include <gtest/gtest.h>
#include <lgr_exodus.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>

using namespace lgr;

TEST(exodus, DISABLED_readSimpleFile) {
    using nodes_size_type = hpc::counting_range<node_index>::size_type;
    using elems_size_type = hpc::counting_range<element_index>::size_type;

    material_index mat(1);
    material_index bnd(1);
    input in(mat, bnd);
    state st;

    int err_code = read_exodus_file("test.g", in, st);

    ASSERT_EQ(err_code, 0);

    EXPECT_EQ(st.nodes.size(), nodes_size_type(0));
    EXPECT_EQ(st.elements.size(), elems_size_type(0));
}
