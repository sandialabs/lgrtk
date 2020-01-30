#pragma once

namespace lgr {
namespace search {

void initialize_otm_search();

void finalize_otm_search();

void do_otm_point_node_search(lgr::state &s,
    hpc::device_range_sum<point_node_index, point_index> &points_to_point_nodes);

}
}
