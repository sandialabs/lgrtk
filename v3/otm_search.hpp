#pragma once

namespace lgr {
namespace search_util {
template <typename T> struct nearest_neighbors;
}
}

namespace lgr {
namespace search {

void initialize_otm_search();

void finalize_otm_search();

void do_otm_point_nearest_node_search(state &s, int max_support_nodes_per_point);

void do_otm_iterative_point_support_search(state &s, int min_support_nodes_per_point);

void do_otm_node_nearest_node_search(const state &s, search_util::nearest_neighbors<node_index> &n,
    int max_nodes_per_node);

void do_otm_point_nearest_point_search(const state &s, search_util::nearest_neighbors<point_index> &n,
    int max_nodes_per_node);

}
}
