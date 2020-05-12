#include <lgr_state.hpp>
#include <otm_adapt.hpp>

namespace lgr {

otm_adapt_state::otm_adapt_state(state const &s) :
    node_criteria(s.nodes.size()),
    point_criteria(s.points.size()),
    other_node(s.nodes.size()),
    other_point(s.points.size()),
    node_op(s.nodes.size()),
    point_op(s.points.size()),
    point_counts(s.points.size()),
    node_counts(s.nodes.size()),
    old_points_to_new_points(s.points.size() + point_index(1)),
    old_nodes_to_new_nodes(s.nodes.size() + node_index(1)),
    new_points_to_old_points(),
    new_nodes_to_old_nodes(),
    new_point_nodes_to_nodes(),
    new_points_are_same(),
    new_nodes_are_same(),
    interpolate_from_nodes(),
    interpolate_from_points(),
    new_points(point_index(0)),
    new_nodes(node_index(0))
{
}

}
