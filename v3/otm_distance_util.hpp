#pragma once

namespace lgr {
class state;
}

namespace lgr {

void otm_update_nearest_point_neighbor_distances(state &s);
void otm_update_nearest_node_neighbor_distances(state &s);
void otm_update_min_nearest_neighbor_distances(state &s);

}
