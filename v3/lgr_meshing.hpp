#pragma once

namespace lgr {

class input;
class state;

void build_mesh(input const& in, state& s);
void propagate_connectivity(state& s);

}
