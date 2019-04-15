#pragma once

namespace lgr {

class input;
class state;

void build_mesh(input const& in, state& s);
void invert_connectivity(state& s);

}
