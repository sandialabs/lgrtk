#pragma once

#include <string>

namespace lgr {

class input;
class state;

void read_exodus_file(std::string const& filepath, input const& in, state& s);

}
