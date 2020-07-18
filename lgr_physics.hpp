#pragma once
#include <string>

namespace lgr {

class input;
class state;

void
run(input const& in, std::string const& filename = "");

}  // namespace lgr
