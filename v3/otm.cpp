#include <cassert>
#include <string>
#include <otm_meshless.hpp>

int main(int ac, char* av[])
{
  assert(ac > 0);
  std::string const filename(av[1]);
  lgr::otm_run(filename);
  return 0;
}
