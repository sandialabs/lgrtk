#include <cassert>
#include <string>
#include <hpc_macros.hpp>
#include <otm_meshless.hpp>

int main(int ac, char* av[])
{
  HPC_TRAP_FPE();
  if (ac < 2) {
    HPC_ERROR_EXIT("File name of Exodus mesh required. None provided.");
  }
  std::string const filename(av[1]);
  lgr::otm_run(filename);
  return 0;
}
