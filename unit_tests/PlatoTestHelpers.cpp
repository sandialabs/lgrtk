#include "PlatoTestHelpers.hpp"

namespace PlatoUtestHelpers
{

static Teuchos::RCP<Omega_h::Library> libOmegaH;

void finalizeOmegaH() {
  libOmegaH.reset();
}

void initializeOmegaH(int *argc , char ***argv)
{
  libOmegaH = Teuchos::rcp(new Omega_h::Library(argc, argv));
}

Teuchos::RCP<Omega_h::Library> getLibraryOmegaH()
{
  return libOmegaH;
}

} // namespace PlatoUtestHelpers
