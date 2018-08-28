#include "LGRTestHelpers.hpp"

#include <fstream>

namespace lgr {
  static Teuchos::RCP<comm::Machine>  commMachine;
  static Teuchos::RCP<Omega_h::Library> libOmegaH;

  bool fileExists(const std::string &filePath)
  {
    std::ifstream f(filePath.c_str());
    return f.good();
  }
  
  void finalizeOmegaH() {
    libOmegaH.reset();
  }

  void finalizeCommMachine() {
    commMachine.reset();
  }
  
  void initializeCommMachine(int *argc , char ***argv)
  {
    commMachine = Teuchos::rcp(new comm::Machine(argc, argv));
  }

  comm::Machine getCommMachine()
  {
    return *commMachine;
  }

  void initializeOmegaH(int *argc , char ***argv)
  {
    libOmegaH = Teuchos::rcp(new Omega_h::Library(argc, argv));
  }

  Teuchos::RCP<Omega_h::Library> getLibraryOmegaH()
  {
    return libOmegaH;
  }
}
