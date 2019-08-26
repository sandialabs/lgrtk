//
//  AmgXConfigs.hpp
//  
#ifndef LGR_AMGX_CONFIGS_HPP
#define LGR_AMGX_CONFIGS_HPP

#include <string>
#include <plato/PlatoTypes.hpp>

namespace Plato {
  std::string configurationString(std::string configOption, Plato::Scalar tol=1e-10, int maxIters=10000, bool absTolType=true);
}

#endif
