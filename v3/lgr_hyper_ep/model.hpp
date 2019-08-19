#ifndef LGR_HYPER_EP_MODEL_HPP
#define LGR_HYPER_EP_MODEL_HPP

#define COMPILE_TIME_MATERIAL_BRANCHING

#ifdef COMPILE_TIME_MATERIAL_BRANCHING
#include "model_ct.hpp"
#else
#include "model_rt.hpp"
#endif

#endif // LGR_HYPER_EP_MODEL_HPP
