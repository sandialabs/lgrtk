#pragma once

#ifdef _MSC_VER
#ifdef lgr_library_EXPORTS
#define LGR_DLL __declspec(dllexport)
#else
#define LGR_DLL __declspec(dllimport)
#endif
#else
#define LGR_DLL
#endif
