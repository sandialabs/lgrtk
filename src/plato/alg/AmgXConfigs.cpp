#include <sstream>

#include "AmgXConfigs.hpp"

namespace Plato {

  std::string configurationString(std::string configOption, double tol, int maxIters, bool absTolType)
  {
      using namespace std;
      ostringstream strStream;
    if(configOption == "eaf")
    {
          strStream <<
"{\
    \"config_version\": 2,\
    \"solver\": {\
        \"preconditioner\": {\
            \"print_grid_stats\": 1,\
            \"print_vis_data\": 1,\
            \"solver\": \"AMG\",\
            \"algorithm\": \"AGGREGATION\",\
            \"max_levels\": 50,\
            \"dense_lu_num_rows\":10000,\
            \"selector\": \"SIZE_8\",\
            \"smoother\": {\
                \"scope\": \"jacobi\",\
                \"solver\": \"BLOCK_JACOBI\",\
                \"monitor_residual\": 1,\
                \"print_solve_stats\": 1\
            },\
            \"print_solve_stats\": 1,\
            \"presweeps\": 1,\
            \"max_iters\": 1,\
            \"monitor_residual\": 1,\
            \"store_res_history\": 0,\
            \"scope\": \"amg\",\
            \"cycle\": \"CGF\",\
            \"postsweeps\": 1\
        },\
        \"solver\": \"PCG\",\
        \"print_solve_stats\": 1,\
        \"print_config\": 0,\
        \"obtain_timings\": 1,\
        \"monitor_residual\": 1,\
        \"convergence\": \"RELATIVE_INI_CORE\",\
        \"scope\": \"main\",\
        \"tolerance\": " << tol << ",\
        \"max_iters\": " << maxIters << ",\
        \"norm\": \"L2\"\
    }\
}";
    } else
    if(configOption == "pcg_noprec")
    {
          strStream <<
"{\
    \"config_version\": 2, \
    \"solver\": {\
        \"preconditioner\": {\
            \"scope\": \"amg\", \
            \"solver\": \"NOSOLVER\"\
        }, \
        \"use_scalar_norm\": 1, \
        \"solver\": \"PCG\", \
        \"print_solve_stats\": 1, \
        \"obtain_timings\": 1, \
        \"monitor_residual\": 1, \
        \"convergence\": \"RELATIVE_INI_CORE\", \
        \"scope\": \"main\", \
        \"tolerance\": " << tol << ", \
        \"max_iters\": " << maxIters << ", \
        \"norm\": \"L2\"\
    }\
}";
    } else
    if(configOption == "pcg_v")
    {
          strStream <<
"{\
    \"config_version\": 2, \
    \"solver\": {\
        \"preconditioner\": {\
            \"print_grid_stats\": 1, \
            \"print_vis_data\": 0, \
            \"solver\": \"AMG\", \
            \"smoother\": {\
                \"scope\": \"jacobi\", \
                \"solver\": \"BLOCK_JACOBI\", \
                \"monitor_residual\": 0, \
                \"print_solve_stats\": 0\
            }, \
            \"print_solve_stats\": 0, \
            \"presweeps\": 1, \
            \"max_iters\": 1, \
            \"monitor_residual\": 0, \
            \"store_res_history\": 0, \
            \"scope\": \"amg\", \
            \"max_levels\": 100, \
            \"cycle\": \"V\", \
            \"postsweeps\": 1\
        }, \
        \"solver\": \"PCG\", \
        \"print_solve_stats\": 1, \
        \"obtain_timings\": 1, \
        \"max_iters\": " << maxIters << ", \
        \"monitor_residual\": 1, \
        \"convergence\": \"RELATIVE_INI_CORE\", \
        \"scope\": \"main\", \
        \"tolerance\": " << tol << ", \
        \"norm\": \"L2\"\
    }\
}";
    } else
    if(configOption == "pcg_w")
    {
          strStream <<
"{\
    \"config_version\": 2, \
    \"solver\": {\
        \"preconditioner\": {\
            \"print_grid_stats\": 1, \
            \"print_vis_data\": 0, \
            \"solver\": \"AMG\", \
            \"smoother\": {\
                \"scope\": \"jacobi\", \
                \"solver\": \"BLOCK_JACOBI\", \
                \"monitor_residual\": 0, \
                \"print_solve_stats\": 0\
            }, \
            \"print_solve_stats\": 0, \
            \"presweeps\": 1, \
            \"max_iters\": 1, \
            \"monitor_residual\": 0, \
            \"store_res_history\": 0, \
            \"scope\": \"amg\", \
            \"max_levels\": 100, \
            \"cycle\": \"W\", \
            \"postsweeps\": 1\
        }, \
        \"solver\": \"PCG\", \
        \"print_solve_stats\": 1, \
        \"obtain_timings\": 1, \
        \"max_iters\": " << maxIters << ", \
        \"monitor_residual\": 1, \
        \"convergence\": \"RELATIVE_INI_CORE\", \
        \"scope\": \"main\", \
        \"tolerance\": " << tol << ", \
        \"norm\": \"L2\"\
    }\
}";
    } else
    if(configOption == "pcg_f")
    {
          strStream <<
"{\
    \"config_version\": 2, \
    \"solver\": {\
        \"preconditioner\": {\
            \"print_grid_stats\": 1, \
            \"print_vis_data\": 0, \
            \"solver\": \"AMG\", \
            \"smoother\": {\
                \"scope\": \"jacobi\", \
                \"solver\": \"BLOCK_JACOBI\", \
                \"monitor_residual\": 0, \
                \"print_solve_stats\": 0\
            }, \
            \"print_solve_stats\": 0, \
            \"presweeps\": 1, \
            \"max_iters\": 1, \
            \"monitor_residual\": 0, \
            \"store_res_history\": 0, \
            \"scope\": \"amg\", \
            \"max_levels\": 100, \
            \"cycle\": \"F\", \
            \"postsweeps\": 1\
        }, \
        \"solver\": \"PCG\", \
        \"print_solve_stats\": 1, \
        \"obtain_timings\": 1, \
        \"max_iters\": " << maxIters << ", \
        \"monitor_residual\": 1, \
        \"convergence\": \"RELATIVE_INI_CORE\", \
        \"scope\": \"main\", \
        \"tolerance\": " << tol << ", \
        \"norm\": \"L2\"\
    }\
}";
   } else // "agg_cheb4"
   {
	  strStream <<
"{\
    \"config_version\": 2, \
    \"determinism_flag\": 1, \
    \"solver\": {\
        \"print_grid_stats\": 1, \
        \"algorithm\": \"AGGREGATION\", \
        \"obtain_timings\": 1, \
        \"error_scaling\": 3,\
        \"solver\": \"AMG\", \
        \"smoother\": \
        {\
            \"solver\": \"CHEBYSHEV\",\
            \"preconditioner\" : \
            {\
                \"solver\": \"JACOBI_L1\",\
                \"max_iters\": 1\
            },\
            \"max_iters\": 1,\
            \"chebyshev_polynomial_order\" : 4,\
            \"chebyshev_lambda_estimate_mode\" : 2\
        },\
        \"presweeps\": 0, \
        \"postsweeps\": 1, \
        \"print_solve_stats\": 1, \
        \"selector\": \"SIZE_8\", \
        \"coarsest_sweeps\": 1, \
        \"monitor_residual\": 1, \
        \"min_coarse_rows\": 2, \
        \"scope\": \"main\", \
        \"max_levels\": 1000, \
        \"convergence\": \"";
        if (absTolType == false)
          strStream << "RELATIVE_INI_CORE";
        else if (absTolType == true)
          strStream << "ABSOLUTE";
        strStream << "\", \
        \"tolerance\": " << tol << ",\
        \"max_iters\": " << maxIters << ",\
        \"norm\": \"L2\",\
        \"cycle\": \"V\"\
    }\
}\
";
    }
    return strStream.str();
  }

}
