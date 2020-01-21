#ifndef LGR_RAY_TRACE_HPP
#define LGR_RAY_TRACE_HPP

#include <Omega_h_input.hpp>
#include <cmath>

namespace lgr {

#define EPSILON (1.0e-16)
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2];

/* Ray-Triangle Intersection Test Routines          */
/* Different optimizations of my and Ben Trumbore's */
/* code from journals of graphics tools (JGT)       */
/* http://www.acm.org/jgt/                          */
/* by Tomas Moller, May 2000                        */
/* Repository: https://github.com/erich666/jgt-code/blob/master/
 *                                         Volume_02/
 *                                         Number_1/
 *                                         Moller1997a/
 *                                         raytri.c 
 */
using tuple3 = Omega_h::Write<double>;
OMEGA_H_DEVICE int intersect_triangle1(tuple3 orig, tuple3 dir,
			tuple3 vert0, tuple3 vert1, tuple3 vert2,
			double& t)
{

   double u,v;
   double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
   double det,inv_det;
   
   /* catch case where ray is pointing in opposite of possible directions */
   double test[3], sign;
   SUB(test,vert0,orig);
   sign = DOT(test,dir);
   if (sign < 0.0) return 0;

   /* find vectors for two edges sharing vert0 */
   SUB(edge1, vert1, vert0);
   SUB(edge2, vert2, vert0);

   /* begin calculating determinant - also used to calculate U parameter */
   CROSS(pvec, dir, edge2);

   /* if determinant is near zero, ray lies in plane of triangle */
   det = DOT(edge1, pvec);

   if (det > EPSILON)
   {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, orig, vert0);

      /* calculate U parameter and test bounds */
      u = DOT(tvec, pvec);
      if (u < 0.0 || u > det)
	 return 0;

      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);

      /* calculate V parameter and test bounds */
      v = DOT(dir, qvec);
      if (v < 0.0 || u + v > det)
	 return 0;

   }
   else if(det < -EPSILON)
   {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, orig, vert0);

      /* calculate U parameter and test bounds */
      u = DOT(tvec, pvec);
      if (u > 0.0 || u < det)
	 return 0;

      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);

      /* calculate V parameter and test bounds */
      v = DOT(dir, qvec) ;
      if (v > 0.0 || u + v < det)
	 return 0;
   }
   else return 0;  /* ray is parallell to the plane of the triangle */

   inv_det = 1.0 / det;

   /* calculate t, ray intersects triangle */
   t = DOT(edge2, qvec) * inv_det;
   (u) *= inv_det;
   (v) *= inv_det;

   return 1;
}

struct Simulation;

void setup_ray_trace(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
