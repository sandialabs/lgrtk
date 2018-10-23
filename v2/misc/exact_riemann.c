#include <stdio.h>
#include <math.h>

#define N 1000

double f(double p4, double p1, double p5, double rho1, double rho5, double gamma) {
  double z, c1, c5, gm1, gp1, g2, fact;
  z = (p4 / p5 - 1.);
  c1 = sqrt(gamma * p1 / rho1);
  c5 = sqrt(gamma * p5 / rho5);
  gm1 = gamma - 1.;
  gp1 = gamma + 1.;
  g2 = 2. * gamma;
  fact = gm1 / g2 * (c5 / c1) * z / sqrt(1. + gp1 / g2 * z);
  fact = pow(1. - fact, g2 / gm1);
  return p1 * fact - p4;
}

// solves the exact riemann problem
// original code from Bruce Fryxell
int main() {
  // declare
  int npts, itmax, iter, i;
  double x[N];
  double rho[N];
  double u[N];
  double p[N];
  double rhol, pl, ul, rhor, pr, ur, gamma, xi, t, xl, xr;
  double rho1, p1, u1, rho5, p5, u5, p40, p41, f0, eps;
  double f1, p4, error, z, c5, gm1, gp1, gmfac1, gmfac2, fact;
  double u4, rho4, w, p3, u3, rho3, c1, c3, xsh, xcd, xft, xhd, dx;
  FILE* file;
  // define initial conditions
  // state at left of discontinuity
  rhol = 10.0;
  pl = 100.0;
  ul = 0.0;
  // state at right of discontinuity
  rhor = 1.0;
  pr = 1.0;
  ur = 0.0;
  if (ul != 0.0 || ur != 0.0) {
    printf("must have ul = ur = 0\n");
    return -1;
  }
  // equation of state
  gamma = 1.4;
  // location of discontinuity at t = 0
  xi = 2.0;
  // time at which solution is desired
  t = 0.40;
  // number of points in solution
  npts = 500;
  if (npts > N) {
    printf("number of points exceeds array size\n");
    return -1;
  }
  // spatial interval over which to compute solution
  xl = 0.0;
  xr = 5.0;
  if (xr < xl) {
    printf("xr must be greater than xl\n");
    return -1;
  }
  // begin solution
  if (pl > pr) {
    rho1 = rhol;
    p1 = pl;
    u1 = ul;
    rho5 = rhor;
    p5 = pr;
    u5 = ur;
  } else {
    rho1 = rhor;
    p1 = pr;
    u1 = ur;
    rho5 = rhol;
    p5 = pl;
    u5 = ul;
  }
  // solve for post-shock pressure by secant method
  // initial guess
  p40 = p1;
  p41 = p5;
  f0 = f(p40, p1, p5, rho1, rho5, gamma);
  // maximum number of iterations and maximum allowable relative error
  itmax = 20;
  eps = 1.e-5;
  for (iter = 0; iter < itmax; ++iter) {
    f1 = f(p41, p1, p5, rho1, rho5, gamma);
    if (f1 == f0) goto converged;
    p4 = p41 - (p41 - p40) * f1 / (f1 - f0);
    error = fabs(p4 - p41) / p41;
    if (error < eps) goto converged;
    p40 = p41;
    p41 = p4;
    f0 = f1;
  }
  printf("iteration failed to converge\n");
  return -1;
converged:
  ;
  // compute post-shock density and velocity
  z = (p4 / p5 - 1.);
  c5 = sqrt(gamma * p5 / rho5);
  gm1 = gamma - 1.;
  gp1 = gamma + 1.;
  gmfac1 = 0.5 * gm1 / gamma;
  gmfac2 = 0.5 * gp1 / gamma;
  fact = sqrt(1. + gmfac2 * z);
  u4 = c5 * z / (gamma * fact);
  rho4 = rho5 * (1. + gmfac2 * z) / (1. + gmfac1 * z);
  // shock speed
  w = c5 * fact;
  // compute values at foot of rarefaction
  p3 = p4;
  u3 = u4;
  rho3 = rho1 * pow(p3 / p1, 1. / gamma);
  // compute positions of waves
  if (pl > pr) {
    c1 = sqrt(gamma * p1 / rho1);
    c3 = sqrt(gamma * p3 / rho3);
    xsh = xi + w * t;
    xcd = xi + u3 * t;
    xft = xi + (u3 - c3) * t;
    xhd = xi - c1 * t;
    // and do say what we found
    printf("  Region    Density        Pressure        Velocity\n");
    printf("     1   %.7E   %.7E   %.7E\n", rho1, p1, u1);
    printf("     2                    RAREFACTION\n");
    printf("     3   %.7E   %.7E   %.7E\n", rho3, p3, u3);
    printf("     4   %.7E   %.7E   %.7E\n", rho4, p4, u4);
    printf("     5   %.7E   %.7E   %.7E\n", rho5, p5, u5);
    printf("\n");
    printf("\n");
    printf("  Head Of Rarefaction    x =  %.7E\n", xhd);
    printf("  Foot Of Rarefaction    x =  %.7E\n", xft);
    printf("  Contact Discontinuity  x =  %.7E\n", xcd);
    printf("  Shock                  x =  %.7E\n", xsh);
    printf("\n");
    printf("\n");
    // compute solution as a function of position
    dx = (xr - xl) / (npts - 1);
    for (i = 0; i < npts; ++i) {
      x[i] = xl + dx * i;
    }
    for (i = 0; i < npts; ++i) {
      if (x[i] < xhd) {
        rho[i] = rho1;
        p[i] = p1;
        u[i] = u1;
      } else if (x[i] < xft) {
        u[i] = 2. / gp1 * (c1 + (x[i] - xi) / t);
        fact = 1. - 0.5 * gm1 * u[i] / c1;
        rho[i] = rho1 * pow(fact, (2. / gm1));
        p[i] = p1 * pow(fact, (2. * gamma / gm1));
      } else if (x[i] < xcd) {
        rho[i] = rho3;
        p[i] = p3;
        u[i] = u3;
      } else if (x[i] < xsh) {
        rho[i] = rho4;
        p[i] = p4;
        u[i] = u4;
      } else {
        rho[i] = rho5;
        p[i] = p5;
        u[i] = u5;
      }
    }
  }
  // if pr > pl, reverse solution
  else {
    c1 = sqrt(gamma * p1 / rho1);
    c3 = sqrt(gamma * p3 / rho3);
    xsh = xi - w * t;
    xcd = xi - u3 * t;
    xft = xi - (u3 - c3) * t;
    xhd = xi + c1 * t;
    // and do say what we found
    printf("  Region    Density        Pressure        Velocity\n");
    printf("     1   %.7E   %.7E   %.7E\n", rho5, p5, u5);
    printf("     2   %.7E   %.7E   %.7E\n", rho4, p4, u4);
    printf("     3   %.7E   %.7E   %.7E\n", rho3, p3, u3);
    printf("     4                    RAREFACTION\n");
    printf("     5   %.7E   %.7E   %.7E\n", rho1, p1, u1);
    printf("\n");
    printf("\n");
    printf("  Head Of Rarefaction    x =  %.7E\n", xhd);
    printf("  Foot Of Rarefaction    x =  %.7E\n", xft);
    printf("  Contact Discontinuity  x =  %.7E\n", xcd);
    printf("  Shock                  x =  %.7E\n", xsh);
    printf("\n");
    printf("\n");
    // compute solution as a function of position
    dx = (xr - xl) / (npts - 1);
    for (i = 0; i < npts; ++i) {
      x[i] = xl + dx * i;
    }
    for (i = 0; i < npts; ++i) {
      if (x[i] < xsh) {
        rho[i] = rho5;
        p[i] = p5;
        u[i] = -u5;
      } else if (x[i] < xcd) {
        rho[i] = rho4;
        p[i] = p4;
        u[i] = -u4;
      } else if (x[i] < xft) {
        rho[i] = rho3;
        p[i] = p3;
        u[i] = -u3;
      } else if (x[i] < xhd) {
        u[i] = -2. / gp1 * (c1 + (xi - x[i]) / t);
        fact = 1. + 0.5 * gm1 * u[i] / c1;
        rho[i] = rho1 * pow(fact, (2. / gm1));
        p[i] = p1 * pow(fact, 2. * gamma / gm1);
      } else {
        rho[i] = rho1;
        p[i] = p1;
        u[i] = -u1;
      }
    }
  }
  file = fopen("output_c", "w");
  fprintf(file, "  i          x            density        pressure        velocity\n");
  fprintf(file, "\n");
  for (i = 0; i < npts; ++i) {
    fprintf(file, "%4d   %.7E   %.7E   %.7E   %.7E\n", i + 1, x[i], rho[i], p[i], u[i]);
  }
  fclose(file);
  return 0;
}
