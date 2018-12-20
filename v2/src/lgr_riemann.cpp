#include <cmath>
#include <lgr_for.hpp>
#include <lgr_riemann.hpp>

namespace lgr {

static double riemann_helper(
    double p4, double p1, double p5, double rho1, double rho5, double gamma) {
  double z, c1, c5, gm1, gp1, g2, fact;
  z = (p4 / p5 - 1.);
  c1 = std::sqrt(gamma * p1 / rho1);
  c5 = std::sqrt(gamma * p5 / rho5);
  gm1 = gamma - 1.;
  gp1 = gamma + 1.;
  g2 = 2. * gamma;
  fact = gm1 / g2 * (c5 / c1) * z / std::sqrt(1. + gp1 / g2 * z);
  fact = std::pow(1. - fact, g2 / gm1);
  return p1 * fact - p4;
}

// solves the exact Riemann problem
// original code from Bruce Fryxell (in Fortran!)
ExactRiemann exact_riemann(double left_density, double right_density,
    double left_pressure, double right_pressure, double shock_x, double gamma,
    double t, Omega_h::Reals x) {
  // declare
  int npts, itmax, iter;
  double rhol, pl, ul, rhor, pr, ur, xi;
  double rho1, p1, u1, rho5, p5, u5, p40, p41, f0, eps;
  double f1, p4, error, z, c5, gm1, gp1, gmfac1, gmfac2;
  double u4, rho4, w, p3, u3, rho3, c1, c3, xsh, xcd, xft, xhd;
  // define initial conditions
  // state at left of discontinuity
  rhol = left_density;
  pl = left_pressure;
  ul = 0.0;
  // state at right of discontinuity
  rhor = right_density;
  pr = right_pressure;
  ur = 0.0;
  // location of discontinuity at t = 0
  xi = shock_x;
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
  f0 = riemann_helper(p40, p1, p5, rho1, rho5, gamma);
  // maximum number of iterations and maximum allowable relative error
  itmax = 20;
  eps = 1.e-5;
  p4 = 0.0;
  for (iter = 0; iter < itmax; ++iter) {
    f1 = riemann_helper(p41, p1, p5, rho1, rho5, gamma);
    if (f1 == f0) goto converged;
    p4 = p41 - (p41 - p40) * f1 / (f1 - f0);
    error = std::abs(p4 - p41) / p41;
    if (error < eps) goto converged;
    p40 = p41;
    p41 = p4;
    f0 = f1;
  }
  Omega_h_fail("Riemann iteration failed to converge\n");
converged:;
  // compute post-shock density and velocity
  z = (p4 / p5 - 1.);
  c5 = std::sqrt(gamma * p5 / rho5);
  gm1 = gamma - 1.;
  gp1 = gamma + 1.;
  gmfac1 = 0.5 * gm1 / gamma;
  gmfac2 = 0.5 * gp1 / gamma;
  {
    auto const fact = std::sqrt(1. + gmfac2 * z);
    u4 = c5 * z / (gamma * fact);
    rho4 = rho5 * (1. + gmfac2 * z) / (1. + gmfac1 * z);
    // shock speed
    w = c5 * fact;
  }
  // compute values at foot of rarefaction
  p3 = p4;
  u3 = u4;
  rho3 = rho1 * std::pow(p3 / p1, 1. / gamma);
  npts = x.size();
  Omega_h::Write<double> u(npts);
  Omega_h::Write<double> rho(npts);
  Omega_h::Write<double> p(npts);
  // compute positions of waves
  if (pl > pr) {
    c1 = std::sqrt(gamma * p1 / rho1);
    c3 = std::sqrt(gamma * p3 / rho3);
    xsh = xi + w * t;
    xcd = xi + u3 * t;
    xft = xi + (u3 - c3) * t;
    xhd = xi - c1 * t;
    // compute solution as a function of position
    auto functor = OMEGA_H_LAMBDA(int i) {
      if (x[i] < xhd) {
        rho[i] = rho1;
        p[i] = p1;
        u[i] = u1;
      } else if (x[i] < xft) {
        u[i] = 2. / gp1 * (c1 + (x[i] - xi) / t);
        auto const fact = 1. - 0.5 * gm1 * u[i] / c1;
        rho[i] = rho1 * std::pow(fact, (2. / gm1));
        p[i] = p1 * std::pow(fact, (2. * gamma / gm1));
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
    };
    parallel_for(npts, std::move(functor));
  }
  // if pr > pl, reverse solution
  else {
    c1 = std::sqrt(gamma * p1 / rho1);
    c3 = std::sqrt(gamma * p3 / rho3);
    xsh = xi - w * t;
    xcd = xi - u3 * t;
    xft = xi - (u3 - c3) * t;
    xhd = xi + c1 * t;
    // compute solution as a function of position
    auto functor = OMEGA_H_LAMBDA(int i) {
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
        auto const fact = 1. + 0.5 * gm1 * u[i] / c1;
        rho[i] = rho1 * std::pow(fact, (2. / gm1));
        p[i] = p1 * std::pow(fact, 2. * gamma / gm1);
      } else {
        rho[i] = rho1;
        p[i] = p1;
        u[i] = -u1;
      }
    };
    parallel_for(npts, std::move(functor));
  }
  return {u, rho, p};
}

}  // namespace lgr
