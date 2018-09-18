import numpy as np

def exact_riemann_solution(q_l,q_r,t=None,gamma=1.4):
    """Return the exact solution to the Riemann problem with initial states q_l, q_r.
       The solution is computed at time t and points x (where x may be a 1D numpy array).
       
       The input vectors are the conserved quantities but the outputs are [rho,u,p].
    """
    rho_l = q_l[0]
    u_l = q_l[1]/q_l[0]
    E_l = q_l[2]
    
    rho_r = q_r[0]
    u_r = q_r[1]/q_r[0]
    E_r = q_r[2]

    # Compute left and right state pressures
    p_l = (gamma-1.)*(E_l - 0.5*rho_l*u_l**2)
    p_r = (gamma-1.)*(E_r - 0.5*rho_r*u_r**2)

    # Compute left and right state sound speeds
    c_l = np.sqrt(gamma*p_l/rho_l)
    c_r = np.sqrt(gamma*p_r/rho_r)
    
    alpha = (gamma-1.)/(2.*gamma)
    beta = (gamma+1.)/(gamma-1.)

    # Check for cavitation
    if u_l - u_r + 2*(c_l+c_r)/(gamma-1.) < 0:
        print('Cavitation detected!  Exiting.')
        return None
    
    # Define the integral curves and hugoniot loci
    integral_curve_1   = lambda p : u_l + 2*c_l/(gamma-1.)*(1.-(p/p_l)**((gamma-1.)/(2.*gamma)))
    integral_curve_3   = lambda p : u_r - 2*c_r/(gamma-1.)*(1.-(p/p_r)**((gamma-1.)/(2.*gamma)))
    hugoniot_locus_1 = lambda p : u_l + 2*c_l/np.sqrt(2*gamma*(gamma-1.)) * ((1-p/p_l)/np.sqrt(1+beta*p/p_l))
    hugoniot_locus_3 = lambda p : u_r - 2*c_r/np.sqrt(2*gamma*(gamma-1.)) * ((1-p/p_r)/np.sqrt(1+beta*p/p_r))
    
    # Check whether the 1-wave is a shock or rarefaction
    def phi_l(p):        
        if p>=p_l: return hugoniot_locus_1(p)
        else: return integral_curve_1(p)
    
    # Check whether the 1-wave is a shock or rarefaction
    def phi_r(p):
        if p>=p_r: return hugoniot_locus_3(p)
        else: return integral_curve_3(p)
        
    phi = lambda p : phi_l(p)-phi_r(p)

    # Compute middle state p, u by finding curve intersection
    p,info, ier, msg = fsolve(phi, (p_l+p_r)/2.,full_output=True,xtol=1.e-14)
    # For strong rarefactions, sometimes fsolve needs help
    if ier!=1:
        p,info, ier, msg = fsolve(phi, (p_l+p_r)/2.,full_output=True,factor=0.1,xtol=1.e-10)
        # This should not happen:
        if ier!=1: 
            print('Warning: fsolve did not converge.')
            print(msg)

    u = phi_l(p)

    
    # Find middle state densities
    rho_l_star = (p/p_l)**(1./gamma) * rho_l
    rho_r_star = (p/p_r)**(1./gamma) * rho_r
        
    # compute the wave speeds
    ws = np.zeros(5) 
    # The contact speed:
    ws[2] = u
    
    # Find shock and rarefaction speeds
    if p>p_l: 
        ws[0] = (rho_l*u_l - rho_l_star*u)/(rho_l - rho_l_star)
        ws[1] = ws[0]
    else:
        c_l_star = np.sqrt(gamma*p/rho_l_star)
        ws[0] = u_l - c_l
        ws[1] = u - c_l_star

    if p>p_r: 
        ws[4] = (rho_r*u_r - rho_r_star*u)/(rho_r - rho_r_star)
        ws[3] = ws[4]
    else:
        c_r_star = np.sqrt(gamma*p/rho_r_star)
        ws[3] = u+c_r_star
        ws[4] = u_r + c_r    
    

    # Compute return values

    # Choose a time based on the wave speeds
    if t is None: t = 0.8*max(np.abs(x))/max(np.abs(ws))
    
    print("ws[0] ", ws[0])
    print("ws[1] ", ws[1])
    print("ws[2] ", ws[2])
    print("ws[3] ", ws[3])
    print("ws[4] ", ws[4])
#   xs = ws*t # Wave locations
#       
#   # Find solution inside rarefaction fans
#   xi = x/t
#   u1 = ((gamma-1.)*u_l + 2*(c_l + xi))/(gamma+1.)
#   u3 = ((gamma-1.)*u_r - 2*(c_r - xi))/(gamma+1.)
#   rho1 = (rho_l**gamma*(u1-xi)**2/(gamma*p_l))**(1./(gamma-1.))
#   rho3 = (rho_r**gamma*(xi-u3)**2/(gamma*p_r))**(1./(gamma-1.))
#   p1 = p_l*(rho1/rho_l)**gamma
#   p3 = p_r*(rho3/rho_r)**gamma
#   
#   rho_out = (x<=xs[0])*rho_l + (x>xs[0])*(x<=xs[1])*rho1 + (x>xs[1])*(x<=xs[2])*rho_l_star + (x>xs[2])*(x<=xs[3])*rho_r_star + (x>xs[3])*(x<=xs[4])*rho3 + (x>xs[4])*rho_r
#   u_out   = (x<=xs[0])*u_l + (x>xs[0])*(x<=xs[1])*u1 + (x>xs[1])*(x<=xs[2])*u + (x>xs[2])*(x<=xs[3])*u + (x>xs[3])*(x<=xs[4])*u3 + (x>xs[4])*u_r
#   p_out   = (x<=xs[0])*p_l + (x>xs[0])*(x<=xs[1])*p1 + (x>xs[1])*(x<=xs[2])*p + (x>xs[2])*(x<=xs[3])*p + (x>xs[3])*(x<=xs[4])*p3 + (x>xs[4])*p_r

def plot_exact_riemann_solution(rho_l=3.,u_l=0.,p_l=3.,rho_r=1.,u_r=0.,p_r=1.,t=0.4):
    gamma = 1.4
    E_l = p_l/(gamma-1.) + 0.5*rho_l*u_l**2
    E_r = p_r/(gamma-1.) + 0.5*rho_r*u_r**2
    
    q_l = [rho_l, rho_l*u_l, E_l]
    q_r = [rho_r, rho_r*u_r, E_r]
    
    exact_riemann_solution(q_l, q_r, t, gamma=gamma)

plot_exact_riemann_solution(rho_l=1.0,
                            u_l = 0.0,
                            p_l = 1.0,
                            rho_r=0.125,
                            u_r = 0.0,
                            p_r = 0.1)
