from casadi import *
from cstr_params import *

r1 = 3.0
r2 = 1.0
r3 = 1.0 / 20.0
r4 = 1.0 / 3e11

def get_xk1(xk, uk, noise=None):
    xk1 = xk
    if noise is None:
        for _ in range(int(0.01 / 1e-4)):
            xk1[0] += 1e-4 * (F/V * (uk[0] + CA0s - xk1[0] - CAs) - k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2)
            xk1[1] += 1e-4 * (F/V * (T0 - xk1[1] - Ts) + (-delta_H)/sigma/Cp * k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2 + (uk[1] + Qs)/sigma/Cp/V)
    else:
        for _ in range(int(0.01 / 1e-4)):
            xk1[0] += 1e-4 * (F/V * (uk[0] + CA0s - xk1[0] - CAs) - k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2) + 1e-4 * noise[0]
            xk1[1] += 1e-4 * (F/V * (T0 - xk1[1] - Ts) + (-delta_H)/sigma/Cp * k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2 + (uk[1] + Qs)/sigma/Cp/V) + 1e-4 * noise[1]

    return xk1

def lmpc(inix):

    T = 0.1  # Time horizon
    N = 10 # number of control intervals，p
    
    # Declare model variables
    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x = vertcat(x1, x2)
    
    u1 = MX.sym('u1')
    u2 = MX.sym('u2')
    u = vertcat(u1, u2)
    
    # Model equations
    xdot = vertcat((F/V)*((u1+CA0s) - (x1+CAs)) - k0*(np.exp(-E/(R*(x2+Ts))))*((x1+CAs)**2),
                   (F/V)*(T0-x2-Ts) -((delta_H)/(sigma*Cp))*k0*(np.exp(-E/(R*(x2+Ts))))*((x1+CAs)**2) + (u2+Qs)/(sigma*Cp*V))
    
    # Objective term
    L = r1*(x1**2)+r2*(x2**2)+r3*(u1**2)+r4*(u2**2)#x.T*Q*x+u.T*R*u
    
    # Formulate discrete time dynamics
    # CVODES from the SUNDIALS suite
    dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
    opts = {'tf':T/N}#dekta
    Fc = integrator('Fc', 'cvodes', dae, opts)
    
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []
    
    # Formulate the NLP
    #initial x value

    # Xk = MX([inix[0], inix[1]])
    inix = [float(item) for item in inix]
    Xk = MX(np.array(inix))
    
    for k in range(N):
        # New NLP variable for the control
        #Uk = MX.sym('U_' + str(k),2)
        
        U1 = MX.sym('U1'+ str(k))
        U2 = MX.sym('U2'+ str(k))
        U = vertcat(U1, U2)
        
        # constraints of u
        w += [U1]
        lbw += [-3.5]
        ubw += [3.5]
        w0 += [0]
        
        w += [U2]
        lbw += [-5e+5]
        ubw += [5e+5]
        w0 += [0]
        

        #Add inequality constraint            
        if k == 0:
            g += [(2*(Xk[1]))*((F/V)*(T0-Xk[1]-Ts) -((delta_H)/(sigma*Cp))*k0*(np.exp(-E/(R*(Xk[1]+Ts))))*((Xk[0]+CAs)**2) + (U2+Qs)/(sigma*Cp*V))+gamma*(Xk[1])**2]
            lbg += [-inf]
            ubg += [0]            

        # Integrate till the end of the interval
        Fk = Fc(x0=Xk, p=U)
        Xk = Fk['xf']
        #J=J+Fk['qf']
        J=J+r1*(Xk[0]**2)+r2*(Xk[1]**2)+r3*(U1**2)+r4*(U2**2)
        # Add inequality constraint
        #g += [Xk[0]]
        #lbg += [-.25]
        #ubg += [inf]
    
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    # solver = nlpsol('solver', 'ipopt', prob,{'ipopt':{'max_iter':100}})
    solver = nlpsol('solver', 'ipopt', prob,{'ipopt':{'max_iter':100, 'print_level': 0}, 'print_time': 0})
    
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']
    
    fin_u1=w_opt[::2]
    fin_u2=w_opt[1::2]    
    
    return fin_u1,fin_u2

