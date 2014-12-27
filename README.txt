surfaceLinearEq
Solve the surface equations linearized at the first order :

D = [0 1]x[-1 0]
for all t
L{Phi}= 0
d_t{Phi}+g*eta = 0 at z =0 (surface)
d_z {Phi} = 0 at z =-h
d_x {Phi} = 0 at x =0 and 1
Surface
d_t{eta} = 0

Solved using MGV and 2nd Order RK 
and boundary charges


To solve we proceed as follow :
1st iteration
calculate the boundary charges => f = 0 everywhere but at the top boundary for a step of dt/2
solve the Poisson equation 
	L{Phi} = f with initial guess Phi_old 
		and Neumann BC everywhere and Dirichlet at the surface
	=> Deduce phi1
Enforce BC on phi1 to verify the BC 

Now update eta = > eta1 for a step of dt/2
	
Calculate eta => eta for a step of dt with Phi1

Update f with eta1 and a step of dt

Solve a new Poisson equation => Phi

Enforce BC on Phi

and loop

We can test the convergence for different mesh


Dependencies: 
 matplotlib.pyplot as plt (for plotting)


***********************************************************************


