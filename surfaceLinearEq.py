#!/usr/local/bin/python
from math import *
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import errno
import numpy as np
import sys

## MGV.py
## Remi Carmigniani
#
# 

directory = 'result'
if not os.path.exists(directory):
    os.makedirs(directory)
sys.setrecursionlimit(5000)

##################################################################################################################
############################			Useful functions		      ############################
##################################################################################################################
def normInf(A):
	#Norm infinity
	#return the max of the abs of the matrix
	n = len(A)
	m = len(A[1])	
	mTemp = zeros([n,m])
	for i in range(1,n-1):
		for j in range(1,m-1):
			mTemp[i][j] = abs(A[i][j])
	return max(max(mTemp))
	
def  Poisson(pold,f,resMax):
	n = len(pold)
	m = len(pold[1])
	p = [[pold[i][j] for j in range(m)] for i in range(n)]
	res =1
	while res>resMax:
		p = MGV(p,f)
		p = bc(p)
		r = residual(p,f)
		#uncomment to print residual
		#print 'Residiual :  ' 
		#rescale BC
		f = rescale(f,r)
		res = normInf(r)
	p = bc(p)
	return p
def rescale(f,r):
	n = len(f)
	m = len(f[1])
	ds = doublesum(r)/(n-2)/(m-2)
	for i in range(1,n-1):
		for j in range(1,m-1):
			f[i][j] = f[i][j] - ds
	return f
def doublesum(r):
	n = len(r)
	m = len(r[1])
	res = 0 
	for i in range(1,n-1):
		for j in range(1,m-1):
			res = res +  r[i][j]
	return res
def MGV(Ain,f):
	#Full Geometric Multigrid
	#N = 2^k+2
	#Solve the Laplace equation
	#u guess solution
	#f is a matrix nx*ny RHS
	n = len(Ain)
	m = len(Ain[1])
	#if on the coarsest level
	if n==4 or m==4:
		return relax(Ain,f,10)
	else:
	#relax 10 times (pre-smoothing)
		Asmooth = relax(Ain,f,10)
	#Compute residual	
		res = residual(Asmooth,f)
	#restrict the residual to te next coarser grid
		res2 = restrict(res)
	#enforce BC
		res2 = bc(res2)
	#solve the error equation for the residual
		err = MGV(zeros(size(res2)),res2)
	#Add the prolongated error to the solution
		Asmooth = sumM(Asmooth,prolong(err))
	#relax 10 times
		Aout = relax(Asmooth,f,10)
	return Aout
def relax(xguess,b,ktimes):
	#solves the Poisson equation using Gauss-Seidel
	n = len(xguess)
	m = len(xguess[1])
	#
	dx = 1./(n-2)
  	dy = 1./(m-2)
	#coefficients for Poisson equation
  	coefx = 1./dx/dx
  	coefy = 1./dy/dy
  	coef0 = 2.*(coefx + coefy)
	#initialization
	u = [[xguess[i][j] for j in range(m)] for i in range(n)];
	u = bc(u)
	#iteration
  	for k in range(ktimes):
    		for i in range(1,n-1):
      			for j in range(1,m-1):
				u[i][j] = (coefx*(u[i+1][j]+u[i-1][j]) + coefy*(u[i][j+1]+u[i][j-1]) - b[i][j])/coef0;
		u=bc(u)
	return u	
	
def restrict(Afine):
	#Restriction routine (full weighting)
	r = len(Afine)
	c = len(Afine[1])
	n = r-2
   	m = c-2 
   	n2 = n/2
  	m2 = m/2
   	Acoarse = zeros([n2+2,m2+2])
   	for i in range(1,n2+1):
		for j in range(1,m2+1):
			Acoarse[i][j]=(Afine[2*i-1][2*j-1]+Afine[2*i][2*j-1]+Afine[2*i-1][2*j]+Afine[2*i][2*j])*.25
	return Acoarse
	
def residual(Ain,f):
	#residual routine for pressure Poisson equation 
	#resolution
   	n = len(Ain)
	m = len(Ain[1])
	dx = 1./(n-2)
  	dy = 1./(m-2)
	#coefficients for the Poisson equation
   	coefx = 1./dx/dx
   	coefy = 1./dy/dy
	coef0 = 2.*(coefx + coefy)
	#implement boundary conditions	
	Ain = bc(Ain)
	#residual computation
   	res = zeros(size(Ain))
	for i in range(1,n-1):
		for j in range(1,m-1):
			res[i][j]=f[i][j]+Ain[i][j]*coef0-(Ain[i][j+1]+Ain[i][j-1])*coefx-(Ain[i+1][j]+Ain[i-1][j])*coefy
	return res
def size(arr):
	return [len(arr),len(arr[1])]
def zeros(arrSize):
	return [[0 for j in range(arrSize[1])] for i in range(arrSize[0])]
def ones(arrSize):
	return [[1 for j in range(arrSize[1])] for i in range(arrSize[0])]
def prolong(Acoarse):
	#prolongation routine (for cell-centered quantities) 
	#resolution
   	n = len(Acoarse)
	m = len(Acoarse[1])
   	n2 = 2*(n-2) + 2
	m2 = 2*(m-2) + 2

   	Afine = zeros([n2,m2])

   	# prolongation operation
   	for i in range(n-1):     
		for j in range(m-1):
       			ifine = 2*i
		        jfine = 2*j
		        Afine[ifine][jfine] = 9./16.*Acoarse[i][j]+3./16.*Acoarse[i+1][j] +3./16.*Acoarse[i][j+1] + 1./16.*Acoarse[i+1][j+1]
       			Afine[ifine+1][jfine] = 3./16.*Acoarse[i][j]+9./16.*Acoarse[i+1][j] + 3./16.*Acoarse[i+1][j+1] + 1./16.*Acoarse[i][j+1]
       			Afine[ifine][jfine+1] = 3./16.*Acoarse[i][j]+1./16.*Acoarse[i+1][j] +9./16.*Acoarse[i][j+1]+3./16.*Acoarse[i+1][j+1]
       			Afine[ifine+1][jfine+1] = 1./16.*Acoarse[i][j]+ 3./16.*Acoarse[i+1][j] +3./16.*Acoarse[i][j+1] + 9./16.*Acoarse[i+1][j+1]
	return Afine

def printM(A):
	#Print a matrix row by row
	n=len(A)
	m=len(A[1])
	
	for i in range(n):
		text = ''
		for j in range(m):
			text = text + '	'+ repr(A[i][j])
		print text

def bc(q):
	n=len(q)
	m=len(q[1])
	
	# ghost cell mapping
	for i in range(n):	
		q[i][0]=q[i][1]
		q[i][m-1]=-q[i][m-2]
	for j in range(m):
		q[0][j]=q[1][j]
		q[n-1][j]=q[n-2][j]
	#corner elements (only needed for prolongation)
	q[0][0] = q[1][1]
  	q[n-1][1] = q[n-2][1]
	q[0][m-1] = -q[1][m-2]
  	q[n-1][m-1] = -q[n-2][m-2]
	return q

def bcTot(q,qold,eta,dt):
	#correct the BC
	n=len(q)
	m=len(q[1])
	
	# ghost cell mapping
	for i in range(n):	
		q[i][0]=q[i][1]
		q[i][m-1]=-q[i][m-2]+2.*((qold[i][m-2]+qold[i][m-1])*.5-dt*g*eta[i])
	for j in range(m):
		q[0][j]=q[1][j]
		q[n-1][j]=q[n-2][j]
	#corner elements (only needed for prolongation)
	q[0][0] = q[1][1]
  	q[n-1][1] = q[n-2][1]
	q[0][m-1] = -q[1][m-2]+2.*((qold[1][m-2]+qold[1][m-1])*.5-dt*g*eta[1])
  	q[n-1][m-1] = -q[n-2][m-2]+2.*((qold[n-2][m-2]+qold[n-2][m-1])*.5-dt*g*eta[n-2])
	return q
def bcEta(q):
	n = len(q)
	q[0] = q[1]
	q[n-1] =q[n-2]
	return q
def sumM(A,B):
	n=len(A)
	m=len(A[1])
	nB=len(B)
	mB=len(B[1])
	if (nB-n)**2>0:
		print 'ERROR Matrix Dimension SHOULD AGREE!!!!'
	else:
		for i in range(n):
			for j in range(m):
				A[i][j]=A[i][j]+B[i][j]
	return A
	

def ICeta(A,t,n):
	dx = 1./(n-2)
	dx = dx*(2.*pi)
	res = [0 for i in range(n)]
	for i in range(1,n-1):
		res[i] = A*cos((i-.5)*dx)
	res=bcEta(res)
	return res

def bcRHS(phi,eta,dt):
	#Return the boundary charges
	n=len(phi)
	m=len(phi[1])
	dx = 1./(n-2)
	res = zeros([n,m])
	for i in  range(1,n-1):
		res[i][m-2] = -2.*((phi[i][m-2]+phi[i][m-1])*.5-dt*g*eta[i])/dx**2
	return res
		

def vz(q,i):
#	return 	the value of dphi/dz at the position i using a center second order scheme at the position m-2
	n=len(q)
	m=len(q[1])
	dx = 1./(n-2)
	dy =1./(m-2)
	return (q[i][m-1]-q[i][m-2])/(dy)

##Exact solution 
def etaExact(xx,tt,k,om,A):
	return A*cos(k*xx)*cos(om*tt)
def errorEta(q,tt,om,k,A):
	n = len(q)
	dx = 1./(n-2)
	x = [(i-.5)*dx for i in range(n)]
	lis = [(q[i]-etaExact(x[i],tt,k,om,A))**2 for i in range(1,n-1)]
	res = 0
	for i in range(n-3):
		res = res + lis[i]+lis[i+1]
	res =res*.5*dx
	res = sqrt(res)
	return res
##################################################################################################################
############################				End			      ############################
##################################################################################################################

p=4
nx=2**p+2#n has to be of the form 2^k+2
dx0 = 1./(nx-2)
plist = [2,3,4,5]
errorList = []
dxList =[]
#Physical Parameter
h=1.
L=1.
g = 1.
k = 2.*pi/L
om = sqrt(g*k*tanh(k*h))
T = 2*pi/om
A = 0.01
resMax = 10**(-10)

for i in range(len(plist)):
	## Discretization parameter 
	p=plist[i]
	nx=2**p+2#n has to be of the form 2^k+2
	ny=nx
	n=nx
	m=ny
	#print repr(n) + '  ' + repr(m)
	dx = 1./(n-2)
	dy = 1./(m-2)
	x = [(i-.5)*dx for i in range(n)]
	y = [(i-.5)*dy for i in range(n)]
	dxList.append(dx)
	#Time parameters	
	t = 0
	dt = 0.01*(dx/dx0)**2
	Tend = T
	time = []
	#Initial conditions
	phi = zeros([n,m])
	f = zeros([n,m])
	eta = ICeta(A,t,n)
	posmax = []
	#get stack place for the time loop
	phi1= zeros([n,m])
	eta1 = ICeta(A,t,n)
	#plt.plot(x,eta)
	posmax.append(max(eta))
	time.append(t)
	#time loop
	while t<Tend:
		phiold = [[phi[i][j] for j in range(m)]for i in range(n)]
		#first iteration
		#Calculate the BC and update f
		f = bcRHS(phi,eta,dt*.5)
		#solve the POisson equation L(phi) = f
		phi1 = Poisson(phi,f,resMax)
		#r = residual(phi1,f)
		#print 'Residual first iteration ' + repr(normInf(r))
		
		#Enforce the BC
		phi1=bcTot(phi1,phiold,eta,dt*.5)
	
		#Calculate eta1 and eta
		for i in range(1,n-1):	
			eta1[i] = eta[i]+dt*.5*vz(phiold,i)
			#print repr(vz(phi,i))
			eta[i] = eta[i]+dt*vz(phi1,i)
		#Enforce the BC on eta
		eta1 = bcEta(eta1)
		eta = bcEta(eta)
		
		#recalculate f with eta1 and phi
		f = bcRHS(phi,eta1,dt)
		#solve the second iteration of phi and enforce BC
		phi = Poisson(phi,f,resMax)
		phi = bcTot(phi1,phiold,eta1,dt)
		
		#update time
		t= t+dt
		#Save position of the surface elevation 
		posmax.append(max(eta))
		time.append(t)
		
		print repr(t)

	#Calculate the error 
	err = errorEta(eta,t,om,k,A)
	print 'Error for n = ' + repr(n-2) + ' is ' +repr(err) 
	errorList.append(err)
	plt.close()
	plt.plot([x[i] for i in range(1,n-1)],[eta[i] for i in range(1,n-1)])
	plt.plot([x[i] for i in range(1,n-1)],[etaExact(x[i],t,k,om,A) for i in range(1,n-1)])
	plt.savefig('result/n'+repr(n-2)+'.png')

plt.close()
plt.loglog(dxList,errorList)
plt.title('Error')
plt.savefig('Error.png')



print 'Simulation Completed without error'


