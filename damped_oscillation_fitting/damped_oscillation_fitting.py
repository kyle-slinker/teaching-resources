#code to model the motion of a damped harmonic oscillator
#Kyle Slinker
#kyle.slinker@ncssm.edu
#June 24 2020

#bring in numerical functions and plotting utilities
import numpy as np
import matplotlib.pyplot as plt

###################################
##  begin function definitions   ##
###################################
#each function definition contains general information about what the function accomplishes
#it also contains information about inputs and outputs in the form:
# - variable_name; data_type; information about the variable
#if the return is just returned without being named, the variable name will be listed as N/A

#purpose:
# - determine the sign of a number, used to orient drag forces in the correct direction
#input(s):
# - x; number; any number, positive, nevative, or zero
#output:
# - N/A; number; -1 if x is negative, +1 if x is positive, and 0 if x is 0
def sign(x):
    if(x<0):
        return -1.0
    elif(x>0):
        return 1.0
    return 0.0    

#purpose:
# -  perform one step of 4th order Runge-Kutta integration
#input(s):
# - x0; rank 1 Python list; vector of (time,position,velocity) at start of step
# - fit_parameters; rank 1 Python list; vector of fit parameters
# - dt; number; duration of this step
#output:
# - N/A; rank 1 Python list; vector of (time,position,velocity) at end of step
def take_step(x0,fit_parameters,dt):
    k1=dt*get_derivatives(x0,fit_parameters)
    k2=dt*get_derivatives(x0+k1/2.0,fit_parameters)
    k3=dt*get_derivatives(x0+k2/2.0,fit_parameters)
    k4=dt*get_derivatives(x0+k3,fit_parameters)
    return x0+(k1+2.0*k2+2.0*k3+k4)/6.0

#purpose:
# - perform numerical integration (via 4th order Runge-Kutta) to determine the best-fit function
#input(s):
# - fit_parameters; rank 1 Python list; vector of fit parameters
# - dt; number; duration of this step
# - imax; integer; length of the data set, this tells the Runge-Kutta method how many points to output
# - rk_substeps; integer; how many Runge-Kutta steps to take between each data point
#   (i.e., divide the data set's dt by this...more iterations means better resolution)
#output:
# - f; rank 2 Python list; (time,position,velocity) at each time-step (corresponding to the time steps of data),
#   first index selects timestep number, second index selects from (time,position,velocity) at that timestep
def fit_function(fit_parameters,dt,imax,rk_substeps):
    step_dt=dt/rk_substeps
    i=1
    f=[[0,fit_parameters[0],fit_parameters[1]]]
    while (i<imax):
        new_f=take_step(f[-1],fit_parameters,step_dt)
        j=1
        while (j<rk_substeps):
            new_f=take_step(new_f,fit_parameters,step_dt)
            j+=1
        f.append([i*dt,new_f[1],new_f[2]])
        i+=1
    return f

#purpose:
# - compute the gradient of the fit function with respect to fit parameters using finite differencing
#input(s):
# - f; rank 2 Python list; (time,position,velocity) at each time-step (see fit_function)
# - fit_parameters; rank 1 Python list; vector of fit parameters
# - dt; number; time between data points
# - imax; integer; the number of data points
# - rk_substeps; integer; number of numerical integration steps between data points (see fit_function)
# - fd_scale; number; controls size of finite differencing (see grad_fit_function)
#output:
# - N/A; rank 2 Python list; derivative of the fit function with respect to fit parameter at each timestep
#   first index selects fit parameter, second index selects timestep
def grad_fit_function(f,fit_parameters,dt,imax,rk_substeps,fd_scale):
    #rename fit_parameters, otherwise the expresssions get way too long
    a=fit_parameters
    
    f0=fit_function([a[0]*(1+fd_scale),a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9]],dt,imax,rk_substeps)
    f1=fit_function([a[0],a[1]*(1+fd_scale),a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9]],dt,imax,rk_substeps)
    f2=fit_function([a[0],a[1],a[2]*(1+fd_scale),a[3],a[4],a[5],a[6],a[7],a[8],a[9]],dt,imax,rk_substeps)
    f3=fit_function([a[0],a[1],a[2],a[3]*(1+fd_scale),a[4],a[5],a[6],a[7],a[8],a[9]],dt,imax,rk_substeps)
    f4=fit_function([a[0],a[1],a[2],a[3],a[4]*(1+fd_scale),a[5],a[6],a[7],a[8],a[9]],dt,imax,rk_substeps)
    f5=fit_function([a[0],a[1],a[2],a[3],a[4],a[5]*(1+fd_scale),a[6],a[7],a[8],a[9]],dt,imax,rk_substeps)
    f6=fit_function([a[0],a[1],a[2],a[3],a[4],a[5],a[6]*(1+fd_scale),a[7],a[8],a[9]],dt,imax,rk_substeps)
    f7=fit_function([a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]*(1+fd_scale),a[8],a[9]],dt,imax,rk_substeps)
    f8=fit_function([a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]*(1+fd_scale),a[9]],dt,imax,rk_substeps)
    dfd0=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[0]),f,f0))
    dfd1=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[1]),f,f1))
    dfd2=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[2]),f,f2))
    dfd3=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[3]),f,f3))
    dfd4=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[4]),f,f4))
    dfd5=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[5]),f,f5))
    dfd6=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[6]),f,f6))
    dfd7=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[7]),f,f7))
    dfd8=list(map(lambda foo0,foo1: (foo1[1]-foo0[1])/(fd_scale*a[8]),f,f8))
    return [dfd0,dfd1,dfd2,dfd3,dfd4,dfd5,dfd6,dfd7,dfd8]

#purpose:
# - compute the gradient of chi^2 with respect to fit parameters, given the gradient of the fit function
#   (essentially implementing the chain rule)
#input(s):
# - f; rank 2 Python list; (time,position,velocity) at each time-step (see fit_function)
# - fit_parameters; rank 1 Python list; vector of fit parameters
# - data; rank 2 numpy array; fit data, first index is data point number, second index selects (time,position)
# - rk_substeps; integer; number of numerical integration steps between data points (see fit_function)
# - fd_scale; number; controls size of finite differencing (see grad_fit_function)
#output:
# - N/A; rank 2 Python list; gradient of fit function, first index selects gradient component (derivative
#   with respect to which fit parameter), second index selects time
def grad_chi2(f,fit_parameters,data,rk_substeps,fd_scale):
    imax=len(data)
    dt=(data[-1][0]-data[0][0])/(imax-1)
    [dfd0,dfd1,dfd2,dfd3,dfd4,dfd5,dfd6,dfd7,dfd8]=grad_fit_function(f,fit_parameters,dt,imax,rk_substeps,fd_scale)
    dcd0=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd0,data))))
    dcd1=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd1,data))))
    dcd2=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd2,data))))
    dcd3=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd3,data))))
    dcd4=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd4,data))))
    dcd5=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd5,data))))
    dcd6=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd6,data))))
    dcd7=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd7,data))))
    dcd8=2*np.sum(np.array(list(map(lambda foo0,foo1,foo2: foo1*(foo0[1]-foo2[1]),f,dfd8,data))))
    return np.array([dcd0,dcd1,dcd2,dcd3,dcd4,dcd5,dcd6,dcd7,dcd8])

#purpose:
# - compute chi^2, the sum of the square of the residuals between the fit and data
#input(s):
# - data; rank 2 numpy array; fit data, first index is data point number, second index selects (time,position)
# - f; rank 2 Python list; (time,position,velocity) at each time-step (see fit_function)
#output:
# - N/A; number; the value of chi^2 for the given fit f
def chi2(data,f):
    return np.sum(np.array(list(map(lambda foo0,foo1: (foo0[1]-foo1[1])**2,f,data))))

#purpose:
# - perform a gradient descent search for fit parameters which minimize chi^2
#input(s):
# - gd_iterations; integer; how many steps of gradient descent to do
# - alpha; number; parameter controlling the size of each gradient descent step
# - data; rank 2 numpy array; fit data, first index is data point number, second index selects (time,position)
# - initial_fit_parameters; rank 1 Python list; vector of fit parameters before fit (initial guess)
# - rk_substeps; integer; number of numerical integration steps between data points (see fit_function)
# - fd_scale; number; controls size of finite differencing (see grad_fit_function)
#output:
# - fit_parameters; rank 1 Python list; vector of fit parameters after fit
# - chi2s; rank 2 numpy array; the value of chi^2 at each gradient descent iteration, first index is iteration number, 
#   second index selects (iteration #,chi^2)
# - note that these returns are grouped together in a ragged Python list to make one output
def gradient_descent(gd_iterations,alpha,data,initial_fit_parameters,rk_substeps,fd_scale):
    imax=len(data)
    dt=(data[-1][0]-data[0][0])/(imax-1)
    fit_parameters = np.array(initial_fit_parameters)
    f=fit_function(fit_parameters,dt,imax,rk_substeps)
    chi2s=[[0,chi2(data,f)]]
    i=1
    while (i<=gd_iterations):
        print("gradient_descent iteration: " + str(i) + "/" + str(gd_iterations), end="\r")
        fit_parameters -= alpha * np.append(np.array(grad_chi2(f,fit_parameters.tolist(),data,rk_substeps,fd_scale)),0.0)
        f=fit_function(fit_parameters,dt,imax,rk_substeps)
        chi2s.append([i,chi2(data,f)])
        i+=1
    print("\n")
    print("final       chi2=" + str(chi2s[-1][1]))
    print("final Delta chi2=" + str(np.abs(chi2s[-1][1]/chi2s[-2][1]-1.0)))
    print("\n")
    return [fit_parameters,np.array(chi2s)]

#purpose:
# - display data about the fit which was determined
#input(s):
# - fit_parameters; rank 1 Python list; vector of fit parameters
# - data; rank 2 numpy array; fit data, first index is data point number, second index selects (time,position)
# - f; rank 2 Python list; (time,position,velocity) at each time-step (see fit_function)
# - chi2s; rank 2 Python list; the value of chi^2 throughout the fit (see gradient_descent)
#output:
# - no formal return, prints new values of parameters to screen and saves plots showing fit, residuals,
#   and learning to the code's directory
def output_info(fit_parameters,data,f,chi2s):
    #print out the new fit parameters in a form which can easily be copied over to use as initial guesses on the next fit
    print("\n")
    print("updated fit parameters:")
    print("x0=" + str(fit_parameters[0]))
    print("v0=" + str(fit_parameters[1]))
    print("m=" + str(fit_parameters[2]))
    print("xeq=" + str(fit_parameters[3]))
    print("c0=" + str(fit_parameters[4]))
    print("c1=" + str(fit_parameters[5]))
    print("c2=" + str(fit_parameters[6]))
    print("cn=" + str(fit_parameters[7]))
    print("n=" + str(fit_parameters[8]))
    print("k=" + str(fit_parameters[9]))
    print("\n")
    
    #plot the data along with the fit determined
    plt.figure(figsize=(16, 12), dpi=300)
    plt.plot(data[:,0],data[:,1],marker='.',linestyle='')
    plt.plot(np.array(f)[:,0],np.array(f)[:,1])
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.savefig('fit.png')
    
    #plot the residuals
    plt.figure(figsize=(16, 12), dpi=300)
    plt.plot(data[:,0],np.array(f)[:,1]-data[:,1])
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (m)")
    plt.savefig('residuals.png')
    
    #plot the learning curve, chi^2 throughout the gradient descent
    plt.figure(figsize=(16, 12), dpi=300)
    plt.semilogy(chi2s[:,0],chi2s[:,1],marker='.',linestyle='')
    plt.xlabel("Iteration")
    plt.ylabel("chi^2 (m^2)")
    plt.savefig('learning.png')
    
    output=np.array(list(map(lambda foo0,foo1: [foo0[0],foo0[1],foo1[1],foo0[1]-foo1[1]],data,f)))
    np.savetxt('c0_c1_c2_cn_data_fit_residuals.csv',output,delimiter=',')

#purpose:
# - compute the value of the derivatives at each timestep in the numerical integration,
#   note that this is where the physics comes in and is also where the differential equation is communicated
#   to the code
#input(s):
# - x0; rank 1 Python list; vector of (time,position,velocity) at start of step
# - fit_parameters; rank 1 Python list; vector of fit parameters
#output:
# - N/A; rank 1 Python list; vector of (dt/dt,dx/dt,dv/dt)
def get_derivatives(x,fit_parameters):
    [time,position,velocity]=x
    [x0,v0,m,xeq,c0,c1,c2,cn,n,k]=fit_parameters
    force=-(k*(position-xeq)+c1*velocity+(c0+c2*velocity**2+cn*np.abs(velocity)**n)*sign(velocity))
    return np.array([1.0,velocity,force/m])

##############################################################################
##  definitions done; this is where the code begins its thought process   ##
##############################################################################

#read in the data, skip the first line because it should contain column titles
data=(np.genfromtxt('20cmTrial1.csv',delimiter=','))[1:]

#initial guesses for fit parameters, collected into a list
x0=-0.18693030855815537
v0=-0.3051726546250762
m=1.1466645537288631
xeq=-0.0015746613895467575
c0=0.022089169719687676
c1=0.007436547041692171
c2=0.005139077158679122
cn=0.0282202394837076
n=1.4999047900466689
k=48.84
fit_parameters=[x0,v0,m,xeq,c0,c1,c2,cn,n,k]

#set some paramters for the fit
# gd_iterations controls how many steps of gradient descent the fitter does
# alpha controls the alpha parameter in the gradient descent method
# rk_substeps controls the number of numerical integration steps between data points
# fd_scale controls the size of the difference when doing finite differencing in computing the gradient of the fit function
gd_iterations=2
alpha=1.0e-6
rk_substeps=128
fd_scale=1.0e-9

#do the fit
# fit_parameters is the updated set of fit parameters after doing the fit
# chi2s is a list of chi^2 values throughout the gradient descent (a "learning curve")
[fit_parameters,chi2s]=gradient_descent(gd_iterations,alpha,data,fit_parameters,rk_substeps,fd_scale)

#compute the best fit line
#need to know how many data points there are and the time between them
imax=len(data)
dt=(data[-1][0]-data[0][0])/(imax-1)
f=fit_function(fit_parameters,dt,imax,rk_substeps)

#output new fit parameters and graphs
output_info(fit_parameters,data,f,chi2s)