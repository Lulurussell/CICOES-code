#This brings in the modules that you need
from scipy import *
from scipy.optimize import leastsq
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt

# These are the functions for use in the least squares fit
def residuals1(p, x, y, yerr,fit):
    if yerr !=0:
        err = (y - peval(x,p,fit))/yerr
    else:
        err = (y - peval(x,p,fit))
    #print('err',err)
    #print('p',p)
    return err

def residuals(p, x, y, yerr,fit):
    try:
        err = (y - peval(x,p,fit))/yerr
    except:
        err=0
    #print('err',err)
    #print('p',p)
    return err

def peval(x, p,fit='Gaussian'):
    if fit == 'Gaussian':
        pev = p[0]*exp(-(x-p[1])**2/(2*p[2]**2))
    elif fit == 'Poisson':
        factorials =[]
        for i in x:
            fact = np.math.factorial(int(i))
            factorials.append(fact)
        pev = 99*(exp(-p[0])*p[0]**x)/factorials
    elif fit == 'double':
        pev = p[0]*exp(-(x-p[1])**2/(2*p[2]**2))+p[3]*exp(-(x-p[4])**2/(2*p[5]**2))
    elif fit == 'exp':
        pev = p[0] * exp(-1*(x)/p[1]) +p[2]  # this example is fitting an exponential
    elif fit == 'horiz':
        pev = np.ones_like(x)*p[0]
    elif fit == 'triangle':
        pev = p[0]-np.abs((x-p[1]/2)/(p[2]/2))
    elif fit == 'line':
        pev = p[0] + p[1]*np.asarray(x)
    elif fit == 'poly':
        pev = p[4]*np.asarray(x)**4 + p[3]*np.asarray(x)**3 + p[2]*np.asarray(x)**2 + p[1]*np.asarray(x) + p[0]
    return pev

def FITCURVE(pm1,Am1,p0,fit,xmin,xmax,Title='',xlabel='',ylabel='',em1=None, plt = True, figsize=(10,8)):
    """
    pm1: array of the x values of the data
    Am1: array of the y values of the data
    p0: array of starting parameters
    """
    if plt:
        plt.figure(figsize=figsize)
        if em1 is None:
            em1 = sqrt(Am1)
        elif em1 != 0:    
            plt.errorbar(pm1, Am1, em1, fmt='kh', markersize=3)
        else:
            plt.plot(pm1, Am1,'kh',markersize=3)

    """do the least squares fit. It makes use of the function called residuals, above.
    note: leastsq() returns an array of the solutions (plsq[0]) as well as
    returning an integer (plsq[1]) if plsq[1] == 1,2,3, || 4, the solution was found. 
    This is not important for our purposes here. """
    plsq = leastsq(residuals1, p0, args=(pm1, Am1, em1,fit), maxfev=20000)

    # These are the best fit parameters returned
    p1 = plsq[0]
    #print(p1)
    #print('best fit: {:1.2f}, {:1.2f},{:1.2f}'.format(p1[0],p1[1],p1[2]))
    # calculate the chi2 by comparing function evaluates at best fit parameters to data.
    testpt = peval(pm1, p1,fit)
    if em1 != 0:
        sqerr = ((Am1 - testpt)/em1)**2
        chisq = sum(sqerr)
        csndf = chisq/(len(pm1) - len(p1))
        print('chisq per dof: {:1.2f}'.format(csndf))
        print('chisq={:1.2f}'.format(chisq))
        print('NDF = ',(len(pm1) - len(p1)))


    # This plots the best fit function across the range of interest on the same plot as data with error bars
    x = np.linspace(xmin, xmax,30)
    resids = residuals1(p1, pm1, Am1,0,fit)
    curve = peval(x,p1,fit)
           
    if plt:
        plt.plot(x,peval(x,p1,fit))
        

        # This sets up some nice labels
        plt.ylabel(ylabel, fontsize=16)
        plt.xlim(xmin, xmax)
        plt.xlabel(xlabel, fontsize=16)

        # add a comment
        plt.title(Title, fontsize=16)

        #Now display it.
        plt.show()
    if em1 !=0: 
        return p1, curve, resids, csndf
    else:
        return p1, curve, resids