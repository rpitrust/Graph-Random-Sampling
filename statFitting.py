import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy import optimize

# Given the y value calculated by the fit 
# and the original value, to calculate the residual for evaluation 
# of the fit
def calculateResidual(yfit, yvalue):
    sum = 0
    assert( yfit.size != 0)
    for i in range(0, yfit.size):
        r = (yfit[i] - yvalue[i]) **2
        sum += r
    return sum

# Try fit with a polynomial, but plot in log-log scale
def polyFit(xdata, ydata, inFile, w, flag, orig = False): 
    xnew = [];
    ynew = [];
    # Ensure only valid values to take logarithm
    for i in xdata:
        if i != 0.0:
            xnew.append(i);
    for j in ydata:
        if j != 0.0:
            ynew.append(j);
    # Make some size adjustments
    sizeX = len(xnew);
    sizeY = len(ynew);
    if sizeX > sizeY:
        xnew = xnew[:sizeY];  
    elif sizeX < sizeY:
        ynew = ynew[:sizeX];
    xnew = np.array(xnew);
    ynew = np.array(ynew); 
    print('final size: ');
    print(xnew.size);    
   
    fig = plt.figure();
    xdt = np.log(xnew);
    ydt = np.log(ynew);
    if (flag == 0):
        plt.scatter(xnew, ynew,marker = 'o', color = 'b', alpha = 0.7, s = 30);
    else:
        plt.scatter(xnew, ynew,marker = 'x', color = 'r', alpha = 0.7, s = 30);
        
    coefs = np.polyfit(xdt, ydt, 6); # approximate with 6 degree polynomial
    ffit = np.poly1d(coefs);
    yfit = np.exp(ffit(xdt)); # get back in linear scale
    # calculate residual
    res = calculateResidual(yfit, ynew)
    # plot & formatting
    plt.plot(xnew, yfit, 'g-', label = 'fit');
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')  
    plt.xscale('log')
    plt.yscale('log')
    plt.text(6, 0.5, 'least square residual = %5.3f' % res)
    if (orig == False): 
        if (flag == 0):     
            title = 'Curve Fit For Random Sample Out-Degree Distributions'   
        else:
            title = 'Curve Fit For Random Sample In-Degree Distributions'
        if (flag == 0):
            outGraph = 'FittedGraph/{}-outdegree-distribution-sample-{}.png'.format(inFile, w)
        else:
            outGraph = 'FittedGraph/{}-indegree-distribution-sample-{}.png'.format(inFile, w)   
    else:
        if (flag == 0):     
            title = 'Curve Fit For WikiTalk Out-Degree Distributions'   
        else:
            title = 'Curve Fit For WikiTalk In-Dgree Distributions'
        if (flag == 0):
            outGraph = 'FittedGraph/{}-outdegree-distribution-{}.png'.format(inFile, w)
        else:
            outGraph = 'FittedGraph/{}-indegree-distribution-{}.png'.format(inFile, w)          
    plt.title(title);    
    plt.legend();    
    plt.savefig(outGraph)
    plt.close()    


# fitting using power law
# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)
# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y: (y - fitfunc(p, x))
pinit = [1, -1, 0.]

def powerLawFit(xdata, ydata, inFile, w, flag, orig = False):
    xnew = [];
    ynew = [];
    # Ensure only valid values to take logarithm
    for i in xdata:
        if i != 0.0:
            xnew.append(i);
    for j in ydata:
        if j != 0.0:
            ynew.append(j);
    # Make some size adjustments
    sizeX = len(xnew);
    sizeY = len(ynew);
    if sizeX > sizeY:
        xnew = xnew[:sizeY];  
    elif sizeX < sizeY:
        ynew = ynew[:sizeX];
    xnew = np.array(xnew);
    ynew = np.array(ynew); 
    logx = np.log(xnew)
    logy = np.log(ynew)
    print('final size: ');
    print(xnew.size);
    out = optimize.leastsq(errfunc, pinit, args = (logx, logy), full_output = 1)
    pfinal = out[0]
    covar = out[1]
    print('pfinal = ')
    print(pfinal)
    print('covar = ')
    print(covar)
    index = pfinal[1]
    amp = np.exp(pfinal[0])
    
    # plotting the data
    fig = plt.figure();    
    yval = powerlaw(xnew, amp, index);
    plt.loglog(xnew, yval,label = 'fit') # fit
    # data
    if flag == 0:
        plt.loglog(xnew, ynew, 'o', color = 'g',label = 'data')
    else:
        plt.loglog(xnew, ynew, 'x', color = 'r', label = 'data')        
    plt.xlabel('Degree')    
    plt.ylabel('Percentage of nodes')  
    plt.text(6, 0.5, 'fit(in linear scale) = %5.2fx^%5.2f' % (amp, index))
    # calculate residual
    res = calculateResidual(yval, ynew)
    # print out text
    plt.text(6, 0.2, 'least square residual = %5.3f' % res)
    
    if (orig == False): 
        if (flag == 0):     
            title = 'Curve Fit For Random Sample Out-Degree Distributions'   
        else:
            title = 'Curve Fit For Random Sample In-Degree Distributions'
        if (flag == 0):
            outGraph = 'FittedGraph/{}-outdegree-distribution-sample-{}-pLaw.png'.format(inFile, w)
        else:
            outGraph = 'FittedGraph/{}-indegree-distribution-sample-{}-pLaw.png'.format(inFile, w)   
    else:
        if (flag == 0):     
            title = 'Curve Fit For WikiTalk Out-Degree Distributions'   
        else:
            title = 'Curve Fit For WikiTalk In-Dgree Distributions'
        if (flag == 0):
            outGraph = 'FittedGraph/{}-outdegree-distribution-{}-pLaw.png'.format(inFile, w)
        else:
            outGraph = 'FittedGraph/{}-indegree-distribution-{}-pLaw.png'.format(inFile, w)      
    plt.title(title);
    plt.legend();    
    plt.savefig(outGraph)
    plt.close()       
