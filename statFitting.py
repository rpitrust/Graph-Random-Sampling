import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy import optimize
import random
from pathlib import Path
import matplotlib.patches as patches

# To select a portion of points from the sample to fit and to test
# return value: two lists, one list contains all sample data points to derive 
# the polynomial; the other list contains all the test data points used to 
# test the polynomial fit
# Input: xdata -- x coordinates for the plot
#        ydata -- y coordinates for the plot
# Output: Sample -- a list containing two sub lists, represents the sample x,y
# data used to generate fitting polynomial; Test -- a list containing two sub lists
# represents the data points used to test the fitting polynomial
def drawSample(xdata, ydata):
    xSampleSet = [];
    ySampleSet = [];
    xTestSet = [];
    yTestSet = [];
    for i in range(0, len(xdata)):
        x = random.uniform(0, 1) # Random sampling from the sample
        if (x <= 0.8):
            xSampleSet.append(xdata[i])
            ySampleSet.append(ydata[i])
        else:
            xTestSet.append(xdata[i])
            yTestSet.append(ydata[i])
    Sample = [];
    Test = [];
    Sample.append(xSampleSet);
    Sample.append(ySampleSet);
    Test.append(xTestSet);
    Test.append(yTestSet);
    return Sample, Test

# Try fit with a polynomial, plot in log-log scale
# Input: xdata -- x corrdinates; ydata -- y coordinates
#        deg -- what degree should the fitting polynomial be, default is 4
# Output: coefficients and residual of the fitting polynomial
def polyFit(xdata, ydata, deg = 4): 
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
   
    fig = plt.figure();
    xdt = np.log(xnew);
    ydt = np.log(ynew);
    coefs, res, t1, t2, t3 = np.polyfit(xdt, ydt, deg, full = True);     
    return coefs, res

# To run fitting on a subset of data points
# Input: coefs -- the coefficient of fitting poly; xTest: test x coordinates
#        yTest: test y coordinates
# Output: r -- residual, xnew -- x coordinates after fix, ynew -- y coordinates
#         after fix, yval -- y coordinates calculated by the fitting poly
def polyTest(coefs, xTest, yTest):
    xnew = [];
    ynew = [];
    # Ensure only valid values to take logarithm
    for i in xTest:
        if i != 0.0:
            xnew.append(i);
    for j in yTest:
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
    xdt = np.log(xnew);
    ydt = np.log(ynew);    
    f = np.poly1d(coefs) # reconstruct the polynomial
    yval = np.exp(f(xdt))
    return xnew, ynew, yval

# Function that used to graph the sample/test data statistics 
# Input: xdata -- x coordinate; y data -- y coordinate, yfit -- y calculated by the fit
#        res -- the residual, test = True if data set is the test data set; False
#        if data set is the sampled data set
# Require: to have a directory named FittedGraph so the graphs will be stored under
#          this directory
def graphStat(xdata, ydata, yfit, res, title, test = False):
    path = './FittedGraph/' + title;
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    plt.title(title);
    if (test == False):
        plt.text(0.1, 0.9, 'least square residual %5.3f' % res, transform = ax.transAxes);
    plt.xlabel('Degree')    
    plt.ylabel('Percentage of nodes')     
    plt.loglog(xdata, ydata, 'o', label = 'data')
    plt.loglog(xdata, yfit, 'k-', label = 'fit');
    plt.legend();
    plt.savefig(path)
    plt.close()

# Graph the absolute error of the fitting polynomial
# Input output similar to the above function
def graphError(xdata, ydata, yfit, title):
    path = './FittedGraph/' + title;    
    plt.figure();
    plt.xlabel('Degree')
    plt.ylabel('Error')
    err = abs(yfit - ydata)
    plt.loglog(xdata, err, 'b--', label = 'Absolute Error')
    plt.legend();   
    plt.title(title)
    plt.savefig(path)
    plt.close()

# Find the best degree for the fitting polynomial -- note that only out degree
# is used since in-degree distribution resembles that. Run 100 times; for each
# degree, sample and fit for 10 times. Degree fitting from 1 to 9
# Input: outKeys -- x coordinate data, outVals -- y coordinate data orig: True
#        if it's the original graph, False if it's a sample
# Output: a txt file that contains the frequency distribution of best degree
#         at each run; will also return the best degree found to caller

def findDegree(outKeys, outVals, orig = False):
    frequencyDict = {}
    for i in range(1, 10):
        frequencyDict[i] = 0;
    if (orig == False):
        fileName = "./result.txt"
    else:
        fileName = "./result-orig.txt"
    if (Path(fileName).is_file()):
        out = open(fileName, 'w');
    else:
        out = open(fileName, 'x');
    for j in range(0, 100): # trial of 100 times
        minDeg = 1;
        minAvg = 10000;
        for deg in range(1, 10):    
            err = [];        
            for i in range(0, 10):    
                S, T = drawSample(outKeys, outVals);
                coefs, error = polyFit(S[0], S[1], deg);
                x, y, yprime = polyTest(coefs, T[0], T[1]);
                err.append(error)
            avg = sum(err)/len(err)
            if (avg < minAvg):
                minAvg = avg
                minDeg = deg
        print('SUMMARY: ', file = out)
        print('Run ', file = out)
        print(j, file = out)
        print('Minumum average error = ', file = out)
        print(minAvg, file = out)
        print('Minumum degree for fitting polynomial = ', file = out)
        print(minDeg,file = out)
        frequencyDict[minDeg] += 1 # update frequency count        
    print('Frequency Count', file = out)
    print(frequencyDict, file = out)    
    lst = list(frequencyDict.values())
    lst.sort() # sort by ascending order
    for k, v in frequencyDict.items():
        if v == lst[-1]:
            return k;

