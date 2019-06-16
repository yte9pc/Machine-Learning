# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(textFile):
    # Reads the textfiles and splits by each line
    file = open(textFile).read().split('\r\n')
    
    # Empty list for x0
    x0 = []
    # Empty list for x1
    x1 = []
    # Empty list for yVal
    yVal = []
    
    # For each line
    for line in file:
        # Split by tab-delimited value
        splitLine = line.split('\t')
        # Only add values if x0, x1, & yVal exist
        if len(splitLine) == 3:
            x0.append(splitLine[0])
            x1.append(splitLine[1])
            yVal.append(splitLine[2])
    
    # Convert xVal to a Matrix and set datatype to float
        # Had to transpose since I inputed the matrix transposed
    xVal = np.matrix([x0,x1]).getT().astype(float)
    # Convert yVal to a Matrix and set datatype to float
    yVal = np.matrix(yVal).getT().astype(float)
    
    # Figure 1
    plt.figure(1)
    # Plot the dataset
    plt.plot(x1, np.array(yVal), 'ro')
    # Axes
    plt.axis([0, 1, 0, 5])
    plt.show()
    return xVal, yVal

def standRegres(xVal, yVal):
    # Calulate theta (Normal Equation)
    theta = ((xVal.getT() * xVal).getI())*xVal.getT()*yVal
    
    # Slope of linear regression model
    m = theta.item(1)
    # Intercept of linear regression model
    b = theta.item(0)
    
    # An array x1
    x1 = np.array(xVal[:,1])
    # An array of yVal
    yVal = np.array(yVal[:,0])
    
    # Figure 2
    plt.figure(2)
    # Plot the dataset in red and best-fit line in blue
    plt.plot(x1, yVal, 'ro', x1, m*x1 + b, '-')
    plt.text(.2, 2, r'y = ' + str(m) + r'x + ' + str(b))
    plt.text(.2, .1, theta)
    plt.axis([0, 1, 0, 5])
    plt.show()
    return theta

def polyRegres(xVal, yVal):
    # An array of x1 squared
    x2 = (np.array(xVal[:,1]))**2
    # Insert x^2 into xVal matrix
    # New xVal matrix
    xVal = np.matrix(np.append(np.array(xVal), x2, axis=1))
    
    # Calulate theta (Normal Equation)
    theta = ((xVal.getT() * xVal).getI())*xVal.getT()*yVal
    
    # phi2 of Polynomial regression model
    phi2 = theta.item(2)
    # phi1 of Polynomial regression model
    phi1 = theta.item(1)
    # 1 of Polynomial regression model
    b = theta.item(0)
    
    # An array x1
    x1 = np.array(xVal[:,1])
    # An array of yVal
    yVal = np.array(yVal[:,0])

    # Figure 3
    plt.figure(3)
    # Plot the dataset in red and best-fit line in blue
    plt.plot(x1, yVal,'ro', x1, phi2*((x1)**2) + phi1*(x1) + b, '-')
    plt.text(.2, 2, r'y = ' + str(phi2) + r'x^2 + ' + str(phi1) + r'x + '+ str(b))
    plt.text(.2, .1, theta)
    plt.axis([0, 1, 0, 5])
    plt.show()
    return theta
    
xVal, yVal = loadDataSet('Q2data.txt')
theta = standRegres(xVal, yVal)
ptheta = polyRegres(xVal, yVal)