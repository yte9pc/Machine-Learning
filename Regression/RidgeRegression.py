import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# more imports 


def loadDataSet(filename):
    
    # Load data from a text file and ignore the header file
    data = np.loadtxt(filename, dtype=float)

    # Store xVal and yVal as matrices 
    xVal = np.matrix(data[:,:-1])
    yVal = np.matrix(data[:,-1:])
    
    return xVal, yVal

def ridgeRegress(xVal, yVal, lamb, showFigure=True):
    
    # Identity Matrix
    I = np.identity(xVal.shape[1])

    # Calculate beta
    beta = np.matrix(((xVal.getT() * xVal) + (lamb * I)).getI() * xVal.getT() * yVal)
    
    return beta

def fold(xVal, yVal, k):
        
    # Splits xVal and yVal array into 10 folds
    xVal_split = np.array_split(xVal, 4)
    yVal_split = np.array_split(yVal, 4)

    # xVal testing set
    xVal_testing = xVal_split[k]
    # yVal testing set
    yVal_testing = yVal_split[k]
    
    # All the data but the testing set
    xVal_training = [x for i,x in enumerate(xVal_split) if i!=k]
    yVal_training = [x for i,x in enumerate(yVal_split) if i!=k]
    
    # Concatenation of the (kfold - 1) training set
    xVal_training = np.vstack(xVal_training)
    yVal_training = np.vstack(yVal_training)
    
    return xVal_testing, yVal_testing, xVal_training, yVal_training

def cv(xVal, yVal):
    
   
    errors = np.zeros(50)
    
    for i in range(0,4):
        # Call fold function
        xVal_testing, yVal_testing, xVal_training, yVal_training = fold(xVal, yVal, i)
        
        # Set of lambda values
        lambdas = np.arange(0.02,1.02,.02)

        for j in range(len(lambdas)):
            # Perform ridgeRegression to learn beta
            beta = ridgeRegress(xVal_training, yVal_training, lambdas[j], showFigure=True)

            # Predict yVal based on the learned beta and xVal testing fold
            yVal_predicted = xVal_testing * beta

            # Absoulte difference between predicted y and true y
            diff = np.square(yVal_testing - yVal_predicted).mean()
            
            # Error calculation
            errors[j] = errors[j] + diff
            

    # MSE         
    errors = np.divide(errors, 4)
 
    # Find smallest MSE
    lambdaBest = ((np.argmin(errors) + 1) * .02)
    
    # Plot lambda vs mse
    plt.plot(lambdas, errors, '-')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.show()
    return lambdaBest

def standRegress(xVal, yVal, beta):
    
    # An array x1
    x1 = np.array(xVal[:,0])
      # An array x2
    x2 = np.array(xVal[:,1])
    # An array of yVal
    yVal = np.array(yVal)
    
    # Beta values
    beta_0 = beta[0]
    beta_1 = beta[1]
    beta_2 = beta[2]
    
    # y value based on beta
    y = beta_0 + xVal[:,0]*beta_1 + xVal[:,1]*beta_2
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1,x2,yVal)
    ax.hold(True)
    ax.plot_surface(np.array(xVal), np.array(xVal), y)
    ax.set_xlabel('X1 Feature')
    ax.set_ylabel('X2 Feature')
    ax.set_zlabel('Y')
    ax.hold(False)

    return 0

def X1X2(xVal):
   
    # An array x1
    x1 = np.array(xVal[:,0])
    # An array of x2
    x2 = np.array(xVal[:,1])
    
    # Figure 2
    plt.figure(2)
    # Plot the dataset in red and best-fit line in blue
    plt.plot(x1, x2, 'ro', x1, x2, '-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    plt.show()


if __name__ == "__main__":
    
    xVal, yVal = loadDataSet('RRdata.txt')
    betaLR = ridgeRegress(xVal, yVal, lamb=0)
    print betaLR
    lambdaBest = cv(xVal, yVal)
    print lambdaBest
    betaRR = ridgeRegress(xVal, yVal, lamb=lambdaBest)
    print betaRR
    print('X1 vs X2')
    X1X2(xVal[:,-2:])
    print('LR')
    standRegress(xVal[:,-2:], yVal, betaLR)
    print('RR')
    standRegress(xVal[:,-2:], yVal, betaRR)
  
