
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def neural_net(train, test):
    y = []
    #Your code here
    return y

def knn(train, test):
    y = []
    #Your code here
    return y

def pca_LG(train, test):
    # Train x
    train_x = train[:,1:]
    # Train y
    train_y = train[:,0]
    
    # Test x
    test_x = test[:,1:]
    # Test y
    #test_y = test[:,0]
    
    # Principal components
    pca = PCA(n_components=27)
    train_x_p = pca.fit_transform(train_x)
    
    # Train model
    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_x_p, train_y)
    
    # Predict
    y = logisticRegr.predict(pca.transform(test_x))
    
    # Accuracy 
    pca_log_acc(train)
    return y

def LogistRegres(train, test):
    # Train x
    train_x = train[:,1:]
    # Train y
    train_y = train[:,0]
    
     # Test x
    test_x = test[:,1:]
    # Test y
    #test_y = test[:,0]
    
    # Train model
    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_x, train_y)
    
    # Predict
    y = logisticRegr.predict(test_x)
    
    return y

def pca_knn(train, test):
    # Train x
    train_x = train[:,1:]
    # Train y
    train_y = train[:,0]
    
    # Test x
    test_x = test[:,1:]
    # Test y
    test_y = test[:,0]
    
    # Principal components
    pca = PCA(n_components=27)
    train_x_p = pca.fit_transform(train_x)
    
    # Train model
    knn = KNeighborsClassifier()
    knn.fit(train_x_p, train_y)
    
    # Predict
    y = knn.predict(pca.transform(test_x))
    
    # Write to file
    np.savetxt('test.output.txt', y, newline='\n',fmt='%5d')
    
    # Calculate Accuracy 
    pca_knn_acc(train)
    return y

def pca_knn_acc(train):
    # Shuffle Data and random seed
    np.random.seed(37)
    np.random.shuffle(train)
    
    # 80-20 split
    test = train[5833:,:]
    train = train[:5833,:]
    
    # Create test and train split
    train_x = train[:,1:]
    train_y = train[:,0]
    test_x = test[:,1:]
    test_y = test[:,0]
    
    # Principal components
    pca = PCA(n_components=27)
    train_x_p = pca.fit_transform(train_x)
    
    # Train model
    knn = KNeighborsClassifier()
    knn.fit(train_x_p, train_y)
    
    # Predict
    y = knn.predict(pca.transform(test_x))
    
    # Accuracy
    print(accuracy_score(test_y,y))
    
def pca_log_acc(train):
    # Shuffle Data and random seed
    np.random.seed(37)
    np.random.shuffle(train)
    
    # 80-20 split
    test = train[5833:,:]
    train = train[:5833,:]
    
    # Create test and train split
    train_x = train[:,1:]
    train_y = train[:,0]
    test_x = test[:,1:]
    test_y = test[:,0]
    
    pca_values = [1,4,8,12,16,20]
    log_pca = []
    log = []
    for i in pca_values:
    # For logistic regression classifier with PCA
        # Principal components
        pca = PCA(n_components=i)
        train_x_p = pca.fit_transform(train_x)
    
        # Train model
        logisticRegr = LogisticRegression()
        logisticRegr.fit(train_x_p, train_y)
    
        # Predict
        y_log_pca = logisticRegr.predict(pca.transform(test_x))
        log_pca.append(1-accuracy_score(test_y,y_log_pca))
        
    # For logistic regression classifier without PCA
        # Train model
        logisticRegr2 = LogisticRegression()
        logisticRegr2.fit(train_x, train_y)
    
        # Predict
        y_log = logisticRegr2.predict(test_x)
        log.append(1-accuracy_score(test_y,y_log))
    
    x_label = np.array(np.arange(0,21,4))   
    # Figure 2
    plt.figure(2)
    # Plot the dataset in red and best-fit line in blue
    plt.plot(x_label,log_pca, '-', x_label, log, '-')
    plt.xlabel('PCA Value')
    plt.ylabel('Error')
    plt.show()
    plt.show()
    
if __name__ == '__main__':
    model = sys.argv[1]
    model = best_model = "pcaknn"
    train = sys.argv[2]
    test = sys.argv[3]
    
    # Load train data from a text file 
    train = np.loadtxt(train, dtype=float)

    # Load test data from a text file
    test = np.loadtxt(test, dtype=float)
    
    if model == "knn":
        print(knn(train, test))
    elif model == "net":
        print(neural_net(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    elif model == "pcaLG":
        print(pca_LG(train, test))
    elif model == "LG":
        print(LogistRegres(train, test))
    else:
        print("Invalid method selected!")
