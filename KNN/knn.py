# Machine Learning HW2-KNN

import numpy as np
import random

#file is just a filename, this method will just return a 2d array of the file contents
from sklearn.neighbors import KNeighborsClassifier

#this method should return a 2d array of the file contents
def read_csv(file):
    
    # Load data from a text file and ignore the header file
    data = np.loadtxt(file, dtype=float, skiprows=1)
    
    # Shuffle Data
    random.shuffle(data)
    return data 


# data is the full training numpy array
# k is the current iteration of cross validation
# kfold is the total number of cross validation folds
def fold(data, k, kfold):

    # Splits the full data array into kfold
    datasplit = np.array_split(data, kfold)

    # Testing set
    testing = datasplit[k]
    
    # All the data but the testing set
    training = [x for i,x in enumerate(datasplit) if i!=k]
    
    # Concatenation of the (kfold - 1) training set
    training = np.vstack(training)
   
    return training, testing

#training is the numpy array of training data (you run through each testing point and classify based on the training points)
#testing is a numpy array, use this method to predict 1 or 0 for each of the testing points
#k is the number of neighboring points to take into account when predicting the label
def classify(training, testing, k):
    
    # Training Labels
    training_labels = training[:,-1:]
    
    # Remove the labels from the testing and training set
    testing = testing[:,:-1]
    training = training[:,:-1]
    
    # Predicted Labels
    predicted_labels = []
    
    for i in range(len(testing)):

        # Array to store the distance between training and test, and label of the training
        distance_labels = []
       
        for j in range(len(training)):

            # Euclidian Distance Formula
            dis = np.linalg.norm(training[j]-testing[i])
            
            distance_labels.append([float(dis),int(training_labels[j])])

        # Sort and pick top k nearest neighbors
       
        sort = np.sort(distance_labels, axis=0)[:k]
        
        # Count number of 0s and 1s
        zero_count = int(np.count_nonzero(sort == 0, axis=0)[-1])
        one_count = int(np.count_nonzero(sort, axis=0)[-1])
        
        # Predict Label Value
        if zero_count > one_count:
            predicted_labels.append(0)
        elif zero_count < one_count:
            predicted_labels.append(1) 
        #print(predicted_labels)
    return predicted_labels


#predictions is a numpy array of 1s and 0s for the class prediction
#labels is a numpy array of 1s and 0s for the true class label
def calc_accuracy(predictions, labels):
    
    # Sum of number of entries that are for class
    correct = float(np.sum(predictions == labels))
    #print(predictions)
    #print(labels)
    # Length of class label
    length = len(labels)

    # Accuracy
    accuracy = correct/length

    return accuracy

def main():
    
    filename = "Movie_Review_Data.txt"
    #filename = "small.txt"
    kfold = 3
    k = raw_input("Provide an odd k value: ")
    while(not k.isdigit() or int(k)%2 == 0):
        k = raw_input("Provide an odd k value: ")
    k = int(k)
    sum = 0.0
    data = np.asarray(read_csv(filename), dtype=float)
    for i in range(0, kfold):
        training, testing = fold(data, i, kfold)
        predictions = classify(training, testing, k)
        labels = testing[:,-1]
        sum += calc_accuracy(predictions, labels)
        #print('sum', i, sum,)
    accuracy = sum / kfold
    print(accuracy)


if __name__ == "__main__":
    main()

