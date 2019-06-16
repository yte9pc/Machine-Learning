

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random

# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country', 'label']
#        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
#                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
#                     'hours-per-week', 'native-country']
#        col_names_y = ['label']
#
#        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
#                          'hours-per-week']
#        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
#                            'race', 'sex', 'native-country']
        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.
        
        # Read in the csv file
        data = pd.read_csv(csv_fpath, header=None)
        data.columns = col_names
        
        # Dummy variables for categorical columns
        data = pd.get_dummies(data)
        
        # Drop last column        
        if  data.columns[-1] == 'label_ >50K':
            data.insert(81,'native-country_ Holand-Netherlands', 0)
            data = data.drop(data.columns[-1], axis=1)
        
        # Scale numerical cols [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = pd.DataFrame(min_max_scaler.fit_transform(data), columns=data.columns)

        x = data_scaled.iloc[:, :-1]
        y = data_scaled.iloc[:, -1]

        return x, y, data_scaled
    
    def fold(self, data, k):
        
        # Converts dataframe to np array
        data = data.as_matrix()
        
        # Splits the full data array into kfold
        datasplit = np.array_split(data, 3)
    
        # Testing set
        testing = datasplit[k]
        testing = pd.DataFrame(testing)
        
        # All the data but the testing set
        training = [x for i,x in enumerate(datasplit) if i!=k]
        
        # Concatenation of the (kfold - 1) training set
        training = np.vstack(training)
        training = pd.DataFrame(training)
       
        return training, testing

    def train_and_select_model(self, training_csv):
        x_train, y_train, data = self.load_data(training_csv)

        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
        param_set = [
#                     {'kernel': 'rbf', 'C': 1,'gamma': 0.10, 'cache_size': 1000, 'degree' : 3},
#                     {'kernel': 'rbf', 'C': 5,'gamma': 0.10, 'cache_size': 1000, 'degree' : 3},
#                     {'kernel': 'linear', 'C': 1, 'gamma': 0.10, 'cache_size': 1000, 'degree' : 3},
                     {'kernel': 'linear', 'C': 5, 'gamma': 0.10, 'cache_size': 1000, 'degree' : 3},
#                     {'kernel': 'poly', 'C': 1, 'gamma': 0.10, 'cache_size': 1000, 'degree' : 3},
#                     {'kernel': 'poly', 'C': 5, 'gamma': 0.10, 'cache_size': 1000, 'degree' : 5},
        ]
        # Accuracy for each model
        train_accuracy = []
        test_accuracy = []
        # For each hyper-parameter candidiate
        for kernel in param_set:
            classifer = SVC(**kernel)
            
            total_train = 0
            total_test = 0
            # 3 Fold CV
            for j in range(0,3):
                # Split data into test and train
                training, testing = self.fold(data, j)
                
                train_x = training.iloc[:, :-1]
                train_y = training.iloc[:, -1]
                
                test_x = testing.iloc[:, :-1]
                test_y = testing.iloc[:, -1]
                
                # Fit x and y to model
                classifer.fit(train_x, train_y)
                
                # Predict on train and test x
                train_prediction = classifer.predict(train_x)
                test_prediction = classifer.predict(test_x)
                
                # Accuracy 
                train_score = accuracy_score(train_y, train_prediction)
                test_score = accuracy_score(test_y, test_prediction)
                
                total_train += train_score
                total_test += test_score
                
            train_accuracy.append(total_train/3)
            test_accuracy.append(total_test/3)
        
        return classifer, test_accuracy[0]

    def predict(self, test_csv, trained_model):
        x_test, y_test, data = self.load_data(test_csv)
        predictions = trained_model.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 1:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print "The best model was scored %.2f" % cv_score
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)


