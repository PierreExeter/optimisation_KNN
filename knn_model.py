import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
#from sklearn.model_selection import KFold
from sklearn import metrics

from convert_lib import bin_to_params

# LOAD DATA INTO DATAFRAME
train_df = pd.read_csv("input/train.csv")
test_df  = pd.read_csv("input/test.csv")

# DATA CLEANUP

# drop un-insightful columns
idx = test_df['PassengerId'].values
train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_df  = test_df.drop(['PassengerId', 'Name','Ticket','Cabin'], axis=1)

# feature engineering
# Gender: replace male/female with integers
mapping = {'male':1,'female':0}
train_df['Gender'] = train_df['Sex'].map(mapping).astype(int)
test_df['Gender'] = test_df['Sex'].map(mapping).astype(int)
train_df = train_df.drop(['Sex'], axis=1)    
test_df = test_df.drop(['Sex'], axis=1)

# Embarked: fill na and replace C/Q/S with integers
mapping = {'C':0,'Q':1,'S':2}
train_df['Embarked'] = train_df['Embarked'].fillna("S")
test_df['Embarked'] = test_df['Embarked'].fillna("S")

train_df['Embark'] = train_df['Embarked'].map(mapping).astype(int)
test_df['Embark'] = test_df['Embarked'].map(mapping).astype(int)
train_df = train_df.drop(['Embarked'], axis=1)    
test_df = test_df.drop(['Embarked'], axis=1)  

# Age: fill na with medium age
median_age = train_df['Age'].dropna().median()
train_df['Age'] = train_df['Age'].fillna(median_age)
test_df['Age'] = test_df['Age'].fillna(median_age)

# Fare: fill na with medium fare
median_fare = train_df['Fare'].dropna().median()
test_df['Fare'] = test_df['Fare'].fillna(median_fare)

# MACHINE LEARNING

def calc_objectives(model, train_data, test_data, predictors, outcome):
    """
    calculate cross validation and accuracy score
    model: eg. model = LogisticRegression()
    train data: training dataframe
    test_data: test dataframe    
    predictor: list of column labels used to train the model
    outcome: column label for the objective to reach
    """
    #Fit the model:
    model.fit(train_data[predictors], train_data[outcome])
  
    #Make predictions on training set: 
    predictions = model.predict(train_data[predictors])
  
    #calculate accuracy
    accuracy = metrics.accuracy_score(predictions, train_data[outcome])

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(train_data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (train_data[predictors].iloc[train,:])
 
        # The target we're using to train the algorithm.
        train_target = train_data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(train_data[predictors].iloc[test,:], train_data[outcome].iloc[test]))
 
    cross_val = np.mean(error) 

    return cross_val, accuracy

def cost_function(algorithm, leaf_size, n_neighbors, p, weights):
    """ 
    cost function for knn classifier
    arguments are the knn arguments
    returns the objectives to maximise: cross-validation and accuracy
    """    

    model = KNeighborsClassifier(algorithm = algorithm, 
                                 leaf_size = leaf_size, 
                                 metric = 'minkowski', 
                                 metric_params = None, 
                                 n_jobs = 1,
                                 n_neighbors = n_neighbors, 
                                 p = p, 
                                 weights = weights)  
 
    return calc_objectives(model, train_df, test_df, predictor_var, outcome_var)

def write_obj(obj, filename):
    """ write objective to files """
    
    f = open(filename, "w")
    f.write(str(obj))
    f.close()    


train_header = list(train_df.columns.values)
test_header = list(test_df.columns.values)

outcome_var = 'Survived'
predictor_var = test_header

algorithm, leaf_size, n_neighbors, p, weights = bin_to_params('knn_params.txt')

# print bin_to_params('best_params.txt')

obj1, obj2 = cost_function(algorithm, 
                           leaf_size, 
                           n_neighbors, 
                           p, 
                           weights)

write_obj(obj1, 'obj1.txt')
write_obj(obj2, 'obj2.txt')

