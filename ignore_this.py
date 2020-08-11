## Dragon Real Estate House Price Predictor

################################
# importing all needed modules #
################################

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#importing DATA CSV file
housing = pd.read_csv("data.csv")

print("\nThe Original Dataset:\n")
print(housing.describe())

## Spilittng by using Scikit Learn Built-in Function:

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

## Sklearn - Stratified Shuffle Split for Equal Data Distribution (Train & Test)

#creating instance of StratifiedShuffleSplit Class and passing init arguments
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

#Equal distribution of data by using for loop
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("Stratified Shuffle Results **Train Set**: \n{}".format(strat_train_set['CHAS'].value_counts()))
print("\nNormal Shuffle Results **Train Set**: \n{}".format(train_set['CHAS'].value_counts()))


#Copying Stratified Shuffled Data to Original Housing Dataset
#Here the Test data is safe in "strat_test_set" object. We will use after Deployement.
#Overwrite Original 506 values dataset with New 404 Strat Train Data

housing = strat_train_set.copy()

#Splitting the datasets into Features and labels:
housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()

'''
## Missing Attributes (data):
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

# Taking Auto filled values into X object
X = imputer.transform(housing)

housing_tr = pd.DataFrame(X, columns=housing.columns)

'''

## Feature Scaling


########################################################
# Creating Pipeline (sklearn.pipeline import Pipeline) #
########################################################

#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler

#All the data will go through this pipeline so that it will be easy to implement new data
#when inserting new dataset to this Model.
    
#Creating a object of Pipeline class of sklearn.
    
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #Add as many as you want in pipeline
    ('std_scaler', StandardScaler())
])

#Creating NUMPY Array instead of Pandas Dataframe
#Predictos take Numpy array thats why made numpy array

#print("\nBefore fitting into Pipeline:\n")
#print(housing.shape)

housing_num_tr = my_pipeline.fit_transform(housing)

#print("\nAfter fitting into Pipeline:\n")
#print(housing_num_tr.shape)

#print(housing_num_tr[0])

#####################################################
# Selecting a desired model for Dragon Real Estates #
#####################################################

#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor

#model = LinearRegression()

#model = DecisionTreeRegressor()

model = RandomForestRegressor()

model.fit(housing_num_tr, housing_labels)


#+++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++#

## considering some data for testing (NOT Test_data_set)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

## Checking how the predictions are

prepared_data = my_pipeline.transform(some_data)

print("Predicted Values:")
print(model.predict(prepared_data))

print("Original Values:")
print(list(some_labels))


## Evaulating the model (MSE and RMSE)

from sklearn.metrics import mean_squared_error

housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

print(rmse)

########################################################
# Using better evaluation technique - Cross Validation #
########################################################

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print(rmse_scores)

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    print(model.__class__)  #Printing Model Algorithm

print_scores(rmse_scores)

#####################################
# Now Dumping the Model (Exporting) #
#####################################

from joblib import dump, load
#dump(model, "Dragon.joblib")