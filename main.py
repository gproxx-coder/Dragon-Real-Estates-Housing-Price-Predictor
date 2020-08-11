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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#importing DATA CSV file
housing = pd.read_csv("data.csv")


##########################################################
# Spilittng by using Scikit Learn StratifiedShuffleSplit #
##########################################################

## Sklearn - Stratified Shuffle Split for Equal Data Distribution (Train & Test)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

#Equal distribution of data by using for loop
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


#Copying Stratified Shuffled Data to Original Housing Dataset
#Here the Test data is safe in "strat_test_set" object. We will use after Deployement.
#Overwrite Original 506 values dataset with New 404 Strat Train Data

housing = strat_train_set.copy()

#Splitting the datasets into Features and labels:
housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()



########################################################
# Creating Pipeline (sklearn.pipeline import Pipeline) #
########################################################


#All the data will go through this pipeline so that it will be easy to implement new data
#when inserting new dataset to this Model.
    
    
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #Add as many as you want in pipeline
    #('std_scaler', StandardScaler())
])

#Following code Will Create NUMPY Array instead of Pandas Dataframe

housing_num_tr = my_pipeline.fit_transform(housing)



#####################################################
# Selecting a desired model for Dragon Real Estates #
#####################################################

#model = LinearRegression()

#model = DecisionTreeRegressor()

model = RandomForestRegressor()

model.fit(housing_num_tr, housing_labels)


## considering some data for testing (NOT Test_data_set)

some_data = housing.iloc[:10]
some_labels = housing_labels.iloc[:10]

## Checking how the predictions are

prepared_data = my_pipeline.transform(some_data)

print("Predicted Values:")
print(model.predict(prepared_data))

print("Original Values:")
print(list(some_labels))


## Evaulating the model (MSE and RMSE)

housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

print("\nRMSE: ", rmse)
print()
########################################################
# Using better evaluation technique - Cross Validation #
########################################################

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

 
def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    print(model.__class__)  #Printing Model Algorithm Name

print_scores(rmse_scores)


####################################
# Testing the Model with Test Data #
####################################
'''
test_features = strat_test_set.drop('MEDV', axis=1)
test_labels = strat_test_set['MEDV'].copy()

test_features_prepared = my_pipeline.transform(test_features)

final_predictions = model.predict(test_features_prepared)

final_mse = mean_squared_error(test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)

print("\nFinal RMSE: ", final_rmse)
'''

#####################################
# Now Dumping the Model (Exporting) #
#####################################

from joblib import dump, load
#dump(model, "Dragon_original.joblib")