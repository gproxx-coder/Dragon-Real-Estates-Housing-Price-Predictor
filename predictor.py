from joblib import load
import numpy as np

model = load("Dragon_original.joblib")

features_list = np.array([], dtype='f')

features_input = [
    "per capita crime rate by town [CRIM]: ",
    "proportion of residential land zoned for lots over 25,000 sq.ft [ZN]: ",
    "proportion of non-retail business acres per town [INDUS]: ",
    "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) [CHAS]: ",
    "nitric oxides concentration (parts per 10 million) [NOX]: ",
    "average number of rooms per dwelling [RM]: ",
    "proportion of owner-occupied units built prior to 1940 [AGE]: ",
    "weighted distances to five Boston employment centres [DIS]: ",
    "index of accessibility to radial highways [RAD]: ",
    "full-value property-tax rate per $10,000 [TAX]: ",
    "pupil-teacher ratio by town [PTRATIO]: ",
    "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town [B]: ",
    "% lower status of the population [LSTAT]: "
]

for inp in features_input:
    temp = float(input(inp))
    features_list = np.append(features_list, temp)

features_list.resize(1,13)

#print(features_list)

#Original data price: 27.1
#feataures = np.array([[0.14455, 12.5, 7.87, 0, 0.524, 6.172, 96.1, 5.9505, 5, 311, 15.2, 396.9, 19.15]])

#Original data label: 20.9
#feataures = np.array([[0.12816, 12.5, 6.07, 0, 0.409, 5.885, 33, 6.498, 4, 345, 18.9, 396.9, 8.79]])

predictions_price = model.predict(features_list)

print()
print("Predicted Price is: ", predictions_price)