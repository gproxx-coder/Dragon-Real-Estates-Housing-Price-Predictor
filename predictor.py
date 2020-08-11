from joblib import load
import numpy as np

model = load("Dragon_original.joblib")

#feataures_scaled = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
#       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
#       -0.97491834,  0.41164221, -0.86091034]])

#predictions = model.predict(feataures_scaled)

#print(predictions)

#Original data label: 27.1
feataures_original = np.array([[0.14455, 12.5, 7.87, 0, 0.524, 6.172, 96.1, 5.9505, 5, 311, 15.2, 396.9, 19.15]])

#Original data label: 20.9
test = np.array([[0.12816, 12.5, 6.07, 0, 0.409, 5.885, 33, 6.498, 4, 345, 18.9, 396.9, 8.79]])

predictions2 = model.predict(test)

print(predictions2)