from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import parse
import math

def preprocess(x):
    tmpSin = []
    tmpCos = []
    tmpSqr = []
    tmp3 = []
    tmpSqrt = []
    tmpLog = []
    tmpTn = []
    tmpCtn = []
    tmpSigm = []
    for v in x:
        tmpSigm.append(1 / (1 + math.exp(-v)))
        tmpSqr.append(v**2)
        tmp3.append(v**3)
    x+=tmpSigm+tmpSqr+tmp3
    return x


p = parse.Parser('/home/vadim/hackatones/medhack/data/')
p.parse_path(100)
p.delete_from_back(20)
dataset = p.get_split_database(11)
x = []
y_X = []
y_Y = []
y_Z = []
for i in range(len(dataset)):
    if (dataset[i]['person_info']['pathology'] == 'none'):
        tmp = []
        tmp.append(int(dataset[i]['person_info']['age']))
        tmp.append(1. if dataset[i]['person_info']['gender'] == 'male' else 0.)
        tmp.append(int(dataset[i]['person_info']['height']))
        tmp.append(int(dataset[i]['person_info']['feet size']))
        for j in range(len(dataset[i]['data']) - 1):
            tmp.append(dataset[i]['data'][j]['x'])
            tmp.append(dataset[i]['data'][j]['y'])
            tmp.append(dataset[i]['data'][j]['z'])
        x.append(np.asarray(preprocess(tmp)))
        y_X.append(dataset[i]['data'][len(dataset[i]['data']) - 1]['x'])
        y_Y.append(dataset[i]['data'][len(dataset[i]['data']) - 1]['y'])
        y_Z.append(dataset[i]['data'][len(dataset[i]['data']) - 1]['z'])

#x = preprocessing.normalize(x, norm = 'l2')
x = np.asarray(x)
y_X = np.asarray(y_X)
y_Y = np.asarray(y_Y)
y_Z = np.asarray(y_Z)

x_train = x[:int((len(x)/2)*0.7)]
y_X_train = y_X[:int((len(y_X)/2)*0.7)]
y_Y_train = y_Y[:int((len(y_X)/2)*0.7)]
y_Z_train = y_Z[:int((len(y_X)/2)*0.7)]


x_test = x[int((len(x)/2)*0.7)+1:int((len(x)/2))]
y_X_test = y_X[int((len(y_X)/2)*0.7)+1:int((len(y_X)/2))]
y_Y_test = y_Y[int((len(y_Y)/2)*0.7)+1:int((len(y_Y)/2))]
y_Z_test = y_Z[int((len(y_Z)/2)*0.7)+1:int((len(y_Z)/2))]

'''
x_test = x[int((len(x)/2))+1:]
y_X_test = y_X[int((len(y_X)/2))+1:]
y_Y_test = y_Y[int((len(y_Y)/2))+1:]
y_Z_test = y_Z[int((len(y_Z)/2))+1:]
'''



# Create linear regression object
regrX = linear_model.Ridge(.5)
# Train the model using the training sets
regrX.fit(x_train, y_X_train)
# Make predictions using the testing set
y_X_pred = regrX.predict(x_test)

# The coefficients
print('Coefficients (X): \n', regrX.coef_)
# The mean squared error
print("Mean squared error (X): %.2f"
      % mean_squared_error(y_X_test, y_X_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score (X): %.2f' % r2_score(y_X_test, y_X_pred))






# Create linear regression object
regrY = linear_model.Ridge(.5)
# Train the model using the training sets
regrY.fit(x_train, y_Y_train)
# Make predictions using the testing set
y_Y_pred = regrY.predict(x_test)

# The coefficients
print('Coefficients (Y): \n', regrY.coef_)
# The mean squared error
print("Mean squared error (Y): %.2f"
      % mean_squared_error(y_Y_test, y_Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score (Y): %.2f' % r2_score(y_Y_test, y_Y_pred))







# Create linear regression object
regrZ = linear_model.Ridge(.5)
# Train the model using the training sets
regrZ.fit(x_train, y_Z_train)
# Make predictions using the testing set
y_Z_pred = regrZ.predict(x_test)

# The coefficients
print('Coefficients (Z): \n', regrZ.coef_)
# The mean squared error
print("Mean squared error (Z): %.2f"
      % mean_squared_error(y_Z_test, y_Z_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score (Z): %.2f' % r2_score(y_Z_test, y_Z_pred))
