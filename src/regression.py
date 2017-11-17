from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import parse
import math

def get_only_coordinate(coord, dataset, params):
    x = []
    y = []
    #y_X = []
    #y_Y = []
    #y_Z = []
    for i in range(len(dataset)):
        if (dataset[i]['person_info']['pathology'] == 'none') and(dataset[i]['walk_info']['gait'] == 0) and (dataset[i]['person_info']['trauma'] == 'none'):
            tmp = []
            #print(dataset[i]['person_info']['gender'])
            tmp.append(dataset[i]['person_info']['age'])
            tmp.append(dataset[i]['person_info']['gender'])
            tmp.append(dataset[i]['person_info']['height'])
            tmp.append(dataset[i]['person_info']['feet size'])
            tmp.append(dataset[i]['walk_info']['gait'])
            #tmp.append(dataset[i]['walk_info']['footWear'])
            #tmp.append(dataset[i]['walk_info']['hunger'])
            tmp.append(dataset[i]['walk_info']['weight'])
            for j in range(len(dataset[i]['data']) - 1):
                tmp.append(dataset[i]['data'][j][coord])
                #tmp.append(dataset[i]['data'][j]['y'])
                #tmp.append(dataset[i]['data'][j]['z'])
            x.append(np.asarray(preprocess(tmp, params)))
            y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])
            #y_Y.append(dataset[i]['data'][len(dataset[i]['data']) - 1]['y'])
            #y_Z.append(dataset[i]['data'][len(dataset[i]['data']) - 1]['z'])
    return x, y

def preprocess(x, _list_params):
    tmpSin = []
    x_tmp = x
    tmpCos = []
    tmpSqr = []
    tmp3 = []
    tmpSqrt = []
    tmpLog = []
    tmpTn = []
    tmpArctn = []
    tmpSigm = []
    x = np.asarray(x)
    x_mm = preprocessing.minmax_scale(x, feature_range=(-0.5, 0.5))
    #print(x_mm)
    for v in x_mm:
        if 'sin' in _list_params:
            tmpSin.append(math.asin(v))
        if 'cos' in _list_params:
            tmpCos.append(math.acos(v))
        if 'sigm' in _list_params:
            tmpSigm.append(1/ (1 + math.exp(-v)))
        if 'sqr' in _list_params:
            tmpSqr.append(v**2)
        if '^3' in _list_params:
            tmp3.append(v**3)
        if 'tn' in _list_params:
            tmpTn.append(math.tan(v))
        if 'arctn' in _list_params:
            tmpArctn.append(math.atan(v))
        #if 'log10' in _list_params:
        #    tmpLog.append(math.log10(v))
    x_tmp+=tmpSigm+tmpCos+tmpSin+tmpTn+tmp3+tmpSqr+tmp3+tmpArctn+tmpTn
    return x_tmp

def training(dataset, coordinate, params, l2, minmax):
    x, y = get_only_coordinate(coordinate, dataset, params)
    x = np.asarray(x)
    if l2:
        x = preprocessing.normalize(x, norm = 'l2')
    if minmax:
        x = preprocessing.minmax_scale(x, feature_range=(0, 1))
    y = np.asarray(y)
    x_train = x[:int((len(x))*0.7)]
    y_train = y[:int((len(y))*0.7)]

    x_test = x[int((len(x))*0.7)+1:]
    y_test = y[int((len(y)*0.7))+1:]

    # Create linear regression object

    regr = linear_model.LinearRegression()#Ridge (alpha = .5)
    # Train the model using the training sets
    regr.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients : \n', regr.coef_)
    # The mean squared error
    print("Mean squared error : %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score : %.2f' % r2_score(y_test, y_pred))

    #x_train = x[:int((len(x)/2)*0.7)]
    #y_train = y[:int((len(y)/2)*0.7)]

    #x_test = x[int((len(x)/2)*0.7)+1:int((len(x)/2))]
    #y_test = y[int((len(y)/2)*0.7)+1:int((len(y)/2))]

    #x_test = x[int((len(x)/2))+1:]
    #y_test = y[int((len(y)/2))+1:]


if __name__ == "__main__":
    p = parse.Parser('/home/vadim/hackatones/medhack/data/')
    p.parse_path(100)
    p.delete_from_back(500)
    dataset = p.get_split_database(200)
    print(len(dataset))
    p.edit_features()
    training(dataset, 'x', [], False, False)
    training(dataset, 'y', [], False, False)
    training(dataset, 'z', [], False, False)
