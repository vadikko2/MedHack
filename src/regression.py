from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import parse
import math
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def get_only_coordinate(coord, dataset, params):
    x = []
    y = []
    #y_X = []
    #y_Y = []
    #y_Z = []
    for i in range(len(dataset)):
        if (dataset[i]['person_info']['pathology'] == 'none') and(dataset[i]['walk_info']['gait'] == 0) and (dataset[i]['person_info']['trauma'] == 'none'):
            tmp = []
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
            x.append(np.asarray(preprocess(tmp, params)))
            y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])

    return x, y

def get_only_coordinate_to_predict(coord, dataset, params):
    x = []
    y = []
    #y_X = []
    #y_Y = []
    #y_Z = []
    #print(dataset)
    tmp = []
    tmp.append(dataset['person_info']['age'])
    tmp.append(dataset['person_info']['gender'])
    tmp.append(dataset['person_info']['height'])
    tmp.append(dataset['person_info']['feet size'])
    tmp.append(dataset['walk_info']['gait'])
    #tmp.append(datase]['walk_info']['footWear'])
    #tmp.append(datase]['walk_info']['hunger'])
    tmp.append(dataset['walk_info']['weight'])
    for j in range(len(dataset['data']) - 1):
        tmp.append(dataset['data'][j][coord])
    x.append(np.asarray(preprocess(tmp, params)))
    y.append(dataset['data'][len(dataset['data']) - 1][coord])
    #print('y = ', dataset[i]['data'][len(dataset[i]['data']) - 1][coord])
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
    #print(x)
    y = np.asarray(y)
    '''
    x_train = x[:int((len(x))*0.7)]
    y_train = y[:int((len(y))*0.7)]

    x_test = x[int((len(x))*0.7)+1:]
    y_test = y[int((len(y)*0.7))+1:]
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    # Create linear regression object

    regr = linear_model.Ridge (alpha = .5)#Ridge (alpha = .5)
    # Train the model using the training sets
    regr.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    '''
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_pred = svr_rbf.fit(x_train, y_train).predict(x_test)'''
    # The coefficients
    #print('Coefficients : \n', regr.coef_)
    # The mean squared error
    #print("Mean squared error : %.2f"
    #      % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print(coordinate + ': Variance score : %.2f' % r2_score(y_test, y_pred))
    from sklearn.externals import joblib
    joblib.dump(regr, '/home/vadim/hackatones/medhack/src/model_regeression_'+coordinate+'.pkl')

    #x_train = x[:int((len(x)/2)*0.7)]
    #y_train = y[:int((len(y)/2)*0.7)]

    #x_test = x[int((len(x)/2)*0.7)+1:int((len(x)/2))]
    #y_test = y[int((len(y)/2)*0.7)+1:int((len(y)/2))]

    #x_test = x[int((len(x)/2))+1:]
    #y_test = y[int((len(y)/2))+1:]
def predict(dataset, coordinate, params, l2, minmax):
    x, y_true = get_only_coordinate_to_predict(coordinate, dataset, params)
    x = np.asarray(x)
    if l2:
        x = preprocessing.normalize(x, norm = 'l2')
    if minmax:
        x = preprocessing.minmax_scale(x, feature_range=(0, 1))
    #print(x)
    from sklearn.externals import joblib
    regr = joblib.load('/home/vadim/hackatones/medhack/src/model_regeression_'+coordinate+'.pkl')
    y_pred = regr.predict(x)
    return y_pred[0], y_true[0]

if __name__ == "__main__":
    p = parse.Parser('/home/vadim/hackatones/medhack/data/')
    p.parse_path(100)
    p.delete_from_back(500)
    dataset = p.get_split_database(200, 50)
    p.edit_features()
    training(dataset, 'x', ['arctn'], False, False)
    training(dataset, 'y', ['arctn'], False, False)
    training(dataset, 'z', ['arctn'], False, False)
    #print(dataset[0])
    #for i in range(len(dataset)):
    #    if (dataset[i]['person_info']['pathology'] == 'none') and(dataset[i]['walk_info']['gait'] == 0) and (dataset[i]['person_info']['trauma'] == 'none'):
    #        print(predict([dataset[i]], 'x', ['arctn'], False, False))
    #        print(predict([dataset[i]], 'y', ['arctn'], False, False))
    #        print(predict([dataset[i]], 'z', ['arctn'], False, False))
    #        exit()
