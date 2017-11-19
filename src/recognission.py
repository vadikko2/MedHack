import numpy as np
import matplotlib.pyplot as plt
import parse,sys,regression

import sklearn.cluster as sk_cl

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs

import copy
def get_only_coordinate_pathology(dataset, params, pathology ,needMeta = False):
    x = []
    y = []
    mark = []
    for i in range(len(dataset)):
        if (dataset[i]['person_info']['pathology'] != 'none') and (dataset[i]['walk_info']['gait'] == 0) and (dataset[i]['person_info']['trauma'] == 'none'):
            #if dataset[i]['person_info']['pathology'] == 1:
            #    print(dataset[i]['person_info']['pathology'])
            tmp = []
            if needMeta:
                tmp.append(dataset[i]['person_info']['age'])
                tmp.append(dataset[i]['person_info']['gender'])
                tmp.append(dataset[i]['person_info']['height'])
                tmp.append(dataset[i]['person_info']['feet size'])
                tmp.append(dataset[i]['walk_info']['gait'])

                tmp.append(dataset[i]['walk_info']['weight'])

                for j in range(len(dataset[i]['data']) - 1):
                    tmp.append(dataset[i]['data'][j]['x'])
                    tmp.append(dataset[i]['data'][j]['y'])
                    tmp.append(dataset[i]['data'][j]['z'])


                x.append(np.asarray(regression.preprocess(tmp, params)))
            else:
                for j in range(len(dataset[i]['data']) - 1):
                    tmp.append(dataset[i]['data'][j]['x'])
                    tmp.append(dataset[i]['data'][j]['y'])
                    tmp.append(dataset[i]['data'][j]['z'])

                x.append(tmp)

            #y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])

            mark.append(pathology.index(dataset[i]['person_info']['pathology']))

    return x, y, mark
def get_only_coordinate_pathology_predict(dataset, params, pathology ,needMeta = False):
    x = []
    y = []
    mark = []

    tmp = []

    if needMeta:
        tmp.append(dataset['person_info']['age'])
        tmp.append(dataset['person_info']['gender'])
        tmp.append(dataset['person_info']['height'])
        tmp.append(dataset['person_info']['feet size'])
        tmp.append(dataset['walk_info']['gait'])

        tmp.append(dataset['walk_info']['weight'])

        for j in range(len(dataset['data']) - 1):
            tmp.append(dataset['data'][j]['x'])
            tmp.append(dataset['data'][j]['y'])
            tmp.append(dataset['data'][j]['z'])


        x.append(np.asarray(regression.preprocess(tmp, params)))
    else:
        for j in range(len(dataset['data']) - 1):
            tmp.append(dataset['data'][j]['x'])
            tmp.append(dataset['data'][j]['y'])
            tmp.append(dataset['data'][j]['z'])

        x.append(tmp)
    return x, y, mark

def predict(v):
    #clf = OneVsRestClassifier(LinearSVC(random_state=0))
    from sklearn.externals import joblib
    clf = joblib.load('/home/vadim/hackatones/medhack/src/svm_weights.pkl')
    x, _, _ = get_only_coordinate_pathology_predict(v, [], [], needMeta = False)
    y = clf.predict_proba(x)
    return y, ['неправильная осанка (возможен сколеоз)','I степень плоскостопия','II степень плоскостопия']

if __name__ == "__main__":
    p = parse.Parser(sys.argv[1])
    p.parse_path(100)
    p.delete_from_back(500)
    dataset = p.get_split_database(200, 50)
    p.edit_features()

    meta = True

    pathology = []
    for fd in dataset:
    	temp = fd['person_info']['pathology']
    	if not temp in pathology or not temp == 'none':
    		pathology.append(temp)

    #print(pathology)

    #x_none, _ = get_only_coordinate_none_pathology('x', dataset, [], needMeta = meta)
    x, _, y = get_only_coordinate_pathology(dataset, [], pathology, needMeta = False)

    #print(len(x_none), len(x))

    #y_none = [1 for x in range(0,len(x_none))]

    #all_data = copy.deepcopy(x_none) + copy.deepcopy(x)
    #all_y = copy.deepcopy(y_none) + copy.deepcopy(y)
    #all_y = clustering(all_data, method = 'kmeans', n_clusters = 10, random_state = 0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    #tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = OneVsRestClassifier(SVC(kernel = 'linear',probability = True, C=1))
    #clf = OneVsRestClassifier(LinearSVC(random_state=0))
    clf.fit(x_train, y_train)
    print ("One vs rest accuracy: %.3f"  % clf.score(x_test,y_test))
    from sklearn.externals import joblib
    joblib.dump(clf, '/home/vadim/hackatones/medhack/src/svm_weights.pkl')
    #viewData([x_none,x], [len(x_none[0]), len(x[0])], [y_none, y])
    #viewData([all_data], [len(all_data[0])], [all_y])
    '''
    x_none = preprocessing.minmax_scale(x_none, feature_range=(0, 1))
    x = preprocessing.minmax_scale(x, feature_range=(0, 1))

    all_data = preprocessing.minmax_scale(all_data, feature_range=(0, 1))

    viewData([x_none,x], [len(x_none[0]), len(x[0])], [y_none, y])
    viewData([all_data], [len(all_data[0])], [all_y])
    '''
