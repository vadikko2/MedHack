import sys
sys.path.append('/home/vadim/hackatones/medhack/src/')
import regression
import recognission
import parse
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import copy
def distance(y_pred, y_true, **kwargs):
    if kwargs['method'] == 'euclidean':
        dist = euclidean_distances([y_pred], [y_true])
    if kwargs['method'] == 'manhattan':
        dist = manhattan_distances([y_pred], [y_true])
    #if kwargs['method'] == 'manhattan':
        #dist = manhattan_distances(y_pred, y_true)
    return dist
def midl(arr):
    size = len(arr)
    print(len(arr))
    m = [0,0,0]
    for r in arr:
        m[0]+=r[0]
        m[1]+=r[1]
        m[2]+=r[2]
    #    m[3]+=r[3]
    #    m[4]+=r[4]
    m[0]/=size
    m[1]/=size
    m[2]/=size
    #m[3]/=size
    #m[4]/=size
    return m

def return_data(path_name, d):
    try:
        os.chdir('../data/')
    except:
        print('Неправильная структура файлов. Отсутсвует директория ../data/')
        exit()
    _list = os.listdir()
    if(len(_list) == 0):
        print('Папка с выборкой пуста')
        exit()
    i = 0
    for l in _list:
        if  l == path_name:
            p = parse.Parser('/home/vadim/hackatones/medhack/src/ftp_server/data/')
            data = p.parse_path(100)
            #print(len(data))
            #p.delete_from_back(500)
            p.edit_features()
            split_data = p.get_split_database(200, 1)
            y_pred_x = []
            y_true_x = []
            y_pred_y = []
            y_true_y = []
            y_pred_z = []
            y_true_z = []
            dist_x = [[]]
            dist_y = [[]]
            dist_z = [[]]
            y_svm = []
            name = None
            marks = None
            i = 0
            for i in range(len(split_data)):
                name = split_data[i]['person_info']['name']
                y_tmp_pred_x, y_tmp_true_x = regression.predict(split_data[i],'x',['arctn'],False, False)
                y_pred_x.append(y_tmp_pred_x)
                y_true_x.append(y_tmp_true_x)

                y_tmp_pred_y, y_tmp_true_y = regression.predict(split_data[i],'y',['arctn'],False, False)
                y_pred_y.append(y_tmp_pred_y)
                y_true_y.append(y_tmp_true_y)

                y_tmp_pred_z, y_tmp_true_z = regression.predict(split_data[i],'z',['arctn'],False, False)
                y_pred_z.append(y_tmp_pred_z)
                y_true_z.append(y_tmp_true_z)
            #y_pred = np.asarray(y_pred)
            #y_true = np.asarray(y_true)
            dist_x = distance(y_pred_x, y_true_x, method = 'euclidean')
            dist_y = distance(y_pred_y, y_true_y, method = 'euclidean')
            dist_z = distance(y_pred_z, y_true_z, method = 'euclidean')
            print('d_x =', dist_x)
            print('d_y =', dist_y)
            print('d_z =', dist_z)
            if (dist_x[0][0] > d[0]) or (dist_y[0][0] > d[1]) or (dist_z[0][0] > d[2]):
                for sp in split_data:
                    pr, marks = recognission.predict(sp)
                    y_svm.append(pr)
                y_svm = midl(y_svm[0])
        i+=1
        return name, [dist_x[0][0], dist_y[0][0], dist_z[0][0]], y_svm, marks
if __name__ == "__main__":
    print(return_data('sample', [30,60,50]))
    '''del y_pred
    y_pred = []
    del y_true
    y_pred = []
    i = 0'''
