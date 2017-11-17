from sklearn.neighbors import DistanceMetric
import numpy as np
import matplotlib.pyplot as plt
import parse
import regression
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances#, chebyshev_distances
import sys

def get_only_coordinate_none_pathology(coord, dataset, params, needMeta = False):
    x = []
    y = []
    for i in range(len(dataset)):
        if (dataset[i]['person_info']['pathology'] == 'none') and (dataset[i]['walk_info']['gait'] == 0) and (dataset[i]['person_info']['trauma'] == 'none'):
            tmp = []
            if needMeta:
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
            
                x.append(np.asarray(regression.preprocess(tmp, params)))
            else:
                for j in range(len(dataset[i]['data']) - 1):
                    x.append(dataset[i]['data'][j][coord])
            
            y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])
    return x, y


def get_only_coordinate_pathology(coord, dataset, params, needMeta = False):
    x = []
    y = []
   
    for i in range(len(dataset)):
        if (dataset[i]['person_info']['pathology'] != 'none') and (dataset[i]['walk_info']['gait'] == 0) and (dataset[i]['person_info']['trauma'] == 'none'):
            tmp = []

            if needMeta:
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
            
                x.append(np.asarray(regression.preprocess(tmp, params)))
            else:
                for j in range(len(dataset[i]['data']) - 1):
                    x.append(dataset[i]['data'][j][coord])

            y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])

    return x, y
    
p = parse.Parser(sys.argv[1])
p.parse_path(100)
p.delete_from_back(500)
dataset = p.get_split_database(200)
p.edit_features()

x_none, _ = get_only_coordinate_none_pathology('x', dataset, ['arctn'], needMeta = True)
x, _ = get_only_coordinate_pathology('x', dataset, ['arctn'], needMeta = True)

print(len(x_none) , len(x))
x_tmp = x[::-1]
x_tmp_none = x_none[::-1]
points = []

'''Manhattan'''
plt.subplot(3,3,1)
for i in range(len(x)):
    points.append(manhattan_distances([x_none[i]], [x[i]]))
plt.scatter(range(len(points)), points)
plt.title('Manhattan 1-0')

point = []
plt.subplot(3,3,2)
for i in range(len(x)):
    points.append(manhattan_distances([x_none[i]], [x_tmp_none[i]]))
plt.scatter(range(len(points)), points)
plt.title('Manhattan 0-0')

point = []
plt.subplot(3,3,3)
for i in range(len(x)):
    points.append(manhattan_distances([x[i]], [x_tmp[i]]))
plt.scatter(range(len(points)), points)
plt.title('Manhattan 1-1')

'''Euclidean'''
plt.subplot(3,3,4)
for i in range(len(x)):
    points.append(euclidean_distances([x_none[i]], [x[i]]))
plt.scatter(range(len(points)), points)
plt.title('Euclidean 1-0')

point = []
plt.subplot(3,3,5)
for i in range(len(x)):
    points.append(euclidean_distances([x_none[i]], [x_tmp_none[i]]))
plt.scatter(range(len(points)), points)
plt.title('Euclidean 0-0')

point = []
plt.subplot(3,3,6)
for i in range(len(x)):
    points.append(euclidean_distances([x[i]], [x_tmp[i]]))
plt.scatter(range(len(points)), points)
plt.title('Euclidean 1-1')

'''ChebyshevDistance
plt.subplot(3,3,7)
for i in range(len(x)):
    points.append(chebyshev_distances([x_none[i]], [x[i]]))
plt.scatter(range(len(points)), points)
plt.title('ChebyshevDistance 1-0')

point = []
plt.subplot(3,3,8)
for i in range(len(x)):
    points.append(chebyshev_distances([x_none[i]], [x_tmp_none[i]]))
plt.scatter(range(len(points)), points)
plt.title('ChebyshevDistance 0-0')

point = []
plt.subplot(3,3,9)
for i in range(len(x)):
    points.append(chebyshev_distances([x[i]], [x_tmp[i]]))
plt.scatter(range(len(points)), points)
plt.title('ChebyshevDistance 1-1')
'''
plt.show()





#
