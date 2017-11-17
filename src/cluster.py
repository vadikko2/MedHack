import numpy as np
import matplotlib.pyplot as plt
import parse,sys,regression

import sklearn.cluster as sk_cl

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, SpectralEmbedding

from sklearn.datasets.samples_generator import make_blobs

import copy

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
                    tmp.append(dataset[i]['data'][j][coord])

                x.append(tmp)
            
            y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])
    return x, y


def get_only_coordinate_pathology(coord, dataset, params, pathology ,needMeta = False):
    x = []
    y = []
    mark = []
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
                    tmp.append(dataset[i]['data'][j][coord])

                x.append(tmp)

            y.append(dataset[i]['data'][len(dataset[i]['data']) - 1][coord])

            mark.append(pathology.index(dataset[i]['person_info']['pathology']))

    return x, y, mark

def clustering(data,**kwargs):

	if kwargs['method'] == 'kmeans':
		kmeans = sk_cl.KMeans(n_clusters=kwargs['n_clusters'], random_state=kwargs['random_state']).fit(data)
		return kmeans.labels_

	elif kwargs['method'] == 'MiniBatchKMeans':
		mb_kmeans = sk_cl.MiniBatchKMeans(n_clusters = kwargs['n_clusters']).fit(data)
		return mb_kmeans.labels_

	elif kwargs['method'] == 'AffinityPropagation':
		ap = sk_cl.AffinityPropagation(preference=kwargs['preference']).fit(data)
		return ap.labels_

	elif kwargs['method'] == 'SpectralClustering':
		spectral = sk_cl.SpectralClustering(n_clusters=kwargs['n_clusters'], eigen_solver='arpack',affinity="nearest_neighbors").fit(data)
		return spectral.labels_

	elif kwargs['method'] == 'DBSCAN':
		dbscan = sk_cl.DBSCAN().fit(data)
		return dbscan.labels_

	elif kwargs['method'] == 'MeanShift':
		bandwidth = sk_cl.estimate_bandwidth(data, quantile=0.2)
		ms = sk_cl.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data)
		return ms.labels_

def viewData(all_data, components, mark = None, shape = None):

	plt.figure()
	x = 0
	count = 1

	x_d = 2
	y_d = 2

	if shape != None:
		x_d = shape[0]
		y_d = shape[1]
    
	for data in all_data:
		
		if components[x] != 2:
			data_1 = TSNE(n_components=2, random_state = 0).fit_transform(data)
			data_2 = SpectralEmbedding(n_components=2, random_state = 0).fit_transform(data)

			plt.subplot(x_d,y_d,count)
			plt.scatter(*np.transpose(data_1), c = mark[x])
			
			plt.subplot(x_d,y_d,count + 1)
			plt.scatter(*np.transpose(data_2), c = mark[x])
		
			count +=2

		else:
			plt.subplot(x_d,y_d,count)
			plt.scatter(*np.transpose(data), c = mark[x])
			
			count +=1

		x += 1

	plt.show()


p = parse.Parser(sys.argv[1])
p.parse_path(100)
p.delete_from_back(500)
dataset = p.get_split_database(200)
p.edit_features()

pathology = []
for fd in dataset:
	temp = fd['person_info']['pathology']
	if not temp in pathology:
		pathology.append(temp)

print(pathology)

x_none, _ = get_only_coordinate_none_pathology('x', dataset, ['arctn'], needMeta = False)
x, _, y = get_only_coordinate_pathology('x', dataset, ['arctn'], pathology, needMeta = False)

print(len(x_none), len(x))

y_none = [1 for x in range(0,len(x_none))] 

all_data = copy.deepcopy(x_none) + copy.deepcopy(x)
all_y = copy.deepcopy(y_none) + copy.deepcopy(y)

viewData([x_none,x], [len(x_none[0]), len(x[0])], [y_none, y])
viewData([all_data], [len(all_data[0])], [all_y])

x_none = preprocessing.normalize(x_none, norm = 'l2')
x = preprocessing.normalize(x, norm = 'l2')

all_data = preprocessing.normalize(all_data, norm = 'l2')

viewData([x_none,x], [len(x_none[0]), len(x[0])], [y_none, y])
viewData([all_data], [len(all_data[0])], [all_y])