import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()

data = scale(digits.data)
y = digits.target

k = len(np.unique(y))
k = 10

samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print("'{0}'\t'{1}'\t'{2}'\t'{3}'\t'{4}'\t'{5}'\t'{6}'"
          .format(name, estimator.inertia_,
                  metrics.homogeneity_score(y, estimator.labels_),
                  metrics.completeness_score(y, estimator.labels_),
                  metrics.v_measure_score(y, estimator.labels_),
                  metrics.adjusted_rand_score(y, estimator.labels_),
                  metrics.adjusted_mutual_info_score(y, estimator.labels_),
                  metrics.silhouette_score(data, estimator.labels_,
                                           metric='euclidean')))


myclassifier = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)

bench_k_means(myclassifier, "1", data)
