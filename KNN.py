import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine, minkowski


class ClassifierKNN:
    def __init__(self, n_neighbors=5, window_size=None, metric='minkowski', kernel='gaussian', a=1, b=1, weights=None):
        self.neigh = None
        self.n_classes = None
        self.n_features = None
        self.y_train = None
        self.X_train = None
        self.n_samples = None
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.metric = metric
        self.kernel = kernel
        self.a = a
        self.b = b
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_classes = len(set(y))

        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.neigh.fit(X)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances, indices = self.neigh.kneighbors([x])
            distances, indices = distances[0], indices[0]
            weights = self.calculate_weights(distances)
            votes = self.vote(indices, weights)
            predictions.append(votes)
        return np.array(predictions)

    def calculate_weights(self, distances):
        if self.window_size:
            distances = distances / self.window_size
        weights = []
        for dist in distances:
            if self.kernel == 'uniform':
                weights.append(1 if dist <= 1 else 0)
            elif self.kernel == 'gaussian':
                weights.append(np.exp(-dist ** 2 / 2))
            elif self.kernel == 'epanechnikov':
                weights.append(max(0, 1 - dist ** 2))
            elif self.kernel == 'general':
                weights.append(max(0, (1 - abs(dist) ** self.a) ** self.b))
            else:
                raise ValueError("Неизвестное ядро")
        return np.array(weights)

    def vote(self, indices, weights):
        unique_classes = np.unique(self.y_train)
        class_votes = {label: 0 for label in unique_classes}

        for i, idx in enumerate(indices):
            label = self.y_train[idx]
            prior_weight = self.weights[label] if self.weights is not None else 1
            class_votes[label] += weights[i] * prior_weight

        return max(class_votes, key=class_votes.get)

    def distance(self, x1, x2):
        if self.metric == 'cosine':
            return cosine(x1, x2)
        elif self.metric.startswith('minkowski'):
            p = int(self.metric.split('_')[1])
            return minkowski(x1, x2, p)
        elif self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        else:
            raise ValueError("Неизвестная метрика")
