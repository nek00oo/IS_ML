import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, minkowski

class ClassifierKNN:
    def __init__(self, n_neighbors=5, window_size=None, metric='minkowski_2', kernel='gaussian', a=1, b=1, weights=None):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.metric = metric
        self.kernel = kernel
        self.a = a
        self.b = b
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for i in range(X_test.shape[0]):
            x = X_test.iloc[i].values
            distances = np.array([self.distance(x, train_x) for train_x in self.X_train.values])
            if self.window_size is None:
                indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            else:
                indices = np.where(distances < self.window_size)[0]
                if len(indices) == 0:
                    raise ValueError(f"Нет объектов, подходящих под условия (радиус окна {self.window_size}) для объекта {i}.")

            nearest_distances = distances[indices]
            weights = self.calculate_weights(nearest_distances)
            votes = self.vote(indices, weights)
            predictions.append(votes)
        return pd.Series(predictions, index=X_test.index)

    def calculate_weights(self, distances):
        weights = []
        for dist in distances:
            if self.kernel == 'uniform':
                weights.append(1 if dist < 1 else 0)
            elif self.kernel == 'gaussian':
                weights.append((1 / np.sqrt(2 * np.pi)) * np.exp(-dist ** 2 / 2))
            elif self.kernel == 'epanechnikov':
                weights.append(3 / 4 * (1 - dist ** 2) if np.abs(dist) < 1 else 0)
            elif self.kernel == 'general':
                weights.append(max(0, (1 - abs(dist) ** self.a) ** self.b))
            else:
                raise ValueError("Неизвестное ядро")
        return np.array(weights)

    def vote(self, indices, weights):
        unique_classes = np.unique(self.y_train)
        class_votes = {label: 0 for label in unique_classes}

        for i, idx in enumerate(indices):
            label = self.y_train.iloc[idx]
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
