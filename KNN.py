import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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
                # if len(indices) == 0:
                #     raise ValueError(f"Нет объектов, подходящих под условия (радиус окна {self.window_size}) для объекта {i}.")

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
                weights.append(0.75 * (1 - dist ** 2) if np.abs(dist) < 1 else 0)
            elif self.kernel == 'general':
                weights.append(max(0, (1 - abs(dist) ** self.a) ** self.b))
            else:
                raise ValueError("Неизвестное ядро")
        return np.array(weights)

    def lowess_calculate_weights(self, proportion, kernel='gaussian'):
        if kernel == 'uniform':
            return 1 if np.abs(proportion) < 1 else 0
        elif kernel == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (proportion ** 2))
        elif kernel == 'epanechnikov':
            return 0.75 * (1 - proportion ** 2) if np.abs(proportion) < 1 else 0
        elif kernel == 'general':
            return (1 - np.abs(proportion) ** self.a) ** self.b
        else:
            raise ValueError("Неизвестное ядро")

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
            p = float(self.metric.split('_')[1])
            return minkowski(x1, x2, p)
        elif self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        else:
            raise ValueError("Неизвестная метрика")

    def lowess(self, kernel='gaussian'):
        lowess_weights = np.ones(len(self.X_train))

        for i in range(len(self.X_train)):
            x = self.X_train.iloc[i].values
            y = self.y_train.iloc[i]

            x_train_excluded = np.delete(self.X_train.values, i, axis=0)
            y_train_excluded = np.delete(self.y_train.values, i)

            distances = np.array([self.distance(x, train_x) for train_x in x_train_excluded])
            indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]

            nearest_labels = y_train_excluded[indices]

            amount_matching_neighbors = sum(1 for label in nearest_labels if label == y)

            if amount_matching_neighbors == 0:
                amount_matching_neighbors += np.finfo(float).eps

            prop = len(nearest_labels) / amount_matching_neighbors
            weight = self.lowess_calculate_weights(prop, kernel=kernel)
            if weight < 0:
                weight = np.finfo(float).eps
            lowess_weights[i] = weight

        return lowess_weights

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'window_size': self.window_size,
            'metric': self.metric,
            'kernel': self.kernel,
            'a': self.a,
            'b': self.b,
            'weights': self.weights
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self