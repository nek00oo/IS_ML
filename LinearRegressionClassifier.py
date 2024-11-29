import numpy as np


class RidgeRegressionClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Метод поддерживает только бинарную классификацию.")

        y_binary = np.where(y == self.classes_[0], -1, 1)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        I = np.eye(X.shape[1])
        I[0, 0] = 0

        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y_binary

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        predictions = X @ self.weights

        return np.where(predictions >= 0, self.classes_[1], self.classes_[0])


class LinearClassifierGD:
    def __init__(self, alpha=0.5, lambda_=0.1, learning_rate=0.01, max_iter=1000, loss="logistic"):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss
        self.weights = None
        self.bias = None
        self.classes_ = None
        self.loss_history = []

    def _margin(self, X, y):
        return y * (X @ self.weights + self.bias)

    def _loss_and_gradient(self, X, y):
        margins = self._margin(X, y)

        if self.loss == "linear":
            loss_grad = -y * (margins < 0).astype(float)
        elif self.loss == "squared":
            loss_grad = -2 * y * np.maximum(0, 1 - margins)
        elif self.loss == "logistic":
            loss_grad = -y / (1 + np.exp(margins))
        else:
            raise ValueError("Неизвестная функция потерь: " + self.loss)

        dW = (loss_grad.values[:, np.newaxis] * X).mean(axis=0)
        db = loss_grad.mean()

        dW += self.lambda_ * (self.alpha * np.sign(self.weights) + (1 - self.alpha) * 2 * self.weights)

        if self.loss == "linear":
            loss = np.maximum(0, -margins).mean()
        elif self.loss == "squared":
            loss = (np.maximum(0, 1 - margins) ** 2).mean()
        elif self.loss == "logistic":
            loss = np.log(1 + np.exp(-margins)).mean()

        loss += self.lambda_ * (
                    self.alpha * np.abs(self.weights).sum() + (1 - self.alpha) * np.linalg.norm(self.weights) ** 2)

        return loss, dW, db

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Метод поддерживает только бинарную классификацию.")

        y_binary = np.where(y == self.classes_[0], -1, 1)

        n_samples, n_features = X.shape

        self.weights = np.ones(n_features)
        self.bias = 0
        self.loss_history = []

        for i in range(self.max_iter):
            loss, dW, db = self._loss_and_gradient(X, y_binary)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            self.loss_history.append(loss)

    def predict(self, X):
        y_pred = np.sign(X @ self.weights + self.bias)
        y_pred_labels = np.where(y_pred == -1, self.classes_[0], self.classes_[1])

        return y_pred_labels

    def _get_loss_history(self):
        return self.loss_history

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'lambda_': self.lambda_,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'loss': self.loss
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, learning_rate=0.01, max_iter=1000, degree=3, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.degree = degree
        self.gamma = gamma
        self.alpha = None  # Коэффициенты альфа для SVM с ядрами
        self.bias = 0
        self.classes_ = None
        self.X_train = None  # Сохраняем обучающие данные для работы с ядром
        self.loss_history = []

    def _linear_kernel(self, X1, X2):
        return X1 @ X2.T

    def _polynomial_kernel(self, X1, X2):
        return (X1 @ X2.T + 1) ** self.degree

    def _rbf_kernel(self, X1, X2):
        if self.gamma == 'scale':
            gamma = 1 / X1.shape[1]
        else:
            gamma = self.gamma

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        K = np.exp(-gamma * np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
        return K

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Неизвестное ядро: {self.kernel}")

    def _compute_loss(self, margins):
        return np.maximum(0, 1 - margins).mean()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Метод поддерживает только бинарную классификацию.")

        y_binary = np.where(y == self.classes_[0], -1, 1)

        self.alpha = np.zeros(n_samples)
        self.bias = 0
        self.X_train = X

        K = self._compute_kernel(X, X)

        for _ in range(self.max_iter):
            margins = y_binary * (K @ self.alpha + self.bias)
            loss_grad = np.where(margins < 1, -y_binary, 0)

            d_alpha = (K.T @ loss_grad) / n_samples + self.C * self.alpha
            d_bias = -loss_grad.mean()

            self.alpha -= self.learning_rate * d_alpha
            self.bias -= self.learning_rate * d_bias

            loss = self._compute_loss(margins)
            self.loss_history.append(loss)

    def predict(self, X):
        K_test = self._compute_kernel(X, self.X_train)
        margins = K_test @ self.alpha + self.bias
        predictions = np.sign(margins)
        return np.where(predictions == -1, self.classes_[0], self.classes_[1])

    def _get_loss_history(self):
        return self.loss_history

    def get_params(self, deep=True):
        return {
            'kernel': self.kernel,
            'C': self.C,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'degree': self.degree,
            'gamma': self.gamma
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


