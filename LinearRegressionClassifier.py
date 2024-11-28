import numpy as np


class RidgeRegressionClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Должно быть ровно два уникальных класса.")

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
        """
        Линейный классификатор с градиентным спуском и Elastic Net регуляризацией.

        alpha: float, вес L1-регуляризации в Elastic Net (0 <= alpha <= 1)
        lambda_: float, коэффициент регуляризации
        learning_rate: float, шаг градиентного спуска
        max_iter: int, максимальное количество итераций
        loss: str, вид эмпирического риска ("linear", "squared", "logistic")
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss
        self.weights = None
        self.bias = None
        self.classes_ = None

    def _margin(self, X, y):
        """Вычисляет отступ M = y * (w^T * x + b)."""
        return y * (X @ self.weights + self.bias)

    def _loss_and_gradient(self, X, y):
        """Вычисляет значение эмпирического риска и его градиент."""
        margins = self._margin(X, y)
        n_samples, n_features = X.shape  # убрать

        if self.loss == "linear":
            loss_grad = -y * (margins < 0).astype(float)
        elif self.loss == "squared":
            loss_grad = -2 * y * np.maximum(0, 1 - margins)
        elif self.loss == "logistic":
            loss_grad = -y / (1 + np.exp(margins))
        else:
            raise ValueError("Неизвестная функция потерь: " + self.loss)

        # Градиент по весам
        dW = (loss_grad.values[:, np.newaxis] * X).mean(axis=0)
        # Градиент по смещению
        db = loss_grad.mean()

        # Elastic Net регуляризация
        dW += self.lambda_ * (self.alpha * np.sign(self.weights) + (1 - self.alpha) * 2 * self.weights)

        # Эмпирический риск
        if self.loss == "linear":
            loss = np.maximum(0, -margins).mean()
        elif self.loss == "squared":
            loss = np.maximum(0, 1 - margins) ** 2
        elif self.loss == "logistic":
            loss = np.log(1 + np.exp(-margins)).mean()

        loss += self.lambda_ * (
                    self.alpha * np.abs(self.weights).sum() + (1 - self.alpha) * np.linalg.norm(self.weights) ** 2)

        return loss, dW, db

    def fit(self, X, y):
        """
        Обучение модели.
        X: матрица признаков (n_samples, n_features)
        y: целевая переменная (n_samples,)
        """
        # Преобразование меток в бинарный формат
        self.classes_ = np.unique(y)  # Находим уникальные классы
        if len(self.classes_) != 2:
            raise ValueError("Метод поддерживает только бинарную классификацию.")

        # Преобразуем метки классов: один класс в -1, другой в +1
        y_binary = np.where(y == self.classes_[0], -1, 1)

        # Размеры данных
        n_samples, n_features = X.shape

        # Инициализация весов и смещения
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Градиентный спуск
        for i in range(self.max_iter):
            loss, dW, db = self._loss_and_gradient(X, y_binary)

            # Обновление весов и смещения
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            # Вывод текущей потери каждые 100 итераций
            if i % 100 == 0:
                print(f"Итерация {i}, Потеря: {loss}")

    def predict(self, X):
        """Предсказание меток."""
        y_pred = np.sign(X @ self.weights + self.bias)
        y_pred_labels = np.where(y_pred == -1, self.classes_[0], self.classes_[1])

        return y_pred_labels

    def predict_proba(self, X):
        """Предсказание вероятностей (для логистической функции потерь)."""
        if self.loss != "logistic":
            raise ValueError("Вероятности доступны только для логистической функции потерь.")
        margins = X @ self.weights + self.bias

        return 1 / (1 + np.exp(-margins))
