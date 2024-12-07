import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes

        self.tree = None
        self.label_to_index = None
        self.index_to_label = None
        self.tree_height = 0
        self.current_leaf_count = 0

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        prob = counts / m
        return 1 - np.sum(prob ** 2)

    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _find_best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_impurity = float("inf")
        best_split = None

        current_impurity = self._gini(y)
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_impurity = self._gini(y_left)
                right_impurity = self._gini(y_right)
                impurity = (len(y_left) * left_impurity + len(y_right) * right_impurity) / n_samples

                impurity_reduction = current_impurity - impurity
                if impurity < best_impurity and impurity_reduction >= self.min_impurity_decrease:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold
                    best_split = (X_left, X_right, y_left, y_right)

        return best_feature, best_threshold, best_split

    def _build_tree(self, X, y, depth):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split or \
                n_labels == 1:
            self.current_leaf_count += 1
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        if self.max_leaf_nodes is not None and self.current_leaf_count >= self.max_leaf_nodes:
            self.current_leaf_count += 1
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        feature, threshold, split = self._find_best_split(X, y)
        if feature is None:
            self.current_leaf_count += 1
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        X_left, X_right, y_left, y_right = split

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            self.current_leaf_count += 1
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        total_weight = len(y)
        if len(y_left) / total_weight < self.min_weight_fraction_leaf or \
                len(y_right) / total_weight < self.min_weight_fraction_leaf:
            self.current_leaf_count += 1
            return {"type": "leaf", "class": np.argmax(np.bincount(y))}

        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)
        self.tree_height = max(self.tree_height, depth + 1)

        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def fit(self, X, y):
        self.tree_height = 0
        self.current_leaf_count = 0

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        unique_labels, y_numeric = np.unique(y, return_inverse=True)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}

        self.tree = self._build_tree(X, y_numeric, 0)

    def _predict(self, row, tree):
        if tree["type"] == "leaf":
            return tree["class"]
        if row[tree["feature"]] <= tree["threshold"]:
            return self._predict(row, tree["left"])
        else:
            return self._predict(row, tree["right"])

    def predict(self, X_test):
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.to_numpy()

        y_pred_numeric = np.array([self._predict(row, self.tree) for row in X_test])
        return np.array([self.index_to_label[num] for num in y_pred_numeric])


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.trees = []
        self.label_to_index = None
        self.index_to_label = None

    def _sample_features(self, X):
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            size = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            size = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            size = self.max_features
        else:
            size = n_features
        indices = np.random.choice(n_features, size, replace=False)
        return indices

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        unique_labels, y_numeric = np.unique(y, return_inverse=True)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}

        self.trees = []

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[bootstrap_indices]
            y_sample = y_numeric[bootstrap_indices]

            feature_indices = self._sample_features(X_sample)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_impurity_decrease=self.min_impurity_decrease)
            tree.fit(X_sample[:, feature_indices], y_sample)

            self.trees.append((tree, feature_indices))

    def predict(self, X_test):
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.to_numpy()

        predictions = np.zeros((len(X_test), len(self.trees)))
        for i, (tree, feature_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X_test[:, feature_indices])

        y_pred_numeric = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x.astype(int))),
            axis=1,
            arr=predictions
        )

        return np.array([self.index_to_label[num] for num in y_pred_numeric])