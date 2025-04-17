import numpy as np

class LinearRegressionScratch:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_mat = np.array(X, dtype=float)
        y_vec = np.array(y, dtype=float)
        if self.fit_intercept:
            ones = np.ones((X_mat.shape[0], 1))
            X_mat = np.hstack((ones, X_mat))
        XtX = X_mat.T.dot(X_mat)
        Xty = X_mat.T.dot(y_vec)
        beta = np.linalg.pinv(XtX).dot(Xty)
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X):
        X_mat = np.array(X, dtype=float)
        return X_mat.dot(self.coef_) + self.intercept_


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_ = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (n_samples < self.min_samples_split) or (depth >= self.max_depth):
            leaf_value = y.mean()
            return DecisionTreeRegressorScratch.Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.max_features, replace=False)
        best_feat, best_thresh, best_mse = None, None, np.inf
        for feat in feat_idxs:
            for t in np.unique(X[:, feat]):
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                mse = ((y_left - y_left.mean())**2).sum() + ((y_right - y_right.mean())**2).sum()
                if mse < best_mse:
                    best_mse, best_feat, best_thresh = mse, feat, t
        if best_feat is None:
            return DecisionTreeRegressorScratch.Node(value=y.mean())
        left = self._build_tree(X[X[:, best_feat] <= best_thresh], y[X[:, best_feat] <= best_thresh], depth+1)
        right = self._build_tree(X[X[:, best_feat] > best_thresh], y[X[:, best_feat] > best_thresh], depth+1)
        return DecisionTreeRegressorScratch.Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])


class RandomForestRegressorScratch:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            max_feats = self.max_features
        else:
            max_feats = n_features
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=max_feats)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=0)


class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        n_samples, n_features = X.shape
        self.init_pred = y.mean()
        residual = y - self.init_pred
        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            max_feats = self.max_features
        else:
            max_feats = n_features
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=max_feats)
            tree.fit(X, residual)
            pred = tree.predict(X)
            residual -= self.learning_rate * pred
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        y_pred = np.full(X.shape[0], self.init_pred, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


if __name__ == "__main__":
    # Sanity check with sklearn's Boston dataset
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression as SKLinearReg
    from sklearn.tree import DecisionTreeRegressor as SKTreeReg
    from sklearn.ensemble import RandomForestRegressor as SKForestReg
    from sklearn.ensemble import GradientBoostingRegressor as SKGBReg

    data = load_boston()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test scikit-learn models
    models = {
        'LinearRegression': SKLinearReg(),
        'DecisionTree': SKTreeReg(max_depth=3),
        'RandomForest': SKForestReg(n_estimators=10, max_depth=3, random_state=0),
        'GradientBoosting': SKGBReg(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=0)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name} — MSE: {mse:.4f}, R2: {r2:.4f}")
