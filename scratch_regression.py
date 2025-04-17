import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

# ==== Data Loading and Preparation ==== 

def load_gdsc(
    path: str = None,
    dataset_name: str = "samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc",
    excluded_columns: list = None,
) -> pd.DataFrame:
    """
    Load or download the GDSC dataset, drop missing values, and winsorize LN_IC50.
    Returns the cleaned DataFrame.
    """
    if excluded_columns is None:
        excluded_columns = [
            "LN_IC50", "AUC", "Z_SCORE",
            "DRUG_ID", "COSMIC_ID", "DRUG_NAME", "CELL_LINE_NAME",
        ]
    # Load
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        archive = kagglehub.dataset_download(dataset_name)
        df = pd.read_csv(os.path.join(archive, "GDSC_DATASET.csv"))
    # Clean
    df = df.dropna()
    # Winsorize
    q1 = df["LN_IC50"].quantile(0.25)
    q3 = df["LN_IC50"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["LN_IC50"] = df["LN_IC50"].clip(lower=lower, upper=upper)
    return df


def prepare_features(
    df: pd.DataFrame,
    excluded_columns: list = None,
    encode_dummies: bool = True,
    drop_first: bool = False,
) -> (pd.DataFrame, pd.Series):
    """
    From cleaned DataFrame, produce X and y.
    If encode_dummies: one-hot encode categoricals,
    else: label-encode each categorical column.
    """
    if excluded_columns is None:
        excluded_columns = [
            "LN_IC50", "AUC", "Z_SCORE",
            "DRUG_ID", "COSMIC_ID", "DRUG_NAME", "CELL_LINE_NAME",
        ]
    y = df["LN_IC50"].reset_index(drop=True)
    X = df.drop(columns=excluded_columns).reset_index(drop=True)
    # identify categoricals
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if encode_dummies:
        X = pd.get_dummies(X, columns=cats, drop_first=drop_first)
    else:
        for col in cats:
            X[col] = pd.factorize(X[col])[0]
    return X, y


def split_data(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split indices once, returning train/test sets
    for both feature DataFrames X1 and X2, and y.
    """
    n = len(y)
    idx = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(idx)
    split = int((1 - test_size) * n)
    train_idx, test_idx = idx[:split], idx[split:]
    return (
        X1.iloc[train_idx], X1.iloc[test_idx],
        X2.iloc[train_idx], X2.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx]
    )

# ==== Scratch Models ==== 

class LinearRegressionScratch:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_mat = np.array(X, dtype=float)
        y_vec = np.array(y, dtype=float)
        if self.fit_intercept:
            X_mat = np.hstack((np.ones((X_mat.shape[0], 1)), X_mat))
        beta = np.linalg.pinv(X_mat.T.dot(X_mat)).dot(X_mat.T.dot(y_vec))
        if self.fit_intercept:
            self.intercept_, self.coef_ = beta[0], beta[1:]
        else:
            self.intercept_, self.coef_ = 0.0, beta
        return self

    def predict(self, X):
        return np.array(X).dot(self.coef_) + self.intercept_


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth or np.inf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_ = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature, self.threshold = feature, threshold
            self.left, self.right, self.value = left, right, value

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.max_features = self.max_features or X.shape[1]
        self.tree_ = self._build_tree(X, y, 0)
        return self

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return DecisionTreeRegressorScratch.Node(value=y.mean())
        feat_idxs = np.random.choice(X.shape[1], self.max_features, replace=False)
        best = (None, None, np.inf)  # feat, thresh, mse
        for feat in feat_idxs:
            for thresh in np.unique(X[:, feat]):
                left, right = y[X[:, feat] <= thresh], y[X[:, feat] > thresh]
                if len(left)==0 or len(right)==0: continue
                mse = ((left-left.mean())**2).sum() + ((right-right.mean())**2).sum()
                if mse < best[2]: best = (feat, thresh, mse)
        feat, thresh, _ = best
        if feat is None:
            return DecisionTreeRegressorScratch.Node(value=y.mean())
        mask = X[:, feat] <= thresh
        left = self._build_tree(X[mask], y[mask], depth+1)
        right = self._build_tree(X[~mask], y[~mask], depth+1)
        return DecisionTreeRegressorScratch.Node(feat, thresh, left, right)

    def _predict_one(self, x, node):
        if node.value is not None: return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in np.array(X)])


class RandomForestRegressorScratch:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=None):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_split, self.max_features = min_samples_split, max_features
        self.random_state, self.trees = random_state, []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        n, f = X.shape
        mf = int(np.sqrt(f)) if self.max_features=='sqrt' else (self.max_features or f)
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n, n, True)
            tree = DecisionTreeRegressorScratch(self.max_depth, self.min_samples_split, mf)
            tree.fit(X[idxs], y[idxs]); self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.vstack([t.predict(X) for t in self.trees])
        return preds.mean(axis=0)


class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, max_features=None, random_state=None):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.max_features, self.random_state = max_features, random_state
        self.trees, self.init_pred = [], None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        self.init_pred = y.mean(); residual = y - self.init_pred
        mf = int(np.sqrt(X.shape[1])) if self.max_features=='sqrt' else (self.max_features or X.shape[1])
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressorScratch(self.max_depth, self.min_samples_split, mf)
            tree.fit(X, residual); pred = tree.predict(X)
            residual -= self.lr * pred; self.trees.append(tree)
        return self

    def predict(self, X):
        y_pred = np.full(len(X), self.init_pred)
        for tree in self.trees: y_pred += self.lr * tree.predict(X)
        return y_pred


def visualize_results(models, X_test, y_test):
    metrics = {}
    for name, m in models.items():
        p = m.predict(X_test)
        mse = ((p - y_test)**2).mean()
        r2 = 1 - ((p - y_test)**2).sum()/((y_test-y_test.mean())**2).sum()
        metrics[name] = {'MSE': mse, 'R2': r2}
    names = list(metrics)
    x = np.arange(len(names)); w=0.35
    plt.figure(); plt.bar(x-w/2, [metrics[n]['MSE'] for n in names], w)
    plt.bar(x+w/2, [metrics[n]['R2'] for n in names], w)
    plt.xticks(x, names, rotation=45); plt.legend(['MSE','R2']); plt.tight_layout(); plt.show()

    for name,m in models.items():
        p = m.predict(X_test)
        plt.figure(); plt.scatter(y_test,p); plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--')
        plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(name); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # Load & prepare\office
    df = load_gdsc()
    X_dummy, y = prepare_features(df, encode_dummies=True)
    X_label, _ = prepare_features(df, encode_dummies=False)
    Xd_tr, Xd_te, Xl_tr, Xl_te, y_tr, y_te = split_data(X_dummy, X_label, y)

    # Instantiate and fit
    models_lin = {'LinearRegression': LinearRegressionScratch()}
    models_tree = {
        'DecisionTree': DecisionTreeRegressorScratch(max_depth=3),
        'RandomForest': RandomForestRegressorScratch(n_estimators=10, max_depth=3, random_state=0),
        'GradientBoosting': GradientBoostingRegressorScratch(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=0)
    }
    for m in models_lin.values(): m.fit(Xd_tr, y_tr)
    for m in models_tree.values(): m.fit(Xl_tr, y_tr)

    # Visualize each
    print("=== Linear Regression on Dummy Features ===")
    visualize_results(models_lin, Xd_te, y_te)
    print("=== Tree-based on Label-Encoded Features ===")
    visualize_results(models_tree, Xl_te, y_te)
