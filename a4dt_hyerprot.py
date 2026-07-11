from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

def load_data(data_path):
    data = np.load(data_path)
    x = data[:, :1024]
    y = data[:, 1024]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    return x_train, x_test, y_train, y_test

def optimize_hyperparameters(space, x_train, y_train, kf, max_evals=50):
    def objective_function(args):
        print(f"Hyperparameters: {args}")
        total_loss = 0
        for train_index, valid_index in kf.split(x_train):
            x_train_cv, x_valid_cv = x_train[train_index], x_train[valid_index]
            y_train_cv, y_valid_cv = y_train[train_index], y_train[valid_index]
            model = DecisionTreeClassifier(
                max_depth=args['max_depth'],
                max_features=args['max_features'],
                criterion=args['criterion'],
                min_samples_split=args['min_samples_split'],
                min_samples_leaf=args['min_samples_leaf'],
                max_leaf_nodes=args['max_leaf_nodes'],
                random_state=1,
                class_weight='balanced'
            )
            model.fit(x_train_cv, y_train_cv)
            val_preds = model.predict_proba(x_valid_cv)[:, 1]
            loss = 1 - roc_auc_score(y_valid_cv, val_preds)
            total_loss += loss
        avg_loss = total_loss / kf.get_n_splits()
        return {'loss': avg_loss, 'status': STATUS_OK}

    trials = Trials()
    best_results = fmin(objective_function, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best_results

data_path = "traindata_gaussian.npy"
x_train, x_test, y_train, y_test = load_data(data_path)

space_dt = {
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 5)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'min_samples_split': hp.choice('min_samples_split', range(2, 20)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 20)),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 20))
}

kf = KFold(n_splits=10)

best_results_cv = optimize_hyperparameters(space_dt, x_train, y_train, kf, max_evals=50)

# Convert indices to actual values
best_params = {
    'max_depth': best_results_cv['max_depth'] + 1,  # Adjust for 0-based index
    'max_features': best_results_cv['max_features'] + 1,  # Adjust for 0-based index
    'criterion': ["gini", "entropy"][best_results_cv['criterion']],
    'min_samples_split': best_results_cv['min_samples_split'] + 2,  # Adjust for 0-based index
    'min_samples_leaf': best_results_cv['min_samples_leaf'] + 1,  # Adjust for 0-based index
    'max_leaf_nodes': best_results_cv['max_leaf_nodes'] + 2  # Adjust for 0-based index
}

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

best_model_cv = DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    criterion=best_params['criterion'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    random_state=1,
    class_weight='balanced'
)

best_model_cv.fit(x_train, y_train)
joblib.dump(best_model_cv, "dt.pkl")
