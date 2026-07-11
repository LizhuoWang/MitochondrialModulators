from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
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
            model = RandomForestClassifier(
                n_estimators=args['n_estimators'],
                max_depth=args['max_depth'],
                min_samples_leaf=args['min_samples_leaf'],
                max_features=args['max_features'],
                min_impurity_decrease=args['min_impurity_decrease'],
                n_jobs=6, random_state=1, verbose=0, class_weight='balanced'
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

space_rf = {
    'n_estimators': hp.choice('n_estimators', [10, 50, 100, 200, 300, 400, 500]),
    'max_depth': hp.choice('max_depth', list(range(3, 12))),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5, 10, 20, 50]),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.01),
    'max_features': hp.choice('max_features', [0.5, 0.6, 0.7, 0.8, 0.9])
}

n_splits = 10
kf = KFold(n_splits=n_splits)

best_results_cv = optimize_hyperparameters(space_rf, x_train, y_train, kf, max_evals=50)

# Convert indices to actual values
best_params = {
    'n_estimators': [10, 50, 100, 200, 300, 400, 500][best_results_cv['n_estimators']],
    'max_depth': list(range(3, 12))[best_results_cv['max_depth']],
    'min_samples_leaf': [1, 3, 5, 10, 20, 50][best_results_cv['min_samples_leaf']],
    'max_features': [0.5, 0.6, 0.7, 0.8, 0.9][best_results_cv['max_features']],
    'min_impurity_decrease': best_results_cv['min_impurity_decrease']
}

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

best_model_cv = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    min_impurity_decrease=best_params['min_impurity_decrease'],
    n_jobs=6, random_state=1, verbose=0, class_weight='balanced'
)

best_model_cv.fit(x_train, y_train)
joblib.dump(best_model_cv, "rf.pkl")