from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn import svm
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
            model = svm.SVC(**args, random_state=1, probability=True,
                            class_weight='balanced', cache_size=10000, max_iter=1000)
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

space_svm = {
    'C': hp.uniform('C', 0.1, 100), 
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']), 
    'gamma': hp.uniform('gamma', 0.0001, 2)
}
kernel_ls = ['linear', 'sigmoid', 'poly', 'rbf']

kf = KFold(n_splits=10)

best_results_cv = optimize_hyperparameters(space_svm, x_train, y_train, kf, max_evals=50)

# Convert indices to actual values
best_params = {
    'C': best_results_cv['C'],
    'gamma': best_results_cv['gamma'],
    'kernel': kernel_ls[best_results_cv['kernel']]
}

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

best_model_cv = svm.SVC(
    C=best_params['C'], 
    gamma=best_params['gamma'], 
    kernel=best_params['kernel'], 
    random_state=1, 
    probability=True,
    class_weight='balanced', 
    cache_size=10000, 
    max_iter=1000
)

best_model_cv.fit(x_train, y_train)
joblib.dump(best_model_cv, "svm.pkl")
