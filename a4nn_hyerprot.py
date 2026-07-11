# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:52:00 2022

@author: mi'to
"""


from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.neural_network import MLPClassifier
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
        total_loss = 0
        for train_index, valid_index in kf.split(x_train):
            x_train_cv, x_valid_cv = x_train[train_index], x_train[valid_index]
            y_train_cv, y_valid_cv = y_train[train_index], y_train[valid_index]
            model = MLPClassifier(
                solver=args['solver'], 
                activation=args['activation'],
                hidden_layer_sizes=args['hidden_layer_sizes'],
                alpha=args['alpha'],
                random_state=1
            )
            model.fit(x_train_cv, y_train_cv)
            val_preds = model.predict_proba(x_valid_cv)
            loss = 1 - roc_auc_score(y_valid_cv, val_preds[:, 1])
            total_loss += loss
        avg_loss = total_loss / kf.get_n_splits()
        return {'loss': avg_loss, 'status': STATUS_OK}

    trials = Trials()
    best_results = fmin(objective_function, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best_results

data_path = "traindata_gaussian.npy"
x_train, x_test, y_train, y_test = load_data(data_path)

space_nn = {
    'solver': hp.choice('solver', ['adam', 'sgd', 'lbfgs']),
    'alpha': hp.uniform('alpha', 0.1, 1.0),
    'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,), (100, 30)])
}

n_splits = 10
kf = KFold(n_splits=n_splits)

best_results_cv = optimize_hyperparameters(space_nn, x_train, y_train, kf, max_evals=50)

print("Best Hyperparameters:")
for key, value in best_results_cv.items():
    print(f"{key}: {value}")

best_model_cv = MLPClassifier(
    solver=['adam', 'sgd', 'lbfgs'][best_results_cv['solver']],
    activation=['identity', 'logistic', 'tanh', 'relu'][best_results_cv['activation']],
    hidden_layer_sizes=[(100,), (100, 30)][best_results_cv['hidden_layer_sizes']],
    alpha=best_results_cv['alpha'],
    random_state=1
) 

best_model_cv.fit(x_train, y_train)
joblib.dump(best_model_cv, "nn.pkl")


