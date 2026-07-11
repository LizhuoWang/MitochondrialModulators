
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import statistics
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def load_data(data_path):

    data = np.load(data_path)
    x = data[:, :1024]
    y = data[:, 1024:]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=0)

    return x_train, x_test, y_train, y_test


def optimize_hyperparameters(space, x_train, y_train, max_evals=50):


    def objective_function(args):
        total_loss = 0
        for train_index, valid_index in kf.split(x_train):
            x_train_cv, x_valid_cv = x_train[train_index], x_train[valid_index]
            y_train_cv, y_valid_cv = y_train[train_index], y_train[valid_index]
            model = XGBClassifier(
                **args, n_jobs=6, tree_method='gpu_hist', gpu_id=0, random_state=1, seed=1)
            model.fit(x_train_cv, y_train_cv, eval_metric='auc',
                      eval_set=[(x_valid_cv, y_valid_cv)],
                      early_stopping_rounds=30, verbose=False)
            val_preds = model.predict_proba(
                x_valid_cv, ntree_limit=model.best_ntree_limit)
            loss = 1 - roc_auc_score(y_valid_cv, val_preds[:, 1])
            total_loss += loss
        avg_loss = total_loss / n_splits
        return {'loss': avg_loss, 'status': STATUS_OK}

    trials = Trials()
    best_results = fmin(objective_function, space,
                        algo=tpe.suggest, max_evals=max_evals, trials=trials)

    return best_results



# Load data
data_path = "traindata_gaussian.npy"
x_train, x_test, y_train, y_test = load_data(data_path)

# Define hyperparameter search space
space_ = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
          'gamma': hp.uniform('gamma', 0, 0.2),
          'min_child_weight': hp.choice('min_child_weight', range(1, 6)),
          'subsample': hp.uniform('subsample', 0.7, 1.0),
          'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
          'max_depth': hp.choice('max_depth', range(3, 10)),
          'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 1000, 1500, 2000])}
min_child_weight_ls = range(1, 6)
max_depth_ls = range(3, 10)
n_estimators_ls = [100, 200, 300, 400, 500, 1000, 1500, 2000]


# Define number of cross-validation folds
n_splits = 10
kf = KFold(n_splits=n_splits)

# Hyperparameter optimization
best_results_cv = optimize_hyperparameters(
    space_, x_train, y_train, max_evals=50)

# Print best hyperparameters
print("Best Hyperparameters:")
for key, value in best_results_cv.items():
    print(f"{key}: {value}")

# Use best hyperparameters from cross-validation
best_model_cv = XGBClassifier(n_estimators=n_estimators_ls[best_results_cv['n_estimators']],
                              max_depth=max_depth_ls[best_results_cv['max_depth']],
                              min_child_weight=min_child_weight_ls[best_results_cv['min_child_weight']],
                              learning_rate=best_results_cv['learning_rate'],
                              gamma=best_results_cv['gamma'],
                              subsample=best_results_cv['subsample'],
                              colsample_bytree=best_results_cv['colsample_bytree'],
                              tree_method='gpu_hist',  # Use GPU acceleration
                              gpu_id=0,  # Set GPU device ID
                              )

# Train the final model on the full training set
best_model_cv.fit(x_train, y_train)

# Save the trained model
joblib.dump(best_model_cv, "XGboost.pkl")


def evaluate_model_cv_collect_results(model, x_test, y_test, n_splits=10, n_repeats=2):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    all_val_preds_proba = []
    all_y_test = []
    all_conf_matrices = []
    all_fprs = []
    all_tprs = []
    all_precisions = []
    all_recalls = []
    auc_scores = []
    pr_auc_scores = []
    all_accuracies = []
    all_precisions_scores = []
    all_recalls_scores = []
    all_f1_scores = []

    for i, (train_index, test_index) in enumerate(rkf.split(x_test)):
        print(f"Evaluating times: {i + 1}")
        x_train_cv, x_test_cv = x_test[train_index], x_test[test_index]
        y_train_cv, y_test_cv = y_test[train_index], y_test[test_index]
        model.fit(x_train_cv, y_train_cv)
        val_preds_proba = model.predict_proba(x_test_cv)[:, 1]
        val_preds = model.predict(x_test_cv)

        conf_matrix = confusion_matrix(y_test_cv, val_preds)
        fpr, tpr, _ = roc_curve(y_test_cv, val_preds_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_cv, val_preds_proba)

        auc_score = roc_auc_score(y_test_cv, val_preds_proba)
        pr_auc_score = auc(recall_curve, precision_curve)

        accuracy = accuracy_score(y_test_cv, val_preds)
        precision_score_val = precision_score(y_test_cv, val_preds)
        recall_score_val = recall_score(y_test_cv, val_preds)
        f1_score_val = f1_score(y_test_cv, val_preds)

        all_val_preds_proba.extend(val_preds_proba)
        all_y_test.extend(y_test_cv)
        all_conf_matrices.append(conf_matrix)
        all_fprs.append(fpr)
        all_tprs.append(tpr)
        all_precisions.append(precision_curve)
        all_recalls.append(recall_curve)
        auc_scores.append(auc_score)
        pr_auc_scores.append(pr_auc_score)
        all_accuracies.append(accuracy)
        all_precisions_scores.append(precision_score_val)
        all_recalls_scores.append(recall_score_val)
        all_f1_scores.append(f1_score_val)

    return (all_val_preds_proba, all_y_test, all_conf_matrices, all_fprs, all_tprs, auc_scores,
            all_precisions, all_recalls, pr_auc_scores, all_accuracies, all_precisions_scores,
            all_recalls_scores, all_f1_scores)


def plot_roc_curves_with_mean(mean_fpr, mean_tpr, avg_auc, auc_std, all_fprs, all_tprs, auc_scores):
    plt.figure(dpi=200)
    for fpr, tpr in zip(all_fprs, all_tprs):
        plt.plot(fpr, tpr, color='skyblue', lw=2)
    plt.plot(mean_fpr, mean_tpr, color='darkblue', label=f'AUC = {avg_auc:.3f} ± {auc_std:.3f}', linewidth=4)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(handlelength=0, loc='lower right', frameon=False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve')
    plt.savefig('XGBoost ROC Curve.svg', format='svg', dpi=200)
    plt.show()


def plot_precision_recall_curves_with_mean(mean_precision, mean_recall, avg_pr_auc, auc_pr_std, all_precisions, all_recalls, pr_auc_scores):
    plt.figure(dpi=200)
    for precision, recall in zip(all_precisions, all_recalls):
        plt.plot(recall, precision, color='skyblue', lw=2)
    plt.plot(mean_recall, mean_precision, color='darkblue', label=f'AUC = {avg_pr_auc:.3f} ± {auc_pr_std:.3f}', linewidth=4)
    plt.plot([0, 1], [1, 0], color='gray', linestyle='--', lw=2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(handlelength=0, loc='lower left', frameon=False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('XGBoost Precision-Recall Curve')
    plt.savefig('XGBoost Precision-Recall Curve.svg', format='svg', dpi=200)
    plt.show()


# Evaluate model using cross-validation and collect results
(all_val_preds_proba, all_y_test, all_conf_matrices, all_fprs, all_tprs, auc_scores,
 all_precisions, all_recalls, pr_auc_scores, accuracies, precisions_scores,
 recalls_scores, f1_scores) = evaluate_model_cv_collect_results(best_model_cv, x_test, y_test)  # Evaluate using the independent test set

# Calculate mean and standard deviation
metrics_summary = {
    'AUC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores)},
    'PR AUC': {'mean': np.mean(pr_auc_scores), 'std': np.std(pr_auc_scores)},
    'Accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
    'Precision': {'mean': np.mean(precisions_scores), 'std': np.std(precisions_scores)},
    'Recall': {'mean': np.mean(recalls_scores), 'std': np.std(recalls_scores)},
    'F1 Score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)}
}

# Create DataFrame
metrics_df = pd.DataFrame({
    'AUC': auc_scores,
    'PR AUC': pr_auc_scores,
    'Accuracy': accuracies,
    'Precision': precisions_scores,
    'Recall': recalls_scores,
    'F1 Score': f1_scores
})


summary_df = pd.DataFrame(metrics_summary).T


# Export DataFrame to Excel file
metrics_df.to_excel("XGBoost model_evaluation_summary.xlsx", index=True)
summary_df.to_excel("XGBoost mean and std.xlsx", index=True)
print("XGBoost Results saved to model_evaluation_summary.xlsx")



# Calculate mean confusion matrix and round to integer
mean_conf_matrix = np.round(np.mean(all_conf_matrices, axis=0)).astype(int)
print("Mean Confusion Matrix:\n", mean_conf_matrix)

# Create heatmap of confusion matrix, disable x and y axis ticks
plt.figure(figsize=(1.5, 1), dpi=200)
sns.heatmap(mean_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
plt.title("XGBoost")

# Save as SVG format
plt.savefig('XGBoost Confusion.svg', format='svg', dpi=200)

plt.show()



# Plot ROC curve
mean_fpr = np.linspace(0, 1, 200)  # Use 200 points as interpolation targets
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fprs, all_tprs)], axis=0)
avg_auc = np.mean(auc_scores)
auc_std = np.std(auc_scores)
print("Average AUC:", avg_auc)
plot_roc_curves_with_mean(mean_fpr, mean_tpr, avg_auc, auc_std, all_fprs, all_tprs, auc_scores)


# Plot Precision-Recall curve
mean_precision = np.linspace(0, 1, 200)  # Use 200 points as interpolation targets
mean_recall = np.mean([np.interp(mean_precision, p, r) for p, r in zip(all_precisions, all_recalls)], axis=0)
auc_pr_std = np.std(pr_auc_scores)

avg_pr_auc = np.mean(pr_auc_scores)
auc_pr_std = np.std(pr_auc_scores)
plot_precision_recall_curves_with_mean(mean_precision, mean_recall, avg_pr_auc, auc_pr_std, all_precisions, all_recalls, pr_auc_scores)

# Create DataFrame
roc_pr_data = pd.DataFrame({
    'False Positive Rate': mean_fpr,
    'True Positive Rate': mean_tpr,
    'recall_curve': mean_recall,
    'precision_curve': mean_precision
})

# Write DataFrame to Excel file
roc_pr_data.to_excel("XGBoost ROC_PR_curve_data.xlsx", index=False)
