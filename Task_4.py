import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, silhouette_score, mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("loan.csv")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size= .2, random_state= 42, stratify= y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size= .25, random_state= 42, stratify= y_temp
)

#feature engineering
def add_custom_features(data: pd.DataFrame) -> pd.DataFrame:
    df_feat = data.copy()
    df_feat["income_to_loan_ratio"] = df_feat["person_income"]/(df_feat["loan_amnt"] + 1)
    df_feat["log_person_income"] = np.log1p(df_feat["person_income"])
    df_feat["approx_birth_rate"] = 2026 - df_feat["person_age"]
    df_feat["loan_to_emp_length"] = df_feat["loan_amnt"] / (df_feat["person_emp_length"].fillna(0) +1)
    return df_feat

X_train_fe = add_custom_features(X_train)
X_val_fe = add_custom_features(X_val)
X_test_fe = add_custom_features(X_test)

num_features = X_train_fe.select_dtypes(include=["int64", "Float64"]).columns.tolist()
cat_features = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
ordinal_features = ["loan_grade"]

# #identify feature columns
num_features = X_train_fe.select_dtypes(include=["int64", "Float64"]).columns.tolist()
cat_features = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
ordinal_features = ["loan_grade"]
grade_order = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]

# column transform pipeline with SimpleImputer
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
ord_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy= "most_frequent")),
    ("encoder", OrdinalEncoder(categories= grade_order))
])

preprocessor = ColumnTransformer(
    transformers= [
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
        ("ord", ord_transformer, ordinal_features)
    ]
)

X_tr_proc = preprocessor.fit_transform(X_train_fe)
X_val_proc = preprocessor.transform(X_val_fe)

# model metrics imbalance explanation 
models = {
    'logistic_regression': LogisticRegression(max_iter= 1000, random_state= 42).fit(X_tr_proc, y_train),
    "decision_tree" : DecisionTreeClassifier(max_depth= 5, random_state= 42).fit(X_tr_proc, y_train),
    "random_forest" : RandomForestClassifier(n_estimators= 100, max_depth= 5, random_state= 42).fit(X_tr_proc, y_train),
    "xgboost" : XGBClassifier(n_estimators= 100, eval_metric= "logloss", random_state= 42).fit(X_tr_proc, y_train)
}

print("validation classification Performance :")
for name, model in models.items():
    y_pred = model.predict(X_val_proc)
    print(f"\n---{name}---")
    print("Accuracy : ", accuracy_score(y_val, y_pred),
           " | Precision : ", precision_score(y_val, y_pred),
             " | Recall : ", recall_score(y_val, y_pred),
               " | F1-score : ", f1_score(y_val, y_pred))

# ROC curve plot 
plt.figure(figsize=(8, 6))
plot_models = {
    "logistic_regression": models["logistic_regression"],
    "random_forest" : models["random_forest"],
    "xgboost" : models["xgboost"],
}

for name, model in plot_models.items():
    y_proba = model.predict_proba(X_val_proc)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation Set)")
plt.legend()
plt.grid()
plt.show()

# Decision threshold turning
probs_log = models["logistic_regression"].predict_proba(X_val_proc)[:, 1]
thresholds = np.linspace(0.01, 0.99, 100)
for th in thresholds:
    preds_th = (probs_log >= th).astype(int)
    recall_th = recall_score(y_val, preds_th)
    if recall_th >= 0.75:
        prec_th = precision_score(y_val, preds_th)
        print(f"Threshold: {th:.2f} achieves Recall: {recall_th:.2f} with Precision: {prec_th:.2f}") 
        break

#hyper tuning
param_grid_rf = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state= 42), param_grid_rf, cv= 5, scoring= "roc_auc")
grid_rf.fit(X_tr_proc, y_train)
print("Best parameters for Random Forest: ", grid_rf.best_params_)

#randomized search for XGBoost
param_dist_xgb = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [.8, 1.0]
}

rand_xgb = RandomizedSearchCV(XGBClassifier(random_state= 42), param_dist_xgb, n_iter= 10, cv= 5, scoring= "roc_auc", random_state= 42)
rand_xgb.fit(X_tr_proc, y_train)
print("Best parameters for XGBoost: ", rand_xgb.best_params_)

# regression metrics vs mean baseline
X_reg_tr = X_train_fe.drop(columns = ["loan_amnt"])
y_reg_tr = X_train_fe["loan_amnt"]
X_reg_val = X_val_fe.drop(columns=["loan_amnt"])
y_reg_val = X_val_fe["loan_amnt"]

reg_num_cols = X_reg_tr.select_dtypes(include=["int64", "float64"]).columns.tolist()
reg_preprocessor = ColumnTransformer(
      transformers= [
            ("num", num_transformer, reg_num_cols),
            ("cat", cat_transformer, cat_features),
            ("ord", ord_transformer, ordinal_features)
        ]
)

X_reg_tr_p = reg_preprocessor.fit_transform(X_reg_tr)
x_reg_val_p = reg_preprocessor.transform(X_reg_val)

ridge = Ridge(alpha = 1.0).fit(X_reg_tr_p, y_reg_tr)
reg_preds = ridge.predict(x_reg_val_p)
base_preds = np.full_like(y_reg_val, y_reg_tr.mean())

print("Ridge Regression Metrics vs Mean Baseline:")
print("baseline MAE: ", mean_absolute_error(y_reg_val, base_preds), " | model MAE: ", mean_absolute_error(y_reg_val, reg_preds))
print("baseline RMSE: ", np.sqrt(mean_squared_error(y_reg_val, base_preds)), " | model RMSE: ", np.sqrt(mean_squared_error(y_reg_val, reg_preds)))
print("R2 score: ", r2_score(y_reg_val, reg_preds))

#final model evaluation on test set
best_model = rand_xgb.best_estimator_
x_test_proc = preprocessor.transform(X_test_fe)

final_preds = best_model.predict(x_test_proc)
final_auc = roc_auc_score(y_test, best_model.predict_proba(x_test_proc)[:, 1])

print("Final test evaluation (best model - XGBoost")
print("Test AUC: ", accuracy_score(y_test, final_preds))
print("Test ROC-AUC: ", final_auc)
print("Test f1-score: ", f1_score(y_test, final_preds))