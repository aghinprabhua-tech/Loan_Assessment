import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

encode_cat_cols = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + encode_cat_cols + ordinal_features

#supervised classifiers
log_reg = LogisticRegression(max_iter=1000, random_state= 42).fit(X_tr_proc, y_train)
dt_clf = DecisionTreeClassifier(max_depth=5, random_state= 42).fit(X_tr_proc, y_train)

#random forest with OOB
rf_clf = RandomForestClassifier(n_estimators=100, oob_score= True, random_state= 42).fit(X_tr_proc, y_train)
print("random forest oob score : ", rf_clf.oob_score_)

#xboost with early stopping
xgb_clf = XGBClassifier(n_estimators = 100, eval_metrics = "logloss", early_stopping_round = 10, random_state =42)
xgb_clf.fit(X_tr_proc, y_train, eval_set = [(X_val_proc, y_val)], verbose = False)

# predict proba vs predict
probas = log_reg.predict_proba(X_val_proc[:5])
preds = log_reg.predict(X_val_proc[:5])
print("LogisticRegression first 5row (probablities vs prediction) : ")
for p , pr in zip(probas, preds):
    print("[probablities [class 0, class 1] : ", p.round(3), " --> prediction : ", pr)

#top 5 important feature
rf_top5 = pd.Series(rf_clf.feature_importances_, index= all_feature_names).nlargest(5)
xgb_top5 = pd.Series(xgb_clf.feature_importances_, index = all_feature_names).nlargest(5)
print("top 5 features random forest : \n", rf_top5)
print("top 5 features XGboost : \n", xgb_top5)

#regression task 
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

line_reg = LinearRegression().fit(X_reg_tr_p, y_reg_tr)
ridge = Ridge(alpha = 1.0).fit(X_reg_tr_p, y_reg_tr)
lasso = Lasso(alpha=10.0).fit(X_reg_tr_p, y_reg_tr)

zero_coefs = np.sum(lasso.coef_ ==0)
print("lasso zero coefficient count :", zero_coefs, "out of ", len(lasso.coef_))

#unsupervised
km_cols = ["person_income", "loan_amnt", "person_age"]
X_km = SimpleImputer(strategy="median").fit_transform(X_train[km_cols])
X_km_scaled = StandardScaler().fit_transform(X_km)

inertias = []
sil_scores =[]
k_range  = range(2, 9)

print("kmeans segmentation : " )
for k in k_range:
    km = KMeans(n_clusters=k, random_state= 42, n_init= 10).fit(X_km_scaled)
    sil = silhouette_score(X_km_scaled, km.labels_)
    inertias.append(km.inertia_)
    sil_scores.append(sil)
    print(f"k = {k} | silhouette score : {sil:.4f} | inertia : {km.inertia_:.2f}")

#plot elbow curve
plt.figure(figsize=(8,4))
plt.plot(k_range, inertias, marker="o", color = "purple")
plt.title("k-means elbow curve")
plt.xlabel("number of cluster (k)")
plt.ylabel("inertia")
plt.grid(True)
plt.show()