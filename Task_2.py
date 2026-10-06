import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("loan.csv")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size= .2, random_state= 42, stratify= y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size= .25, random_state= 42, stratify= y_temp
)

#missing values handling
print("Handle missing value")

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

num_df: pd.DataFrame = X_train[num_cols]
#simple imputer
simpl_imp = SimpleImputer(strategy="median")
X_train_simple = simpl_imp.fit_transform(X_train[num_cols])

#KNNImputer
knn_imp = KNNImputer(n_neighbors=5)
X_train_knn = knn_imp.fit_transform(X_train[num_cols])


print("missing value before imputation : ", num_df.isnull().sum().sum()) 
print("missing value after simpleimputer : ", np.isnan(X_train_simple).sum())
print("missing value after KNNImputer : ", np.isnan(X_train_knn).sum())

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
print("custom features successfully added")

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

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state= 42))
])

full_pipeline.fit(X_train_fe,y_train)
cv_pipe_score = cross_val_score(full_pipeline, X_train_fe, y_train, cv=5).mean()
print("pipeline cv accuracy : ", cv_pipe_score)

#save and reload pipeline with joblib

joblib.dump(full_pipeline, "fitted_pipline.joblib")
loaded_Pipeline = joblib.load("fitted_pipline.joblib")

sampl_raw_rows = X_val_fe.iloc[:3]
sample_preds = loaded_Pipeline.predict(sampl_raw_rows)
print("predictions for 3 sample raw rows : ", sample_preds.tolist())