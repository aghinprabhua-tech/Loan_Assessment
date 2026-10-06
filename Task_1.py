import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("credit_risk_dataset.csv")

# Task 1
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size= .2, random_state= 42, stratify= y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size= .25, random_state= 42, stratify= y_temp
)

print("data splits : Train = ", len(X_train), " Val = ", len(X_val), " Test = ", len(X_test))

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()

imputer_base = SimpleImputer(strategy="median")
X_train_num = imputer_base.fit_transform(X_train[num_cols])
X_val_num = imputer_base.transform(X_val[num_cols])

#DecisionTreeClassifier
depths = [1, 3, 5, 10, None]
for d in depths:
    dt = DecisionTreeClassifier(max_depth= d , random_state= 42)
    dt.fit(X_train_num, y_train)
    tr_acc = dt.score(X_train_num, y_train)
    val_acc = dt.score(X_val_num, y_val)
    print("Depth ", str(d), " : train ", tr_acc, " val : ", val_acc)

# 5-Fold cross validation
scalar_base = StandardScaler()
X_train_num_scaled = scalar_base.fit_transform(X_train_num)
lr_cv = LogisticRegression(max_iter= 1000, random_state= 42)
cv_scores = cross_val_score(lr_cv, X_train_num_scaled, y_train, cv= 5)
print(f"5-Fold CV accuracy : {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

#Data leakage demonstration
# Data leakage
scalar_leak = StandardScaler()
X_scaled_all = scalar_leak.fit_transform(X[num_cols])

X_tr_leak, X_val_leak = train_test_split(
    X_scaled_all, test_size= .2, random_state= 42
)

#no data leakage
X_tr_raw, X_val_raw = train_test_split(
    X[num_cols], test_size= .2, random_state= 42
)

scalar_correct = StandardScaler()
X_tr_correct = scalar_correct.fit_transform(X_tr_raw)
X_val_correct = scalar_correct.transform(X_val_raw)

print("leaked scalar mean (full data) : ", scalar_leak.mean_[1].round(2))
print("correct scalar mean (train only) : ", scalar_correct.mean_[1].round(2))