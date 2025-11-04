import pandas as pd
import numpy as np
from pygam import LogisticGAM, s
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector

# ready data
#ready data
df = pd.read_csv(
    r"PimaIndiansDiabetes_classification_models\pimaindiansdiabetes.csv",
    delimiter=",",
    header = 0
)

d = {"pos": 1, "neg": 0}
df["diabetes"] = df["diabetes"].map(d)

X_data = df.drop(columns=["diabetes"])
Y_data = df["diabetes"]
scaler = StandardScaler()
X_data = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)


X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

# Fit base GAM with all variables as splines
base_gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
base_gam.fit(X_train, Y_train)
base_gam.summary()

#Fit feature selection

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

sfs = SequentialFeatureSelector(
    base_gam,
    n_features_to_select="auto",      
    direction="backward",             
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

sfs.fit(X_train, Y_train)

selected_features = X_train.columns[sfs.get_support()]
print("Selected features:", list(selected_features))
# Fit final model with selected features
final_gam = LogisticGAM()
final_gam.gridsearch(X_train[selected_features].values, Y_train.values)
final_gam.summary()

train_errors = 1 - final_gam.accuracy(X_train[selected_features].values, Y_train.values)
test_errors = 1 - final_gam.accuracy(X_test[selected_features].values, Y_test.values)
print(f"Train misclassification error: {train_errors:.4f}")
print(f"Test misclassification error: {test_errors:.4f}")