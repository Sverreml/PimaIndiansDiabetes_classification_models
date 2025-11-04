import pandas as pd
import numpy as np
import sklearn.tree as skt
import sklearn.ensemble as ske
import sklearn.model_selection as skm
from sklearn.model_selection import StratifiedKFold

#ready data
df = pd.read_csv(r"PimaIndiansDiabetes_classification_models\pimaindiansdiabetes.csv",
                delimiter=",",
                header = 0)
df.dropna(inplace=True)

d = {"pos": 1, "neg": 0}
df["diabetes"] = df["diabetes"].map(d)

X_data = df.drop(columns=["diabetes"])
Y_data = df["diabetes"]

X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

########################################################################################################################

# Decision tree model 
path = skt.DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, Y_train)
alphas = path.ccp_alphas
param_grid = {'ccp_alpha': alphas}

grid_tree = skm.GridSearchCV(
    skt.DecisionTreeClassifier(random_state=0),
    param_grid,
    cv=cv
)
grid_tree.fit(X_train, Y_train)

tree_model = grid_tree.best_estimator_
Y_train_tree_pred = tree_model.predict(X_train)    
Y_test_tree_pred = tree_model.predict(X_test)
tree_train_misclass = np.mean(Y_train_tree_pred != Y_train)
tree_test_misclass = np.mean(Y_test_tree_pred != Y_test)

########################################################################################################################

# Bagged decision tree model
param_grid = {'estimator__ccp_alpha': alphas}

grid_bag = skm.GridSearchCV(
    ske.BaggingClassifier(estimator=skt.DecisionTreeClassifier(random_state=0), n_estimators=100, random_state=0),
    param_grid,
    cv=cv
)
grid_bag.fit(X_train, Y_train)

bagged_tree_model = grid_bag.best_estimator_
Y_train_bagged_pred = bagged_tree_model.predict(X_train)    
Y_test_bagged_pred = bagged_tree_model.predict(X_test)
bagged_train_misclass = np.mean(Y_train_bagged_pred != Y_train)
bagged_test_misclass = np.mean(Y_test_bagged_pred != Y_test)

########################################################################################################################

# Random Forest model
param_grid = {
    'ccp_alpha': [0.0, 0.001, 0.005, 0.01],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
grid_rf = skm.GridSearchCV(
    ske.RandomForestClassifier(n_estimators=200, random_state=42),
    param_grid,
    cv=cv
)
grid_rf.fit(X_train, Y_train)


random_forest_model = grid_rf.best_estimator_
Y_train_rf_pred = random_forest_model.predict(X_train)
Y_test_rf_pred = random_forest_model.predict(X_test)
rf_train_misclass = np.mean(Y_train_rf_pred != Y_train)
rf_test_misclass = np.mean(Y_test_rf_pred != Y_test)

print(f"Decision Tree Train Misclassification Error: {tree_train_misclass:.4f}")
print(f"Decision Tree Test Misclassification Error: {tree_test_misclass:.4f}")
print(f"Bagged Tree Train Misclassification Error: {bagged_train_misclass:.4f}")
print(f"Bagged Tree Test Misclassification Error: {bagged_test_misclass:.4f}")
print(f"Random Forest Train Misclassification Error: {rf_train_misclass:.4f}")
print(f"Random Forest Test Misclassification Error: {rf_test_misclass:.4f}")