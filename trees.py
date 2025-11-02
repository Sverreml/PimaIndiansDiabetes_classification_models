import pandas as pd
import numpy as np
import sklearn.tree as skt
import sklearn.ensemble as ske
import sklearn.model_selection as skm

#ready data
df = pd.read_csv(r"PimaIndiansDiabetes\pimaindiansdiabetes.csv",
                delimiter=",",
                header = 0)

d = {"pos": 1, "neg": 0}
df["diabetes"] = df["diabetes"].map(d)

X_data = df.drop(columns=["diabetes"])
Y_data = df["diabetes"]

X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

#Decision tree model
tree_model = skt.DecisionTreeClassifier(random_state=0)
tree_model.fit(X_train, Y_train)
Y_train_pred = tree_model.predict(X_train)
Y_test_pred = tree_model.predict(X_test)
train_misclass = np.mean(Y_train_pred != Y_train)
test_misclass = np.mean(Y_test_pred != Y_test)


#Bagged decision tree model
bagged_tree_model = ske.BaggingClassifier(
    estimator=skt.DecisionTreeClassifier(),
    n_estimators=100,
    random_state=0
)
bagged_tree_model.fit(X_train, Y_train)
Y_train_bagged_pred = bagged_tree_model.predict(X_train)    
Y_test_bagged_pred = bagged_tree_model.predict(X_test)
bagged_train_misclass = np.mean(Y_train_bagged_pred != Y_train)
bagged_test_misclass = np.mean(Y_test_bagged_pred != Y_test)

#Random forest model
random_forest_model = ske.RandomForestClassifier(
    n_estimators=100,
    random_state=0
)
random_forest_model.fit(X_train, Y_train)
Y_train_rf_pred = random_forest_model.predict(X_train)
Y_test_rf_pred = random_forest_model.predict(X_test)
rf_train_misclass = np.mean(Y_train_rf_pred != Y_train)
rf_test_misclass = np.mean(Y_test_rf_pred != Y_test)
print("Decision Tree Training Misclassification Error: {:.4f}".format(train_misclass))
print("Decision Tree Test Misclassification Error: {:.4f}".format(test_misclass))
print("Bagged Decision Tree Training Misclassification Error: {:.4f}".format(bagged_train_misclass))
print("Bagged Decision Tree Test Misclassification Error: {:.4f}".format(bagged_test_misclass))
print("Random Forest Training Misclassification Error: {:.4f}".format(rf_train_misclass))
print("Random Forest Test Misclassification Error: {:.4f}".format(rf_test_misclass))
