import pandas as pd
import numpy as np
import sklearn.tree as skt
import sklearn.ensemble as ske
import sklearn.model_selection as skm

#ready data
df = pd.read_csv(r"PimaIndiansDiabetes_classification_models\pimaindiansdiabetes2.csv",
                delimiter=",",
                header = 0)
df.dropna(inplace=True)

d = {"pos": 1, "neg": 0}
df["diabetes"] = df["diabetes"].map(d)

X_data = df.drop(columns=["diabetes"])
Y_data = df["diabetes"]

X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

#Decision tree model
reg_tree = skt.DecisionTreeClassifier(random_state=0)
reg_tree.fit(X_train, Y_train)

path = reg_tree.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
trees = []
for ccp_alpha in ccp_alphas:
    clf = skt.DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    trees.append(clf)


print(f"Number of trees built: {len(trees)}")


train_scores = [t.score(X_train, Y_train) for t in trees]
test_scores = [t.score(X_test, Y_test) for t in trees]

best_alpha = ccp_alphas[np.argmax(test_scores)]

tree_model = skt.DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha)
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
