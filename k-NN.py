import sklearn.neighbors as skn
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#ready data
df = pd.read_csv(
    r"C:\Users\Sverr\Desktop\Datasets\PimaIndiansDiabetes\pimaindiansdiabetes.csv",
    delimiter=",",
    header = 0
)

d = {"pos": 1, "neg": 0}
df["diabetes"] = df["diabetes"].map(d)

X_data = df.drop(columns=["diabetes"])
Y_data = df["diabetes"]

X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

k_values = list(range(1, 51))
fold_train_errors = []
fold_test_errors = []
Loocv_train_errors = []
Loocv_test_errors = []

for k in k_values:
    knn = skn.KNeighborsClassifier(n_neighbors=k)

    #5-fold CV
    train_errors = []
    test_errors = []
    kf = skm.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(X_train, Y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_fold_train, Y_fold_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

        knn.fit(X_fold_train, Y_fold_train)
        Y_fold_train_pred = knn.predict(X_fold_train)
        Y_fold_val_pred = knn.predict(X_fold_val)

        train_misclass = np.mean(Y_fold_train_pred != Y_fold_train)
        val_misclass = np.mean(Y_fold_val_pred != Y_fold_val)

        train_errors.append(train_misclass)
        test_errors.append(val_misclass)

    fold_train_errors.append(np.mean(train_errors))
    fold_test_errors.append(np.mean(test_errors))

    #LOOCV
    loo = skm.LeaveOneOut()
    loo_train_errors = []
    loo_test_errors = []
    for train_index, val_index in loo.split(X_train):
        X_loo_train, X_loo_val = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_loo_train, Y_loo_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

        knn.fit(X_loo_train, Y_loo_train)
        Y_loo_train_pred = knn.predict(X_loo_train)
        Y_loo_val_pred = knn.predict(X_loo_val)

        train_misclass = np.mean(Y_loo_train_pred != Y_loo_train)
        val_misclass = np.mean(Y_loo_val_pred != Y_loo_val)

        loo_train_errors.append(train_misclass)
        loo_test_errors.append(val_misclass)

    Loocv_train_errors.append(np.mean(loo_train_errors))
    Loocv_test_errors.append(np.mean(loo_test_errors))


best_k_index = np.argmin(fold_test_errors)
best_k = k_values[best_k_index]


#plotting
plt.figure(figsize=(12, 6))
plt.plot(k_values, fold_train_errors, label='5-Fold CV Training Error', marker='o')
plt.plot(k_values, fold_test_errors, label='5-Fold CV Test Error', marker='o')
plt.plot(k_values, Loocv_train_errors, label='LOOCV Training Error', marker='x')
plt.plot(k_values, Loocv_test_errors, label='LOOCV Test Error', marker='x')
plt.axvline(best_k, color='r', linestyle='--', label=f'Best k: {best_k}')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Misclassification Error')
plt.title('k-NN Classifier: Misclassification Error vs k')
plt.legend()
plt.grid()
plt.show()