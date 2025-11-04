import sklearn.neighbors as skn
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#ready data
df = pd.read_csv(
    r"PimaIndiansDiabetes_classification_models\pimaindiansdiabetes2.csv",
    delimiter=",",
    header = 0
)
df.dropna(inplace=True)

d = {"pos": 1, "neg": 0}
df["diabetes"] = df["diabetes"].map(d)

X_data = df.drop(columns=["diabetes"])
Y_data = df["diabetes"]
scaler = StandardScaler()
X_data = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)

X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

k_values = list(range(1, 51))
fold_train_errors = []
fold_test_errors = []
Loocv_train_errors = []
Loocv_test_errors = []

for k in k_values:
    print(f"Evaluating k={k}")
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


best_k_fold_index = np.argmin(fold_test_errors)
best_k_fold = k_values[best_k_fold_index]
best_k_loo_index = np.argmin(Loocv_test_errors)
best_k_loo = k_values[best_k_loo_index]


#plotting
plt.figure(figsize=(12, 6))
plt.plot(k_values, fold_train_errors, label='5-Fold CV Training Error', marker='o')
plt.plot(k_values, fold_test_errors, label='5-Fold CV Test Error', marker='o')
plt.plot(k_values, Loocv_train_errors, label='LOOCV Training Error', marker='x')
plt.plot(k_values, Loocv_test_errors, label='LOOCV Test Error', marker='x')
plt.axvline(best_k_fold, color='r', linestyle='--', label=f'Best k 5-fold: {best_k_fold}')
plt.axvline(best_k_loo, color='g', linestyle='--', label=f'Best k LOOCV: {best_k_loo}')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Misclassification Error')
plt.title('k-NN Classifier: Misclassification Error vs k')
plt.legend()
plt.grid()
plt.show()

#error
print(f"Test error on test set with k={best_k_fold}: {np.mean(knn.fit(X_train, Y_train).predict(X_test) != Y_test)}")
print(f"Test error on test set with k={best_k_loo}: {np.mean(knn.fit(X_train, Y_train).predict(X_test) != Y_test)}")
print(f"train misclassification error with k={best_k_fold}: {np.mean(knn.fit(X_train, Y_train).predict(X_train) != Y_train)}")
print(f"train misclassification error with k={best_k_loo}: {np.mean(knn.fit(X_train, Y_train).predict(X_train) != Y_train)}")