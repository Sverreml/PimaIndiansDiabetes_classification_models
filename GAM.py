import pandas as pd
from pygam import GAM, s   
import sklearn.model_selection as skm

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

X_train, X_test, Y_train, Y_test = skm.train_test_split(X_data, Y_data, random_state=0, test_size=0.33, stratify=Y_data)

# fit GAM model
gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
gam.fit(X_train, Y_train)
gam.summary()