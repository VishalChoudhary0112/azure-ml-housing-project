# scripts/train.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from azureml.core import Run

run = Run.get_context()

# 1. load & clean
df = pd.read_csv("boston.csv")
df = df.drop(columns=["Unnamed: 0"])
X = df.drop("medv", axis=1)
y = df["medv"]

# 2. split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. train
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)

# 4. log & save
run.log("alpha", 0.5)
run.log("r2_test", model.score(X_test, y_test))
joblib.dump(model, "outputs/model.pkl")
