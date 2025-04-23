import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from azureml.core import Run

# Load run context
run = Run.get_context()

# Load data
df = pd.read_csv("data/boston.csv")
X = df.drop("MEDV", axis=1)
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)

# Log metrics
run.log("alpha", 0.5)
run.log("model_score", model.score(X_test, y_test))

# Save model
joblib.dump(model, "outputs/model.pkl")
