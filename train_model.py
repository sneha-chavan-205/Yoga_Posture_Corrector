import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
data = pd.read_csv("pose_dataset.csv")

# split features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = RandomForestClassifier(n_estimators=200)

# train model
model.fit(X_train, y_train)

# check accuracy
accuracy = model.score(X_test, y_test)

print("Model accuracy:", accuracy)

# save model
pickle.dump(model, open("pose_model.pkl", "wb"))

print("Model saved!")