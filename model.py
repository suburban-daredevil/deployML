'''
Python file to create the model to classify the type of flower based on the Iris dataset
Users input in the web UI: petal_length, petal_width, sepal_length, sepal_width
Output from the model: The corresponding type of flower with the aforementioned features belongs to
Which is then displayed onto the UI
'''
import joblib # python library used to save the trained ML model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

seed = 5805

# Read original dataset
iris_df = pd.read_csv('data/iris.csv')
iris_df.sample(frac=1, random_state=seed)

print('\nPrint the head of the dataset:\n\n', iris_df.head().to_string())
print('The unique labels in the Species Target column are:', iris_df['Species'].unique())

# selecting features and target data
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# split data into train and test sets 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=seed,
    stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train.values.ravel())

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') # Accuracy: 0.95

# save the model to disk
joblib.dump(clf, 'rf_model.sav')