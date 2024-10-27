import pandas as pd
# Create Data report with ydata_profiling
# from ydata_profiling import ProfileReport
# Split data to train and test
from sklearn.model_selection import train_test_split
# Scale data
from sklearn.preprocessing import StandardScaler
# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
# Find the best high paramaters for each model with GridSeacrchCV
from sklearn.model_selection import GridSearchCV
# Evaluate model
from sklearn.metrics import classification_report, recall_score, precision_score
# Find the best accuracy of models
from lazypredict.Supervised import LazyClassifier

file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# profile = ProfileReport(data, title='Diabetes Report', explorative=True)
# profile.to_file('report.html')

# data split()
target = 'Outcome'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# data processing
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Train model
model = GaussianNB()
model.fit(x_train, y_train)

# Find the best high parameter for model
# param = {
#     'n_estimators': [100, 200, 300],
#     'criterion': ['gini', 'entropy', 'log_loss']
# }
#
# model = GridSearchCV(estimator=RandomForestClassifier(random_state=100), param_grid=param, cv=5, scoring='recall', verbose=2)
# model.fit(x_train, y_train)
#
# # Evaluate model
y_predict = model.predict(x_test)
unexpected_predict = 0
accuracy = 0
recall_work = 0
for real, predict in zip(y_test, y_predict):
    print(f"predict: {predict} real: {real}")
    if real > predict:
        unexpected_predict += 1
        print("unexpected")
    elif real == predict:
        accuracy += 1
    else:
        recall_work += 1
print(unexpected_predict, '/', len(y_test))
print(recall_work, '/', len(y_test))
print(accuracy/len(y_test))
print(classification_report(y_test, y_predict))

# find the best model with high accuracy
# clf = LazyClassifier(predictions=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
#
# print(models)


'''                                         recall(Possitive class) 
GaussianNB                                           0.69
QuadraticDiscriminantAnalysis                        0.69
'''