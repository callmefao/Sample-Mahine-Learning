import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTEN
from lazypredict.Supervised import LazyClassifier

def filter_location(location):
    result = re.findall(",\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location


# Đọc dữ liệu từ tệp CSV
data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"

# Tách dữ liệu thành các biến độc lập và biến mục tiêu
x = data.drop(target, axis=1)
y = data[target]



# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
#
# print(y_train.value_counts())
ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
    'managing_director_small_medium_company': 500,
    'specialist': 500,
    'director_business_unit_leader': 500,
    'bereichsleiter': 1000
})
x_train, y_train = ros.fit_resample(x_train, y_train)
print("_"*20)
print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95, min_df=0.01), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_select", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])


clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(classification_report(y_test, y_predict))
#                                         precision    recall  f1-score   support With max,min df, without SelectKBest

#                               accuracy                           0.73      1615
#                              macro avg       0.51      0.32      0.34      1615
#                           weighted avg       0.72      0.73      0.69      1615
#
#                                         precision    recall  f1-score   support  SelectKBest = 800

#                               accuracy                           0.75      1615
#                              macro avg       0.68      0.37      0.41      1615
#                           weighted avg       0.74      0.75      0.72      1615

#     accuracy                           0.76      1615 percentile = 5
#    macro avg       0.69      0.38      0.42      1615
# weighted avg       0.76      0.76      0.73      1615
