import json
import glob
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_json_records(path_pattern):
    records = []

    for path in glob.glob(path_pattern):
        with open(path, "r") as f:
            data = json.load(f)

            # Supports either one JSON object or a list of JSON objects
            if isinstance(data, dict):
                data = [data]

            for i, item in enumerate(data):
                row = {}
                # row["_source_file"] = str(path)
                # row["_source_index"] = i

                # Top-level fields
                row["fighter"] = item.get("fighter")
                row["arm"] = item.get("arm")

                # Flatten features
                features = item.get("features", {})
                for key, value in features.items():
                    row[key] = value

                # Label
                if item["punch"]:
                    row["label"] = item["punch_type"]
                else:
                    row["label"] = "no_punch"

                records.append(row)

    return pd.DataFrame(records)

df = load_json_records("data_collection/test_video*/punch_*/features.json")
print(df)
print(df.head())
print(df["label"].value_counts())

valid_types = ["jab", "cross", "hook", "uppercut", "overhand"]

df = df[df["label"].isin(valid_types)]

X = df.drop(columns=["label", "fighter"])
y = df["label"]

valid_types = ["jab", "cross", "hook", "uppercut", "overhand"]

df = df[df["label"].isin(valid_types)]

# Features and label
X = df.drop(columns=["label", "fighter"])
y = df["label"]

categorical_features = ["arm"]
numeric_features = [col for col in X.columns if col not in categorical_features]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["arm"]),
        ("num", numeric_transformer, numeric_features),
    ]
)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

print("\nClasses:")
print(clf.classes_)

X_clean = preprocessor.fit_transform(X)
print(X_clean)

# Save model
joblib.dump(clf, "random_forest_model.joblib")