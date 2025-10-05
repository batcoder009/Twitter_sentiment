import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

CSV_PATH = "twitter_training.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   
    text = re.sub(r"@[A-Za-z0-9_]+","", text)   
    text = re.sub(r"#[A-Za-z0-9_]+","", text)   
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)  
    text = re.sub(r"(.)\1{2,}", r"\1\1", text) 
    return text.strip()

def main():
    df = pd.read_csv(CSV_PATH, header=None)
    df = df.rename(columns={0:"id", 1:"topic", 2:"label", 3:"text"})
    df = df.dropna()
    df["text"] = df["text"].apply(clean_text)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    # parameter grid
    param_grid = [
        {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf': [LinearSVC()],
            'clf__C': [0.1, 1, 10]
        },
        {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf': [LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced")],
            'clf__C': [0.1, 1, 10]
        }
    ]

    # GridSearch
    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(best_model, "sentiment_model.joblib")
    print("✅ Best model saved as sentiment_model.joblib")

if __name__ == "__main__":
    main()
