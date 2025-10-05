
import streamlit as st
import joblib
import re
import pandas as pd

vect, clf = joblib.load("sentiment_model.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+","", text)
    text = re.sub(r"#[A-Za-z0-9_]+","", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# styled prediction box
def styled_box(label):
    colors = {
        "positive": "#4CAF50",   # green
        "negative": "#F44336",   # red
        "neutral": "#9E9E9E",    # gray
        "irrelevant": "#2196F3"  # blue
    }
    color = colors.get(label.lower(), "#333333")
    return f"""
    <div style="background-color:{color}; 
                padding:15px; 
                border-radius:10px; 
                text-align:center; 
                font-size:20px; 
                color:white; 
                font-weight:bold;">
        Prediction: {label.capitalize()}
    </div>
    """

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.title("Twitter Sentiment Analysis 🐦" \
"")

menu = st.sidebar.radio("Navigation", ["🔮 Predict", "📂 Batch Prediction"])

if menu == "🔮 Predict":
    tweet = st.text_area("Enter a tweet:")
    if st.button("Predict"):
        clean = clean_text(tweet)
        vec = vect.transform([clean])
        pred = clf.predict(vec)[0]
        st.markdown(styled_box(pred), unsafe_allow_html=True)

        # save to session history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append((tweet, pred))

    # show history
    if "history" in st.session_state and len(st.session_state["history"]) > 0:
        st.subheader("📜 Prediction History")
        for i, (txt, lab) in enumerate(reversed(st.session_state["history"][-5:]), 1):
            st.markdown(styled_box(lab), unsafe_allow_html=True)
            st.write(f"Tweet: {txt}")

    # Example buttons
    st.subheader("✨ Try Example Tweets")
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        if st.button("Positive Example"):
            st.text_area("Enter a tweet:", value="I absolutely love this update, it's so smooth!", key="pos")
    with col2: 
        if st.button("Negative Example"):
            st.text_area("Enter a tweet:", value="This app keeps crashing and I hate it.", key="neg")
    with col3: 
        if st.button("Neutral Example"):
            st.text_area("Enter a tweet:", value="The update will be released tomorrow.", key="neu")
    with col4: 
        if st.button("Irrelevant Example"):
            st.text_area("Enter a tweet:", value="Just had pasta for dinner 🍝", key="irr")

elif menu == "📂 Batch Prediction":
    st.subheader("Upload a CSV of tweets to classify")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a column named 'text'")
        else:
            df["clean_text"] = df["text"].apply(clean_text)
            df["prediction"] = clf.predict(vect.transform(df["clean_text"]))
            st.dataframe(df.head())

            # download option
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", data=csv_out, file_name="predictions.csv", mime="text/csv")
