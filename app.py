import streamlit as st
import joblib
from datetime import datetime

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Title
st.title("Sentiment Analysis Application")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Linear SVM Model for sentiment analysis")
    st.write("Accuracy: 90%")
    st.write("Precision: 0.94")
    st.write("Recall: 0.97")
    st.write("F1-Score: 0.93")
    st.markdown("---")
    show_metrics = st.checkbox("Show Model Metrics", value=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('sentiment_analysis_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_models()

if model and vectorizer:
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Review")
        user_input = st.text_area("Paste your product review:", height=120)
    
    with col2:
        st.subheader("Quick Examples")
        if st.button("Positive Example"):
            user_input = "Excellent product, highly recommended!"
        if st.button("Negative Example"):
            user_input = "Terrible quality, waste of money"
    
    st.markdown("---")
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Vectorize and predict
            vectorized = vectorizer.transform([user_input])
            prediction = model.predict(vectorized)[0]
            confidence = model.decision_function(vectorized)[0]
            confidence_pct = abs(confidence) / (abs(confidence) + 1) * 100
            
            # Display results
            st.markdown("### Analysis Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("POSITIVE")
                else:
                    st.error("NEGATIVE")
            
            with col2:
                st.metric("Confidence", f"{confidence_pct:.1f}%")
            
            with col3:
                st.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            st.markdown("---")
            st.markdown(f"**Review:** {user_input}")
            sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
            st.markdown(f"**Prediction:** {sentiment}")
            st.markdown(f"**Confidence:** {confidence_pct:.2f}%")
        else:
            st.warning("Please enter some text to analyze!")
    
    # Show metrics if selected
    if show_metrics:
        st.markdown("---")
        st.subheader("Model Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "90%")
        m2.metric("Precision", "0.94")
        m3.metric("Recall", "0.97")
        m4.metric("F1-Score", "0.93")
else:
    st.error("Models not found! Please ensure sentiment_analysis_model.pkl and tfidf_vectorizer.pkl are in the same directory.")
