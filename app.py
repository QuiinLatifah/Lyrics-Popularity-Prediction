import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open('prediksi_lirik.pkl', 'rb'))
vectorizer = pickle.load(open('tf_idf_vectorizer.pkl', 'rb'))

st.title("Will this song be popular?")

# Input for lyrics
lyric = st.text_area("Write the song lyrics here", height=300)

lirik_predict = ''

if st.button('Predict'):
    if not lyric:
        st.warning("Please input your lyrics")
    else:
        # Transform input lyric using the loaded vectorizer
        lyric_transformed = vectorizer.transform([lyric])

        # Make prediction using the loaded model
        predict_lirik = model.predict(lyric_transformed)

        if predict_lirik == 1:
            st.success("The predicted popularity of this lyric is : Very Popular")
        elif predict_lirik == 0:
            st.warning("The predicted popularity of this lyric is : Quite Popular")
        else:
            st.error("The predicted popularity of this lyric is : Less Popular")