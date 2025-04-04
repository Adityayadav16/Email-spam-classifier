import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load the vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    if not isinstance(tfidf, TfidfVectorizer):
        raise ValueError("The loaded object is not a valid TfidfVectorizer.")
    print("TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading vectorizer.pkl: {e}")
    tfidf = None

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    model = None

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# User input
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Check if vectorizer and model are loaded successfully
    if tfidf is None or model is None:
        st.error("Failed to load required files. Please fix the issues and try again.")
    else:
        try:
            # 1. Preprocess the input
            transformed_sms = transform_text(input_sms)
            st.write("Transformed SMS:", transformed_sms)

            # 2. Vectorize the input
            vector_input = tfidf.transform([transformed_sms])  # Ensure transformed_sms is passed as a list
            st.write("Vectorization successful.")

            # 3. Predict the label
            result = model.predict(vector_input)[0]

            # 4. Display the result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
