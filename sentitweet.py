import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

# Cache the model loading
@st.cache(allow_output_mutation=True)
def load_model():
    roberta = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    return model, tokenizer

model, tokenizer = load_model()

def perform_sentiment_analysis(df):
    df['sentiment_score'] = None
    for index, row in df.iterrows():
        tweet = row['Text']
        # [Your existing tweet processing logic]
        # ...
        # Perform sentiment analysis
        encoded_tweet = tokenizer(tweet, return_tensors='pt')
        output = model(**encoded_tweet)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment_score = labels[scores.argmax()]
        df.loc[index, 'sentiment_score'] = sentiment_score
    return df

st.title('Sentiment Analysis Tool')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df_result = perform_sentiment_analysis(df)
    st.write(df_result)
