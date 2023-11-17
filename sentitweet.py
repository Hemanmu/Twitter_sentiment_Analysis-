import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy


# Define a function to take input from an Excel file and perform sentiment analysis
def perform_sentiment_analysis(excel_file_path):
  """
  Performs sentiment analysis on the tweets in the given Excel file.

  Args:
    excel_file_path: The path to the Excel file.

  Returns:
    A Pandas DataFrame containing the tweets and their sentiment scores.
  """

  # Load the Excel file into a Pandas DataFrame
  df = pd.read_excel(excel_file_path)

  # Create a new column for the sentiment score
  df['sentiment_score'] = None

  # Iterate over the rows of the DataFrame and calculate the sentiment score for each entry
  for index, row in df.iterrows():
    tweet = df.loc[index, 'Text']

    # Preprocess the tweet
    tweet_word = []
    for word in tweet.split(' '):
      if word.startswith('@') and len(word) > 1:
        word = '@user'
      elif word.startswith('http'):
        word = 'http'
      tweet_word.append(word)
    tweet_proc = ' '.join(tweet_word)

    # Load the pre-trained RoBERTa model and tokenizer
    roberta = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # Encode the tweet and pass it to the model
    encoded_tweet = tokenizer(tweet_proc, return_tensors = 'pt')
    output = model(**encoded_tweet)

    # Get the sentiment scores
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Assign the sentiment label with the highest probability to the tweet
    sentiment_score = labels[scores.argmax()]

    # Update the DataFrame with the sentiment score
    df.loc[index, 'sentiment_score'] = sentiment_score

  return df


# Get the path to the Excel file from the user
excel_file_path = input('Enter the path to the Excel file: ')

# Perform sentiment analysis on the tweets in the Excel file
df_result = perform_sentiment_analysis(excel_file_path)

# Print the results
print(df_result)
