import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords #stopwords are words like "is", and "the"
from tqdm import tqdm, tqdm_pandas
import string
import re

def printPercentMissingVals(df):
    print("------------Percentage of Missing Values------------")
    for attr in df:
        print(f"{attr}: {round((sum(df[attr].isnull())/df.shape[0])*100,4)}%")

def text_stemming(row):#Stem text, remove punctuation, take out stop words and unhelpful words like wine and drink.
    stop_words = set(stopwords.words("english"))
    unhelpful_words = set(["wine", "drink"])
    stemmer = PorterStemmer()
    wordList = word_tokenize(re.sub(r'\d+', 'number', (row.lower()).translate(str.maketrans('', '', string.punctuation))))
    wordListStemmed = [ stemmer.stem(i) for i in wordList if (i not in stop_words and i not in unhelpful_words)]

    return wordListStemmed

def main():
    df = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)
    printPercentMissingVals(df)
    tqdm.pandas(desc="Progress")
    df['stemmed'] = df['description'].progress_apply(text_stemming)
    df.dropna(subset=['price'], inplace=True)
    print(f"Stemmed Description:{df['stemmed']}")
    df.to_csv("data/data_stemmed.csv", index=False)

if __name__ == "__main__":
    main()