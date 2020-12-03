import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_pandas

def findMostCommon(stemmed):
    vectorizer = CountVectorizer(analyzer='word', min_df=1)
    print("Finding 30 Most Common Words...")
    splitWords = vectorizer.fit_transform(stemmed)
    vocabMap = list(vectorizer.get_feature_names())
    # Taken from http://stackoverflow.com/questions/3337301/numpy-matrix-to-array
    # and http://stackoverflow.com/questions/13567345/how-to-calculate-the-sum-of-all-columns-of-a-2d-numpy-array-efficiently
    counts = splitWords.sum(axis=0).A1
    finalMap = Counter(dict(zip(vocabMap, counts)))
    return(finalMap.most_common(30))

def readData():
    print('Reading data...')
    df = pd.read_csv('data/data_stemmed.csv')
    df['stemmedString'] = df['stemmed'].apply(lambda x: x.strip('[]').replace('\'','').replace(',',''))
    return df

def vectorizeCount(xTrain, xTest, vocab):
    print("Vectorizing Count Data...")
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), vocabulary=vocab)
    vectorizedWordsTrain = vectorizer.fit_transform(xTrain['stemmedString'])
    vectorizedTrain = pd.DataFrame(vectorizedWordsTrain.toarray(), columns = vectorizer.get_feature_names())
    vectorizedWordsTest = vectorizer.fit_transform(xTest['stemmedString'])
    vectorizedTest = pd.DataFrame(vectorizedWordsTest.toarray(), columns = vectorizer.get_feature_names())
    return vectorizedTrain, vectorizedTest

def vectorizeBinary(xTrain, xTest, vocab):
    print("Vectorizing Binary Data...")
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), vocabulary=vocab, binary = True)
    vectorizedWordsTrain = vectorizer.fit_transform(xTrain['stemmedString'])
    vectorizedTrain = pd.DataFrame(vectorizedWordsTrain.toarray(), columns = vectorizer.get_feature_names())
    vectorizedWordsTest = vectorizer.fit_transform(xTest['stemmedString'])
    vectorizedTest = pd.DataFrame(vectorizedWordsTest.toarray(), columns = vectorizer.get_feature_names())
    return vectorizedTrain, vectorizedTest

def main():
    df = readData()
    xTrain, xTest, yTrain, yTest = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size =.3, random_state = 334)
    mostCommon = findMostCommon(xTrain['stemmedString'])
    vocab = [item[0] for item in mostCommon]
    print(vocab)
    vectorizedTrainCount, vectorizedTestCount = vectorizeCount(xTrain, xTest, vocab)
    xTrainCount, xTestCount = xTrain.copy(),xTest.copy()
    xTrainCount[vectorizedTrainCount.columns], xTestCount[vectorizedTestCount.columns] = np.array(vectorizedTrainCount),  np.array(vectorizedTestCount)
    vectorizedTrainBinary, vectorizedTestBinary = vectorizeBinary(xTrain, xTest, vocab)
    xTrainBinary, xTestBinary = xTrain.copy(),xTest.copy()
    xTrainBinary[vectorizedTrainBinary.columns], xTestBinary[vectorizedTestBinary.columns] = np.array(vectorizedTrainBinary),  np.array(vectorizedTestBinary)
    xTrainCount.drop(['stemmedString'], axis=1, inplace=True),xTestCount.drop(['stemmedString'], axis=1, inplace=True)
    xTrainBinary.drop(['stemmedString'], axis=1, inplace=True),xTestBinary.drop(['stemmedString'], axis=1, inplace=True)
    xTrainCount.to_csv("data/xtrain_count.csv", index=False),xTestCount.to_csv("data/xtest_count.csv", index=False)
    xTrainBinary.to_csv("data/xtrain_binary.csv", index=False),xTestBinary.to_csv("data/xtest_binary.csv", index=False)
    yTrain.to_csv("data/ytrain.csv", index=False), yTest.to_csv("data/ytest.csv", index=False)
if __name__ == "__main__":
    main()