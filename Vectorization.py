import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_pandas

def findMostCommon(stemmed):
    vectorizer = CountVectorizer(analyzer='word',ngram_range=(1, 1), min_df=0.1)
    print("Finding Most Common Words...")
    splitWords = vectorizer.fit_transform(stemmed)
    vocabMap = list(vectorizer.get_feature_names())
    return(vocabMap)

def readData():
    print('Reading data...')
    df = pd.read_csv('data/data_stemmed.csv')
    df['stemmed'] = df['stemmed'].apply(lambda x: x.strip('[]').replace('\'','').replace(',',''))
    return df

def vectorizeCount(xTrain, xTest, vocab):
    print("Vectorizing Count Data...")
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), vocabulary=vocab)
    vectorizedWordsTrain = vectorizer.fit_transform(xTrain['stemmed'])
    vectorizedTrain = pd.DataFrame(vectorizedWordsTrain.toarray(), columns = vectorizer.get_feature_names())
    vectorizedWordsTest = vectorizer.fit_transform(xTest['stemmed'])
    vectorizedTest = pd.DataFrame(vectorizedWordsTest.toarray(), columns = vectorizer.get_feature_names())
    return vectorizedTrain, vectorizedTest

def vectorizeBinary(xTrain, xTest, vocab):
    print("Vectorizing Binary Data...")
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), vocabulary=vocab, binary = True)
    vectorizedWordsTrain = vectorizer.fit_transform(xTrain['stemmed'])
    vectorizedTrain = pd.DataFrame(vectorizedWordsTrain.toarray(), columns = vectorizer.get_feature_names())
    vectorizedWordsTest = vectorizer.fit_transform(xTest['stemmed'])
    vectorizedTest = pd.DataFrame(vectorizedWordsTest.toarray(), columns = vectorizer.get_feature_names())
    return vectorizedTrain, vectorizedTest

def hotEncoderWThreshold(xTrain,xTest, threshold, attributeName):
    hotEncoded = pd.get_dummies(xTrain[attributeName])
    count = pd.value_counts(xTrain[attributeName], sort=False, normalize=True) < threshold
    if count.sum() == 0:
        finalEncodedTrain = hotEncoded
    else:
        finalEncodedTrain = hotEncoded.loc[:, ~count].join(hotEncoded.loc[:, count].sum(1).rename(f"Uncommon"))
    cols = list(finalEncodedTrain.columns)
    cols.remove("Uncommon")
    encodedTestData = pd.DataFrame()
    for name in cols:
        encodedTestData[name] = xTest[attributeName] == name
    encodedTestData["Uncommon"] = ~encodedTestData[cols].any(axis= "columns")
    encodedTestData = encodedTestData*1
    encodedTestData, finalEncodedTrain = encodedTestData.add_prefix(f"{attributeName[0:6]}_"), finalEncodedTrain.add_prefix(f"{attributeName[0:6]}_")
    i = xTrain.columns.get_loc(attributeName)
    xTrain = pd.concat([xTrain.iloc[:, :i], 
                finalEncodedTrain, 
                xTrain.iloc[:, i:]], axis=1)
    xTest = pd.concat([xTest.iloc[:, :i], 
                encodedTestData, 
                xTest.iloc[:, i:]], axis=1)
    xTrain, xTest = xTrain.drop(attributeName, axis = 1), xTest.drop(attributeName, axis = 1)
    return(xTrain, xTest)

def main():
    df = readData()
    xTrain, xTest, yTrain, yTest = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size =.3, random_state = 334)
    vocab = findMostCommon(xTrain['stemmed'])
    print(vocab)
    print("Encoding categorical data...")
    xTrain,xTest = hotEncoderWThreshold(xTrain, xTest, 0.04, "province")
    xTrain,xTest = hotEncoderWThreshold(xTrain, xTest, 0.04, "country")
    xTrain,xTest = hotEncoderWThreshold(xTrain, xTest, 0.001, "winery")
    xTrain,xTest = hotEncoderWThreshold(xTrain, xTest, 0.04, "variety")
    vectorizedTrainCount, vectorizedTestCount = vectorizeCount(xTrain, xTest, vocab)
    xTrainCount, xTestCount = xTrain.copy(),xTest.copy()
    xTrainCount[vectorizedTrainCount.columns], xTestCount[vectorizedTestCount.columns] = np.array(vectorizedTrainCount),  np.array(vectorizedTestCount)
    vectorizedTrainBinary, vectorizedTestBinary = vectorizeBinary(xTrain, xTest, vocab)
    xTrainBinary, xTestBinary = xTrain.copy(),xTest.copy()
    xTrainBinary[vectorizedTrainBinary.columns], xTestBinary[vectorizedTestBinary.columns] = np.array(vectorizedTrainBinary),  np.array(vectorizedTestBinary)
    xTrainCount.drop(['stemmed','description'], axis=1, inplace=True),xTestCount.drop(['stemmed','description'], axis=1, inplace=True)
    xTrainBinary.drop(['stemmed','description'], axis=1, inplace=True),xTestBinary.drop(['stemmed','description'], axis=1, inplace=True)
    xTrainCount.to_csv("data/xtrain_count.csv", index=False),xTestCount.to_csv("data/xtest_count.csv", index=False)
    xTrainBinary.to_csv("data/xtrain_binary.csv", index=False),xTestBinary.to_csv("data/xtest_binary.csv", index=False)
    yTrain.to_csv("data/ytrain.csv", index=False), yTest.to_csv("data/ytest.csv", index=False)
if __name__ == "__main__":
    main()