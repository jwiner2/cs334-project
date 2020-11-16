from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import argparse



"""
[Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
   - There are not many parameters to set
   - I dont think feature extraction will work well on our catagorical dataset
"""


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)

    return df.to_numpy()

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        help="filename for features of the training data",
                        default="q4xTrain.csv") #todo: need to change to correct file
    parser.add_argument("--yTrain",
                        help="filename for labels associated with training data",
                        default="q4yTrain.csv")#todo: need to change to correct file
    parser.add_argument("--xTest",
                        help="filename for features of the test data",
                        default="q4xTest.csv")#todo: need to change to correct file
    parser.add_argument("--yTest",
                        help="filename for labels associated with the test data",
                        default="q4yTest.csv")#todo: need to change to correct file

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)


    #run pca on data

if __name__ == "__main__":
    main()