from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import argparse

"""
PCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
def pcaCreate(xFeat, y, xTest, yTest):
    pca = PCA(n_components=0.95)
    pca.fit(xFeat,y)
    newPCA= pca.transform(xFeat)
    newPCATest=pca.transform(xTest)

    #Discuss what characterizes the 3 principal components (i.e., which original features are important).
    indexes=[]
    for i in range(len(newPCA[0])):
        indexes.append("PC-"+str(i))
    print(pd.DataFrame(pca.components_,columns=xFeat.columns,index =indexes))

    #Return PCA
    return (newPCA, newPCATest)

"""
Linear Regression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

Lasso and Ridge
https://scikit-learn.org/stable/modules/linear_model.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
"""

def linearRegressionModel(xFeat, y, xTest, yTest):
    #Linear Regression does not seem to have any Parameters to Set
        #Instead of setting parameters might be best to run LR once with Rige Regularization and Once with Lasso

    #fit and predict with pure LR
    lr= LinearRegression()
    lrFit=lr.fit(xFeat,y)
    yHatLR=lrFit.predict(xTest)
    scoreLR=lrFit.score(xTest,yTest)

    return (yHatLR,scoreLR)


def linearRegressionRidgeModel(xFeat, y, xTest, yTest):
    #requires tuning of several parameters: 
            #alpha {float, ndarray of shape (n_targets,)}, default=1.0
            #solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}, default=’auto’

    

    #fit and predict with LR Ridge
    lr= LinearRegression()
    lrFit=lr.fit(xFeat,y)
    yHatLR=lrFit.predict(xTest)
    scoreLR=lrFit.score(xTest,yTest)

    return (yHatLR,scoreLR)






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
    xTrainPCA, xTestPCA = pcaCreate(xTrain,yTrain,xTest,yTest)


    #run Linear regression
    yHatLR,scoreLR=linearRegressionModel(xTrainPCA, yTrain,xTestPCA,yTest)
    #run Linear Regression Ridge
    yHatLRRidge,scoreLRRidge=linearRegressionRidgeModel(xTrainPCA, yTrain,xTestPCA,yTest)


if __name__ == "__main__":
    main()