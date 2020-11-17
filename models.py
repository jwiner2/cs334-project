from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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

https://towardsdatascience.com/linear-regression-models-4a3d14b8d368
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


#todo: make sure this works
def linearRegressionRidgeModel(xFeat, y, xTest, yTest, models,params):
    #requires tuning of several parameters: 
            #alpha {float, ndarray of shape (n_targets,)}, default=1.0
            #solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}, default=’auto’

    alphaV = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]
    solverV=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    paramsGV={'alpha':alphaV, 'solver':solverV}

    model= Ridge()

    yHatLRR, scoreLRR= gridSearchModel(xFeat,y,xTest,yTest,model,paramsGV)
    return (yHatLRR,scoreLRR)


#todo: make sure this works
def gridSearchModel(xFeat, y, xTest, yTest, model,params):
    #grid search
    gridCV = GridSearchCV(estimator=model, param_grid=params, scoring='r2', verbose=1, n_jobs=-1)

    #fit
    gridCV.fit(xFeat, y)

    # extract best estimator
    print(gridCV.best_estimator_)

    #predict and score
    yHatGS=gridCV.predict(xTest)
    scoreGS=gridCV.score(xTest, yTest)

    return (yHatGS,scoreGS)


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
    #### Used to set up data files
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorF",
                        help="filename for features of the training data",
                        default="data/vectorized_data.csv") #todo: need to change to correct file
    args = parser.parse_args()
    xFull = pd.read_csv(args.vectorF)
    yFull = xFull["price"]
    xFull= xFull.iloc[:,16:]

    xTrain,xTest,yTrain,yTest= train_test_split( xFull, yFull, test_size=0.33, random_state=42)

    xTrain.to_csv("data/xTrain.csv", index=False)
    yTrain.to_csv("data/yTrain.csv", index=False)
    xTest.to_csv("data/xTest.csv", index=False)
    yTest.to_csv("data/yTest.csv", index=False)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--xTrain",
    #                     help="filename for features of the training data",
    #                     default="data/xTrain.csv") #todo: need to change to correct file
    # parser.add_argument("--yTrain",
    #                     help="filename for labels associated with training data",
    #                     default="data/yTrain.csv")#todo: need to change to correct file
    # parser.add_argument("--xTest",
    #                     help="filename for features of the test data",
    #                     default="data/xTest.csv")#todo: need to change to correct file
    # parser.add_argument("--yTest",
    #                     help="filename for labels associated with the test data",
    #                     default="data/yTest.csv")#todo: need to change to correct file
    #
    # args = parser.parse_args()
    # # load the train and test data assumes you'll use numpy
    # xTrain = file_to_numpy(args.xTrain)
    # yTrain = file_to_numpy(args.yTrain)
    # xTest = file_to_numpy(args.xTest)
    # yTest = file_to_numpy(args.yTest)


    #the linear regression as well as PCA are having a hard time working with the catagorical data
    #run pca on data
    # xTrainPCA, xTestPCA = pcaCreate(xTrain,yTrain,xTest,yTest)

    # #run Linear regression
    # yHatLR,scoreLR=linearRegressionModel(xTrain, yTrain,xTest,yTest)
    # #run Linear Regression Ridge
    # yHatLRRidge,scoreLRRidge=linearRegressionRidgeModel(xTrainPCA, yTrain,xTestPCA,yTest)


if __name__ == "__main__":
    main()