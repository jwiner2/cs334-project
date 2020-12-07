from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def linearRegressionModel(xFeat, y, xTest, yTest, doCoefs):
    #Linear Regression does not seem to have any Parameters to Set
        #Instead of setting parameters might be best to run LR once with Rige Regularization and Once with Lasso

    #fit and predict with pure LR
    lr= LinearRegression()
    lrFit=lr.fit(xFeat,y)
    yHatLR=lrFit.predict(xTest)
    scoreLR=lrFit.score(xTest,yTest)
    if doCoefs:
         featureCoefs=list(zip(lrFit.coef_[0], xFeat.columns))
         for coef,feat in sorted(featureCoefs):
             print(str(coef)+"\t"+str(feat))

    return (yHatLR,scoreLR)


#todo: make sure this works
def linearRegressionRidgeModel(xFeat, y, xTest, yTest,model):
    #requires tuning of several parameters: 
            #alpha {float, ndarray of shape (n_targets,)}, default=1.0
            #solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}, default=’auto’

    alphaV = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]
    solverV=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    paramsGV={'alpha':alphaV, 'solver':solverV}

    yHatLRR, scoreLRR= gridSearchModel(xFeat,y,xTest,yTest,model,paramsGV)
    return (yHatLRR,scoreLRR)

def linearRegressionLassoModel(xFeat, y, xTest, yTest,model):
    #requires tuning of several parameters:
            #alpha {float, ndarray of shape (n_targets,)}, default=1.0

    alphaV = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]
    paramsGV={'alpha':alphaV}

    yHatLRR, scoreLRR= gridSearchModel(xFeat,y,xTest,yTest,model,paramsGV)
    return (yHatLRR,scoreLRR)


def decTreeRegModel(xFeat, y, xTest, yTest):
    criterionV=['mae', 'mse']
    max_depthV=[5,10,15,20]
    min_samples_leafV=[10,20,100,200,500,1000]
    paramsGV={'criterion':criterionV,'max_depth':max_depthV,'min_samples_leaf':min_samples_leafV}

    model = DecisionTreeRegressor()

    yHatLRR, scoreLRR= gridSearchModel(xFeat,y,xTest,yTest,model,paramsGV)
    return (yHatLRR,scoreLRR)

def perceptronModel(xFeat,y,xTest,yTest):
    #possibly add in other perams Max Iter
    penaltyV=['l2', 'l1']
    alphaV = [0.01, 0.1, 1, 10]

    paramsGV = {'alpha': alphaV, 'penalty':penaltyV}

    model=Perceptron()

    yHatLRR, scoreLRR, r2LRR= gridSearchModel(xFeat,y,xTest,yTest,model,paramsGV)
    return (yHatLRR,scoreLRR)



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
    # mseScore= mean_squared_error(yTest, yHatGS)
    # print(mseScore)

    return (yHatGS,scoreGS)

def corralte(xTrain,yTrain):
    df = pd.concat([xTrain, yTrain], axis=1)
    print(df)
    pearson = df.corr()
    hm1 = sns.heatmap(pearson, vmin=-1, vmax=1, center=0, cmap="PuRd")
    plt.title("Corralation of Features with the Label")
    plt.xticks(rotation=60, ha='right',fontsize=9)
    plt.yticks(rotation=40, ha='right',fontsize=9)

    plt.show()


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
    ### Used to set up data files
    #set up the program to take in arguments from the command line
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--vectorF",
    #                     help="filename for features of the training data",
    #                     default="data/vectorized_data.csv") #todo: need to change to correct file
    # args = parser.parse_args()
    # xFull = pd.read_csv(args.vectorF)
    # yFull = xFull["price"]
    # xFull2= xFull.iloc[:,16:]
    # xFull2["points"]=xFull["points"]
    #
    # xTrain,xTest,yTrain,yTest= train_test_split( xFull2, yFull, test_size=0.33, random_state=42)
    #
    # xTrain.to_csv("data/xTrain.csv", index=False)
    # yTrain.to_csv("data/yTrain.csv", index=False)
    # xTest.to_csv("data/xTest.csv", index=False)
    # yTest.to_csv("data/yTest.csv", index=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        help="filename for features of the training data",
                        default="data/xtrain_count.csv")
    parser.add_argument("--yTrain",
                        help="filename for labels associated with training data",
                        default="data/ytrain.csv")
    parser.add_argument("--xTest",
                        help="filename for features of the test data",
                        default="data/xtest_count.csv")
    parser.add_argument("--yTest",
                        help="filename for labels associated with the test data",
                        default="data/ytest.csv")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)


    # corralte(xTrain,yTrain)



    #the linear regression as well as PCA are having a hard time working with the catagorical data
    #run pca on data
    xTrainPCA, xTestPCA = pcaCreate(xTrain,yTrain,xTest,yTest)

    # #run Linear regression
    # #without PCA
    # yHatLR,scoreLR=linearRegressionModel(xTrain, yTrain,xTest,yTest,True)
    # print("LR Score \t"+str(scoreLR))
    #
    # #with PCA
    # yHatLR,scoreLR=linearRegressionModel(xTrainPCA, yTrain,xTestPCA,yTest,False)
    # print("LR Score with PCA \t"+str(scoreLR))

    # # #run Linear Regression Ridge
    #
    # yHatLRRidge,scoreLRRidge=linearRegressionRidgeModel(xTrain, yTrain,xTest,yTest, Ridge())
    # print("LR Ridge Score \t" + str(scoreLRRidge))
    #
    # #with PCA
    # yHatLRRidge,scoreLRRidge=linearRegressionRidgeModel(xTrainPCA, yTrain,xTestPCA,yTest, Ridge())
    # print("LR Ridge Score with PCA \t" + str(scoreLRRidge))
    #
    #
    # # #run Linear Regression Lasso
    # yHatLRLasso,scoreLRLasso=linearRegressionLassoModel(xTrain, yTrain,xTest,yTest, Lasso())
    # print("LR Lasso Score \t" + str(scoreLRLasso))
    #
    # yHatLRLasso,scoreLRLasso=linearRegressionLassoModel(xTrainPCA, yTrain,xTestPCA,yTest, Lasso())
    # print("LR Lasso Score with PCA \t" + str(scoreLRLasso))

    #decision tree regressor
    # yHatDTR, scoreDTR = decTreeRegModel(xTrain, yTrain, xTest, yTest)
    # temp= DecisionTreeRegressor(max_depth=5)
    # temp.fit(xTrain,yTrain)
    # scoreDTR=temp.score(xTest,yTest)
    # print("DTR Score \t" + str(scoreDTR))
    # featureCoefs = list(zip(temp.feature_importances_, xTrain.columns))
    # for coef, feat in sorted(featureCoefs):
    #     print(str(coef) + "\t" + str(feat))

    # yHatDTR, scoreDTR = decTreeRegModel(xTrainPCA, yTrain, xTestPCA, yTest)
    # temp= DecisionTreeRegressor(max_depth=5)
    # temp.fit(xTrainPCA,yTrain)
    # scoreDTR=temp.score(xTestPCA,yTest)
    # print("DTR Score with PCA \t" + str(scoreDTR))

    # #perceptron #todo: is asking to be ravelled
    yHatPer, scorePer = perceptronModel(xTrainPCA, yTrain, xTestPCA, yTest)
    print("Perceptron Score \t" + str(scorePer))





if __name__ == "__main__":
    main()