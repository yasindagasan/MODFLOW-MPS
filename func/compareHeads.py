import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

def compareHeads(trueFlow,realFlow,index, output="RMSE", threshold=0.1):
    """
    ~~Yasin~~
    Gets the true and simulated flow fields and compares the head values:

    :trueIm     : True head measurements (Img class)
    :simIm      : Simulated head measurements (Img class)
    :index      : Index values for the measurement locations (Arr)

    :param output:
            1) RMSE
            2) MAE
            3) booleanRMSE (True or False image)
            4) vectSubt
            5) distL1
            6) distL2
            7) imgVecSub
            8) vecHeads

    :return: errors based on the param output (array)
    """

    # get the head measurements for both imageContStat
    headTrueK = trueFlow.val[0,0,index[:,0],index[:,1]]

    if output=="RMSE" or output==1:
        # Create an array of zero to store the MSE values for each realisation
        rmseFlow = np.zeros((realFlow.val.shape[0],))
        # Calculate the MSE values for each of the realisations and return a vector
        for i in range(realFlow.val.shape[0]):
            headRealK = realFlow.val[i,0,index[:,0],index[:,1]]
            rmse = sqrt(mean_squared_error(headTrueK, headRealK))
            rmseFlow[i,] = rmse
        return rmseFlow

    elif output=="MAE" or output==2:
        # Create an array of zeros to store the MAE values
        maeFlow = np.zeros((realFlow.val.shape[0],))
        # Calculate the MAE values for each of the realisations and return a vector
        for i in range(realFlow.val.shape[0]):
            headRealK = realFlow.val[i,0,index[:,0],index[:,1]]
            mae = mean_absolute_error(headTrueK, headRealK)
            maeFlow[i,] = mae
        return maeFlow

    elif output=="booleanRMSE" or output==3:
        # Create an array of zero to store the MSE values for each realisation
        classedFlow = np.ones((realFlow.val.shape[0],), dtype=bool) # Classified vector
        # Calculate the MSE values for each of the realisations and return a vector
        for i in range(realFlow.val.shape[0]):
            headRealK = realFlow.val[i,0,index[:,0],index[:,1]]
            rmse = sqrt(mean_squared_error(headTrueK, headRealK))
            if (rmse > threshold):
                classedFlow[i,] = False
            else:
                classedFlow[i,] = True
        trueValues  = sum(classedFlow[:,])
        falseValues = np.size(classedFlow) - np.count_nonzero(classedFlow)
        print("*********************************")
        print("Number of classes created\n---------------------------------")
        print("True:", trueValues, "\nFalse:",falseValues)
        print("**********************************")

        return classedFlow

    elif output=="vectSubt" or output==4:
        # Create an array of zero to store the differences between the head values
        # at each measurement point
        headSubtract = np.zeros((realFlow.val.shape[0],len(index)))

        for i in range(realFlow.val.shape[0]):
            headRealK = realFlow.val[i,0,index[:,0],index[:,1]]
            headSubtract[i,] = headTrueK-headRealK

        return headSubtract

    elif output=="distL1" or output==5:
        norm = np.zeros((realFlow.val.shape[0],))

        for i in range(realFlow.val.shape[0]):
            headRealK = realFlow.val[i,0,index[:,0],index[:,1]]
            dist = np.linalg.norm((headTrueK - headRealK), ord=1)
            norm[i,] = dist
        return norm
    elif output=="distL2" or output==6:
        norm = np.zeros((realFlow.val.shape[0],))

        for i in range(realFlow.val.shape[0]):
            headRealK = realFlow.val[i,0,index[:,0],index[:,1]]
            dist = np.linalg.norm((headTrueK - headRealK), ord=2)
            norm[i,] = dist
        return norm
    elif output=="vectHead" or output==7:
        # Create an array of zero to store the differences between the head values
        # at each measurement point
        headReals = np.zeros((realFlow.val.shape[0],len(index)))

        for i in range(realFlow.val.shape[0]):
            headReals[i,] = realFlow.val[i,0,index[:,0],index[:,1]]
        return headReals
