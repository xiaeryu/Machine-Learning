# -*- coding: utf-8 -*-

import numpy as np
import operator

def kNN(trainSet,trainLabel,realItem,k):
        # Calculate distance
        trainSize = trainSet.shape[0]
        diffMat = np.tile(realItem,(trainSize,1)) - trainSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5

        # Sort distance
        sortedDistIndicies = distances.argsort()

        # Get the k nearest neighbors and conduct majority voting
        classCount={}
        for i in range(k):
                voteIlabel = trainLabel[sortedDistIndicies[i]]
                classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
                sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
