#coding=utf-8
'''
@author pcb

Created on April 30, 2016
'''

import numpy as nm

'''
将特征值的取值范围进行转换，防止发生震荡现象
'''
def regularize(xMat):
	#inMat = xMat.copy()
	inMeans = nm.mean(xMat, 0)
	inVar = nm.var(xMat, 0)
	
	
	xMat = (xMat - inMeans) / inVar

	return xMat


def autoNorm(dataSet):
	return dataSet

'''
def autoNorm(dataSet):
	maxVals = dataSet.max(0)
	return dataSet / maxVals
'''
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = nm.zeros(len(dataSet))
    m = len(dataSet)
    normDataSet = dataSet - nm.tile(minVals, (m, 1))
    normDataSet = normDataSet / nm.tile(ranges, (m, 1))  # element wise divide
    normData = normDataSet.tolist()
    for i in range(len(normData)):
    	normData[i][0] = 1
    return nm.mat(normData)
'''

'''
根据文件名读取数据,并自动添加特征值的第一列为1
'''
def loadData(fileName):
	numFeat = len(open(fileName).readline().split(' ')) - 1
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = [1]
		curLine = line.strip().split(' ')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat

'''递归下降求解'''
def solutionDigui(xArr, yArr):
	m = len(xArr[0])
	theta = nm.zeros(shape=(1,m))
	
	alpha = 0.0001
	num_iters = 1000
	J_Array = [0] * num_iters
	m = len(yArr)
	#print m
	thetaMat = nm.mat(theta)
	xArr = regularize(xArr)
	for i in range(m):
		xArr[i][0] = 1
	#print thetaMat
	xMat = nm.mat(xArr)
	yMat = nm.mat(yArr)
	#for i in range(m):
	#	xMat[i][0] = 1
	#print xMat
	#print yMat.shape
	diff = nm.zeros(shape=(1, m))
	for iter in range(num_iters):
		
		diff = thetaMat * xMat.T - yMat
		temp1 = (diff * xMat) / m;
		thetaMat = thetaMat - alpha * temp1

		J_Array[iter] = computeCost(xMat, yMat, thetaMat, m)
	#print J_Array
	return thetaMat.T

'''矩阵的逆求解theta'''
def standRegres(xArr,yArr):
    xMat = nm.mat(xArr); yMat = nm.mat(yArr).T
    xTx = xMat.T*xMat
    if nm.linalg.det(xTx) == 0.0:
        print "该矩阵没有逆矩阵，无法求解"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


'''
求岭回归
'''
def lingSolution(xArr, yArr, lam):
	xMat = nm.mat(xArr)
	yMat = nm.mat(yArr).T
	xTx  = xMat.T*xMat
	xTemp = xTx + nm.eye(nm.shape(xMat)[1]) * lam
	if nm.linalg.det(xTemp) == 0.0:
		print "该矩阵没有逆矩阵，无法求解"
		return
	ws = xTemp.I * (xMat.T*yMat)
	return ws

'''
岭回归的交叉校验
'''
def crossValidationLing(n, lam, xArr, yArr):
	#print yArr1
	num = len(xArr)/n
	err = 0
	#此处应该对不能整除的情况进行处理，先留着，以后再处理
	for i in range(n):
		xArrTrain = []
		xArrTest = []
		yArrTrain = []
		yArrTest = []
		for j in range(n):
			for m in range(num):
				if i == j:
					xArrTest.append(xArr[j*num+m])
					yArrTest.append(yArr[j*num+m])
				else:
					xArrTrain.append(xArr[j*num+m])
					yArrTrain.append(yArr[j*num+m])

		theta = lingSolution(xArrTrain, yArrTrain, lam)
		xArrTestMat = nm.mat(xArrTest)
		#print theta
		yPrediction =  (xArrTestMat * theta).T
		#print yPrediction.shape
		yPreDiction = yPrediction.tolist()
		#print len(yPreDiction[0])
		#print yPreDiction
		diff = []
		for i in range(len(yPreDiction[0])):
			diff.append((yArrTest[i] - yPreDiction[0][i])**2)
		
		diff = nm.mat(diff)
		#print 'diff'
		sumDiff = diff.sum(axis=1)
		errl = sumDiff.tolist()[0][0]
		#print errl
		#print errl
		errlist = errl ** 0.5
		err += (errlist) / num
	return err / n

'''岭回归图的测试'''
def lingSolutionTest(xArr, yArr):
	xMat = nm.mat(xArr)
	yMat = nm.mat(yArr)
	result = nm.zeros((30,nm.shape(xMat)[1]))
	for i in range(30):
		ws = lingSolution(xArr, yArr, nm.exp(i-10))
		result[i, :] = ws.T
	return result


'''
计算J函数
'''
def computeCost(xArr, yArr, theta, m):
	hypothe = theta * xArr.T
	sumofcost = (hypothe - yArr).T * (hypothe - yArr) / (2*m)
	J =  sumofcost
	return J


'''
预测y值,返回预测的y值
'''
def prediction(theta, xArrTest, xArr1, xArr2, diff, k):
	#xArr2的行数
	dataSetSize = len(xArr2)
	matXarr1 = nm.mat(xArrTest)
	#求出两矩阵的欧式距离
	diffMat = nm.tile(xArr1, (dataSetSize, 1)) - xArr2
	#print diffMat
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances**0.5
	#print diff.shape
	diff1 = diff.tolist()
	#print diff1
	#求出欧氏距离从小到大的索引值
	#print diff1[0]
	sortedDistIndicies = distances.argsort()
	sum = 0
	for i in range(k):
		#print sortedDistIndicies[i]
		sum += diff1[0][sortedDistIndicies[i]]
	y = matXarr1 * theta
	#print 'y'
	#print y.shape
	#print 'sum'
	#print sum
	y += (sum / k)
	return y.tolist()[0][0]


'''交叉校验'''
def crossValidation(n, k, xArr, yArr):
	num = len(xArr)/n
	err = 0
	#此处应该对不能整除的情况进行处理，先留着，以后再处理
	for i in range(n):
		xArrTrain = []
		xArrTest = []
		yArrTrain = []
		yArrTest = []
		for j in range(n):
			for m in range(num):
				if i == j:
					xArrTest.append(xArr[j*num+m])
					yArrTest.append(yArr[j*num+m])
				else:
					xArrTrain.append(xArr[j*num+m])
					yArrTrain.append(yArr[j*num+m])

		#分别获得了训练集和检测集
		theta = standRegres(xArrTrain, yArrTrain)
		diff = yArrTrain - (xArrTrain * theta).T
		#print 'diff'
		#print diff.shape
		#xMat = mat(xArr)
		#yMat = mat(yArr)
		#yHat = xMat * ws
#print corrcoef(yHat.T, yMat)
		#xTest, yTest = homework1.loadData('testData-Ass1.txt')
		m = len(xArrTest)
		#print 'm:'
		#print m
		yPrediction = []
		matXtest = nm.mat(xArrTest)
		xArrTest1 = autoNorm(matXtest)
		xArrTest1 = xArrTest1.tolist()
		xArrTrain1 = autoNorm(nm.mat(xArrTrain))
		xArrTrain1 = xArrTrain1.tolist()


#表示前k个最近
		for i in range(m):
			yPrediction.append(prediction(theta, xArrTest[i], xArrTest1[i], xArrTrain1, diff, k))
		#print 'yPreDiction:'
		#print len(yPrediction)
		#print 'yTest:'
		#print len(yArrTest)
		diff2 = []
		#print yPrediction
		for i in range(len(yPrediction)):
			diff2.append((yPrediction[i] - yArrTest[i])**2)

		#diff2 = diff2.toArray()
		#matDiff = nm.mat(diff2)
		#sqDiffMat = diff2**2
		sqDiffMat = nm.mat(diff2)
		sqErr = sqDiffMat.sum(axis = 1)
		#print sqErr
		errl = sqErr.tolist()[0][0]
		#print errl
		errlist = errl**0.5
		err += (errlist / m)
	return err / n





