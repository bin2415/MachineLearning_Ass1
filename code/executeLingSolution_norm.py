#coding=utf-8
import homework1
from numpy import *
xArr, yArr = homework1.loadData('traningData-Ass1.txt')



bestLam = 0
err = 999999999
errlist = []
xlist = []
# 规范化处理
xArrNorm = homework1.autoNorm(mat(xArr)).tolist()
yArrNorm = homework1.autoNorm(mat(yArr)).tolist()[0]
#print yArrNorm
for i in range(30):
	err1 = homework1.crossValidationLing(5, exp(i-15), xArrNorm, yArrNorm)
	#print exp(i - 10)
	xlist.append(i)
	errlist.append(err1)
	if err > err1:
		err = err1
		bestLam = i

print bestLam



xArr2, yArr2 = homework1.loadData('testData-Ass1.txt')
theta = homework1.lingSolution(xArr, yArr, bestLam)
yPrediction = (xArr2 * theta).T
print yPrediction

yPreDiction = yPrediction.tolist()
diff = []
for i in range(len(yPreDiction[0])):
	diff.append((yArr2[i] - yPreDiction[0][i]) ** 2)

diff = mat(diff)
sumDiff = diff.sum(axis=1)
err1 = sumDiff.tolist()[0][0]

print err1

import matplotlib.pyplot as plt

xMat = mat(xArr2)
yMat = mat(yArr2)
fig = plt.figure()
ax1 = fig.add_subplot(331)
ax1.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax1.plot(xCopy[:,1], yPredictionList[0])

ax2 = fig.add_subplot(332)

ax2.scatter(xMat[:,2].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax2.plot(xCopy[:,2], yPredictionList[0])

ax3 = fig.add_subplot(333)

ax3.scatter(xMat[:,3].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax3.plot(xCopy[:,3], yPredictionList[0])

ax4 = fig.add_subplot(334)
ax4.scatter(xMat[:,4].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax4.plot(xCopy[:,4], yPredictionList[0])

ax5 = fig.add_subplot(335)
ax5.scatter(xMat[:,5].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax5.plot(xCopy[:,5], yPredictionList[0])

ax6 = fig.add_subplot(336)
ax6.scatter(xMat[:,6].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax6.plot(xCopy[:,6], yPredictionList[0])

ax7 = fig.add_subplot(337)
ax7.scatter(xMat[:,7].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax7.plot(xCopy[:,7], yPredictionList[0])

ax8 = fig.add_subplot(338)
ax8.scatter(xMat[:,8].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax8.plot(xCopy[:,8], yPredictionList[0])

ax9 = fig.add_subplot(339)
ax9.scatter(xMat[:,9].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax9.plot(xCopy[:,9], yPredictionList[0])

plt.show()



'''
#岭回归图测试
ridgeWeights = homework1.lingSolutionTest(xArr, yArr)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()
'''