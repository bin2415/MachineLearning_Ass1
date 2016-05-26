#coding=utf-8
import homework1
from numpy import *
xArr, yArr = homework1.loadData('traningData-Ass1.txt')


##交叉校验求得最好的lamba
bestLam = 0
err = 999999999
errlist = []
for i in range(30):
	err1 = homework1.crossValidationLing(5, exp(i-15), xArr, yArr)
	#print exp(i - 10)
	print i
	print err1
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

fig = plt.figure()
ax = fig.add_subplot(111)
xMat = mat(xArr2)
yMat = mat(yArr2)
ax.scatter(xMat[:,9].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xArr2)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = (xCopy * theta).T
yPredictionList = yPrediction.tolist()
ax.plot(xCopy[:,9], yPredictionList[0])
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