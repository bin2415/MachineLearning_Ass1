#coding=utf-8
import homework1
from numpy import *
xArr, yArr = homework1.loadData('traningData-Ass1.txt')



xArr1 = homework1.autoNorm(mat(xArr))
xArr1 = xArr1.tolist()
#for i in range(len(xArr)):
#	xArr1[i][0] = 1
#print xArr1



#递归下降
#ws = homework1.solutionDigui(xArr, yArr)
#矩阵的逆
ws = homework1.standRegres(xArr, yArr)
diff = yArr - (xArr * ws).T
#print ws.shape
#print 'diff'
#print diff.shape
print diff
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws
#print corrcoef(yHat.T, yMat)
xTest, yTest = homework1.loadData('testData-Ass1.txt')
m = len(xTest)

matXtest = mat(xTest)

xTest1 = homework1.autoNorm(mat(xTest))
xTest1 = xTest1.tolist()
#xTest1 = xTest
#for i in range(len(xTest1)):
#	xTest1[i][0] = 1

#print xTest1
#表示前k个最近
result = []
xlist = []
for i in range(30):
	theta = homework1.lingSolution(xArr, yArr, exp(i-10))
	yPrediction = (xArr * theta).T

	yPreDiction = yPrediction.tolist()
	diff = []
	for j in range(len(yPreDiction[0])):
		diff.append((yArr[j] - yPreDiction[0][j]) ** 2)

	diff = mat(diff)
	sumDiff = diff.sum(axis=1)
	err1 = sumDiff.tolist()[0][0]**0.5
	xlist.append(i+1)
	#print exp(i - 10)
	result.append(err1)



'''
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:,0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()
'''
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xlist, result)
plt.show()