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
for j in range(20):
	yPrediction = []
	for i in range(m):
		yPrediction.append(homework1.prediction(ws, xTest[i], xTest1[i], xArr1, diff, j+1))
	diff2 = []
	for i in range(100):
		diff2.append((yTest[i] - yPrediction[i]) ** 2);
	sqDiffMat = mat(diff2)
	sqErr = sqDiffMat.sum(axis = 1)
		#print sqErr
	errl = sqErr.tolist()[0][0]**0.5
	xlist.append(j+1)
	result.append(errl)


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