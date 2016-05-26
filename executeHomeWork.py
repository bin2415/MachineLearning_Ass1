#coding=utf-8
import homework1
from numpy import *
xArr, yArr = homework1.loadData('traningData-Ass1.txt')



xArr1 = homework1.autoNorm(mat(xArr))
xArr1 = xArr1.tolist()
#for i in range(len(xArr)):
#	xArr1[i][0] = 1
#print xArr1


#找出k的大小

bestK = 0
err = 9999999
errlist = []
xlist = []
for i in range(20):
	err1 = homework1.crossValidation(5, i+1, xArr, yArr)
	xlist.append(i+1)
	errlist.append(err1)
	if err > err1:
		err = err1
		bestK = i+1

print bestK


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
yPrediction = []
matXtest = mat(xTest)

xTest1 = homework1.autoNorm(mat(xTest))
xTest1 = xTest1.tolist()
#xTest1 = xTest
#for i in range(len(xTest1)):
#	xTest1[i][0] = 1

#print xTest1
#表示前k个最近
k = bestK
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework1.prediction(ws, xTest[i], xTest1[i], xArr1, diff, k))

for i in range(m):
	print yPrediction[i]


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

'''
fig = plt.figure()
ax = fig.add_subplot(111)
yMat = mat(yTest)
xMat = mat(linspace(1,100,100))
sortedIns = yMat.T.argsort()
yMat.T.sort(0)
#print yMat
ax.scatter(xMat.T[:,0].flatten().A[0], yMat.T[:,0].flatten().A[0])

yPreMat = mat(yPrediction)

diff2 = []
for i in range(100):
	diff2.append((yTest[i] - yPrediction[i]) ** 2);
sqDiffMat = mat(diff2)
sqErr = sqDiffMat.sum(axis = 1)
		#print sqErr
errl = sqErr.tolist()[0][0]
		#print errl
print errl


yPreMat = mat(yPrediction)
yPreMat.T.sort(0)
ax.plot(xMat.T[:,0].flatten().A[0], yPreMat.T[:, 0].flatten().A[0])
plt.show()
'''

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,2].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework1.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax.plot(xCopy[:,2], yPrediction)
plt.show()
