#coding=utf-8
import homework
from numpy import *
xArr, yArr = homework.loadData('traningData-Ass1.txt')



xArr1 = homework.autoNorm(mat(xArr))
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
	err1 = homework.crossValidation(5, i+1, xArr, yArr)
	xlist.append(i+1)
	errlist.append(err1)
	if err > err1:
		err = err1
		bestK = i+1

print bestK


#递归下降
#ws = homework.solutionDigui(xArr, yArr)
#矩阵的逆
ws = homework.standRegres(xArr, yArr)
diff = yArr - (xArr * ws).T
#print ws.shape
#print 'diff'
#print diff.shape
print diff
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws
#print corrcoef(yHat.T, yMat)
xTest, yTest = homework.loadData('testData-Ass1.txt')
m = len(xTest)
yPrediction = []
matXtest = mat(xTest)

xTest1 = homework.autoNorm(mat(xTest))
xTest1 = xTest1.tolist()
#xTest1 = xTest
#for i in range(len(xTest1)):
#	xTest1[i][0] = 1

#print xTest1
#表示前k个最近
k = bestK
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest1[i], xArr1, diff, k))

for i in range(m):
	print yPrediction[i]


diff1 = []
for i in range(len(yPrediction)):
	diff1.append((yTest[i] - yPrediction[i]) ** 2)

diff1 = mat(diff1)
sumDiff = diff1.sum(axis=1)
err1 = sumDiff.tolist()[0][0]

print err1

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
ax1 = fig.add_subplot(331)
ax1.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax1.plot(xCopy[:,1], yPrediction)

ax2 = fig.add_subplot(332)
ax2.scatter(xMat[:,2].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax2.plot(xCopy[:,2], yPrediction)

ax3 = fig.add_subplot(333)
ax3.scatter(xMat[:,3].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax3.plot(xCopy[:,3], yPrediction)

ax4 = fig.add_subplot(334)
ax4.scatter(xMat[:,4].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax4.plot(xCopy[:,4], yPrediction)

ax5 = fig.add_subplot(335)
ax5.scatter(xMat[:,5].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax5.plot(xCopy[:,5], yPrediction)

ax6 = fig.add_subplot(336)
ax6.scatter(xMat[:,6].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax6.plot(xCopy[:,6], yPrediction)

ax7 = fig.add_subplot(337)
ax7.scatter(xMat[:,7].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax7.plot(xCopy[:,7], yPrediction)

ax8 = fig.add_subplot(338)
ax8.scatter(xMat[:,8].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax8.plot(xCopy[:,8], yPrediction)

ax9 = fig.add_subplot(339)
ax9.scatter(xMat[:,9].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = mat(xTest)
xCopy.sort(0)
xTest = xCopy.tolist()
yPrediction = []
for i in range(m):
	#print 'i:'; print i
	yPrediction.append(homework.prediction(ws, xTest[i], xTest[i], xArr1, diff, k))
yHat = mat(yPrediction)
print xCopy.shape
ax9.plot(xCopy[:,9], yPrediction)

plt.show()
