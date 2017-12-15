from sklearn import svm

X=[[2,0],[1,1],[2,3]]
Y=[0,0,1]
clf=svm.SVC(kernel="linear")
clf.fit(X,Y)
print(clf.support_vectors_)