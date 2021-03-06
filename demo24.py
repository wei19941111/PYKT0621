
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean
from sklearn.svm import SVC

# prepare data
iris = datasets.load_iris()
features = iris.data
label = iris.target

regression1 = LogisticRegression()
svc1 = SVC(kernel='linear')
svc2 = SVC(kernel='poly')
svc3 = SVC(kernel='rbf')
classifiers = [regression1, svc1, svc2, svc3]
for c in classifiers:
    score = cross_val_score(c, features, label, cv=5)
    print(f"using {c}, \n ..score={score}, \n ..average={mean(score):.2f}")