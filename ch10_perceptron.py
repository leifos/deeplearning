import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from datetime import datetime

def main():
    iris = load_iris()
    X = iris.data[:,(2,3)] # petal length, petal width
    y = (iris.target == 0 ).astype(np.int) # Iris Setosa??

    per_clf = Perceptron(random_state=42,max_iter=5,tol=0.2) # meaning of life?

    per_clf.fit(X,y)
    y_pred = per_clf.predict([[2,0.5]])
    print(y_pred)



if __name__ == "__main__":
    main()
