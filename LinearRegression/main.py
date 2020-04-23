import sklearn
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G1"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# create best fit line based on data

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
linear = linear_model.LinearRegression()

best = 0
"""
for _ in range(100):
    #create best fit line based on data

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    #accuracy score
    acc = linear.score(x_test, y_test)
    print(acc)

    # save it
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#calculated coefficients based on best fit
print('Coefficient', linear.coef_,
      '\nIntercept', linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#graph it
p = 'failures'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()