import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import  KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("../car.data")
"""print(data.head())"""

#transform irregular data into integer types
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform((list(data["safety"])))
cls = le.fit_transform(list(data["class"]))\

predict = "class"

#turn data into lists,
#zip makes all the separate lists into 1 list
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

#select data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#make model using K nearest neighbors
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

#test for accuracy
acc = model.score(x_test, y_test)
"""print(acc)"""

#predict results of rest of data and print
predictions = model.predict(x_test)
names = ["unacc","acc","good","vgood"]

for x in range(len(predictions)):
    print("Predicted:",names[predictions[x]],"Data:",x_test[x],"Actual",names[y_test[x]])