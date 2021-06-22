import numpy as np
from sklearn import linear_model, datasets

# 這是一個糖尿病的資料集，主要包括442行資料，10個屬性值，
# 分別是：Age(年齡)、性別(Sex)、Body mass index(體質指數)、
# Average Blood Pressure(平均血壓)、
# S1~S6一年後疾病級數指標。
# Target為一年後患疾病的定量指標。
diabetes = datasets.load_diabetes()
print(type(diabetes))
print("feature", diabetes.data.shape)
print("label", diabetes.target.shape)

dataForTest = -60
#training data
data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print("feature for training", data_train.shape)
print("label for training", target_train.shape)
#Test data
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print("feature for testing", data_test.shape)
print("label for testing", target_test.shape)


regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(f"coef={regression1.coef_}")
print(f"intercept={regression1.intercept_}")

# calculate score 計算分數
print(f"score={regression1.score(data_test, target_test)}")

# predict
for i in range(dataForTest, 0):
    data = data_test[i]
    # print(data.shape)
    data = data.reshape(1, -1)
    print(f"predict={regression1.predict(data)[0]:.2f}, actual={target_test[i]:.2f}")
# mean square error
MSE = np.mean((regression1.predict(data_test) - target_test) ** 2)
print(f"MSE={MSE}")