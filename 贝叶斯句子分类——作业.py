from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# 读取数据集 这里是使用的csv格式的数据集文件，
datasets = pd.read_csv("./Dry_Bean_Dataset_Homework.csv")

#使用excel表格数据集文件
#datasets = pd.readexcel("")

pred_dataset = pd.read_csv("./Dry_Bean_Dataset_Prediction.csv")
#pred_dataset = pd.readexcel("")

# 确保预测数据集和训练数据集的列名相同
pred_dataset.columns = datasets.columns[:-1]  # 使用训练集的特征列名

# 假设最后一列是标签，其他列是特征
x = datasets.iloc[:, :-1]  # 取除最后一列外的所有列作为特征
y = datasets.iloc[:, -1]   # 取最后一列作为目标标签

# 将数据集分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0, shuffle=True)

# 应用高斯朴素贝叶斯模型
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)

"""# 输出测试集预测错误的样本数量
print("在总共 %d 个样本中，有 %d 个样本被错误标记" % (x_test.shape[0], (y_test != y_pred).sum()))"""

# 输出预测结果
pred_result = gnb.predict(pred_dataset)
print("预测结果：", pred_result)

# 打开一个txt文件，如果文件不存在则创建
with open('result.txt', 'w') as file:
    # 将打印输出的内容写入文件
    file.write(np.array2string(pred_result, threshold=np.inf))
