#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.5.2
@author: beibei zhang
@license: Apache Licence 
@contact: 695193839@qq.com
@site: http://www.baidu.com
@software: PyCharm
@file: ImmuneScorePrediction.py
@time: 2024/2/28 22:17
"""
# # # ##**********************************1.lasso**************************************
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

# 读入数据
data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\HealthScore.csv')
ids = data['ID']
# # 提取特征列
#X = data.iloc[:, 5:]  # 从第五列开始是特征列
# 读取数据
feature = pd.read_csv('result_number_name.csv', header=None)
# 1.按条件筛选特征
selected_features = []
for index, row in feature.iterrows():
    if row[4] > 0.95 and row[3] in data.columns:
        selected_features.append(row[3])
# 添加GDF15和IL34到selected_features列表
selected_features.append("GDF15")
selected_features.append("IL34")
# 提取特征列
X = data[selected_features]
# 提取目标变量
y = data['Score']
# 提取Age列
age = data['Age']
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test, age_train, age_test, ids_train, ids_test = train_test_split(X_scaled, y, age,ids, test_size=0.2, random_state=42)#None
# 创建LassoCV模型，cv参数为交叉验证折数
lasso_cv = LassoCV(cv=5)
# 拟合模型
lasso_cv.fit(X_train, y_train)
#print('lasso result:')
print("最优的 alpha 值：", lasso_cv.alpha_)

# 在测试集上评估模型
y_pred = lasso_cv.predict(X_test)
#y_pred = np.round(y_pred).astype(int)##整数
mae = mean_absolute_error(y_test, y_pred)
print("测试集MAE：", mae)
# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print("测试集MSE：", mse)
# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print("测试集RMSE：", rmse)
# 计算决定系数（R^2 Score）
r2 = r2_score(y_test, y_pred)
print("测试集R^2 Score：", r2)

spearman_corr, spearman_pval = spearmanr(y_test, y_pred)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(y_test, y_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)

# 绘制预测结果与真实值的图像
plt.plot(y_test, y_test, color='red', label='True Values')
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Lasso Regression - True vs Predicted Values")
plt.legend(loc='lower right')
# 在图上标出Pearson相关系数的值
plt.text(0.2, 0.9, f'Pearson corr: {pearson_corr:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
# # 保存图片为PDF格式
plt.savefig('Lasso_true_prediction.svg', format='svg')
plt.savefig('Lasso_true_prediction.pdf', format='pdf')
plt.show()  # 显示图像

# 将预测值、真实值和Age列的数值组合成一个DataFrame
results = pd.DataFrame({'ID': ids_test, 'Age': age_test, 'True_Values': y_test, 'Predictions': y_pred})
# 根据Age列的数值进行排序
results = results.sort_values('Age')
#print(results)
results.to_csv('Lasso_Feature_y_pred_y_true.csv', index=False)

# 计算Pearson相关系数
pearson_corr1, pearson_pval1= pearsonr(age_test, y_pred)
print("Pearson相关系数:", pearson_corr1)
print("Pearson p-value:", pearson_pval1)
# # 绘制预测结果与真实值的图像
plt.plot(results['Age'], results['True_Values'], color='red', label='True Values')
# 在图上标出Pearson相关系数的值
plt.text(0.2, 0.2, f'Pearson corr: {pearson_corr1:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(results['Age'], results['Predictions'], color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("Lasso Regression - True vs Predicted Values")
plt.legend()
##保存图片为PDF格式
plt.savefig('Lasso_age_score.svg', format='svg')
plt.savefig('Lasso_age_score.pdf', format='pdf')
plt.show()  # 显示图像
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': lasso_cv.coef_})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
#print(feature_importance_df)
feature_importance_df.to_csv('Lasso_Feature_Importance.csv', index=False)
#

# # # # # #****************************2.DecisionTree*******************
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

# 读入数据
data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\HealthScore.csv')
ids = data['ID']
# 1.按条件筛选特征
selected_features = []
for index, row in feature.iterrows():
    if row[4] > 0.95 and row[3] in data.columns:
        selected_features.append(row[3])
# 添加GDF15和IL34到selected_features列表
selected_features.append("GDF15")
selected_features.append("IL34")
# 读取符合条件的特征对应的数据
X = data[selected_features]
# 提取目标变量
y = data['Score']
# 提取Age列
age = data['Age']
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test, age_train, age_test, ids_train, ids_test = train_test_split(X_scaled, y, age,ids, test_size=0.2, random_state=42)
# 创建决策树模型
dt = DecisionTreeRegressor(criterion='mse',  # 使用均方误差作为分割质量准则
    splitter='best',  # 选择最佳分割点
    max_depth=4,  # 限制树的最大深度为5
    min_samples_split=2,  # 进行分割所需的最小样本数量为2
    min_samples_leaf=1,  # 叶节点上的最小样本数量为1
    max_features=None,  # 考虑全部特征进行分割
    random_state=42  # 设置随机种子为42

)
# 拟合模型
dt.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = dt.predict(X_test)
#y_pred = np.round(y_pred).astype(int)
mae = mean_absolute_error(y_test, y_pred)
print('decision tree result:')
print("测试集MAE：", mae)
# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print("测试集MSE：", mse)
# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print("测试集RMSE：", rmse)
# 计算决定系数（R^2 Score）
r2 = r2_score(y_test, y_pred)
print("测试集R^2 Score：", r2)
spearman_corr, spearman_pval = spearmanr(y_test, y_pred)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(y_test, y_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
#

# 绘制预测结果与真实值的图像
plt.plot(y_test, y_test, color='red', label='True Values')
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Decision Tree - True vs Predicted Values")
plt.legend(loc='lower right')
plt.text(0.2, 0.9, f'Pearson corr: {pearson_corr:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
# # 保存图片为SVG格式
# # 保存图片为PDF格式
plt.savefig('Decision_tree_true_prediction.svg', format='svg')
plt.savefig('Decision_tree_true_prediction.pdf', format='pdf')
plt.show()  # 显示图像

# 将预测值、真实值和Age列的数值组合成一个DataFrame
results = pd.DataFrame({'ID': ids_test, 'Age': age_test, 'True_Values': y_test, 'Predictions': y_pred})
# 根据Age列的数值进行排序
results = results.sort_values('Age')
results.to_csv('Decision_tree_Feature_y_pred_y_true.csv', index=False)


# 计算Pearson相关系数
pearson_corr1, pearson_pval1= pearsonr(age_test, y_pred)
print("Pearson相关系数:", pearson_corr1)
print("Pearson p-value:", pearson_pval1)
# # 绘制预测结果与真实值的图像
plt.plot(results['Age'], results['True_Values'], color='red', label='True Values')
plt.text(0.2, 0.2, f'Pearson corr: {pearson_corr1:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(results['Age'], results['Predictions'], color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("Decision_tree Regression - True vs Predicted Values")
plt.legend()
# 保存图片为PDF格式
plt.savefig('Decision_tree_age_score.svg', format='svg')
plt.savefig('Decision_tree_age_score.pdf', format='pdf')
plt.show()  # 显示图像
# 获取特征重要性
feature_importance = dt.feature_importances_
# 创建特征重要性 DataFrame
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
# 根据重要性排序
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
# 保存特征重要性到 CSV 文件
#print(feature_importance_df)
feature_importance_df.to_csv('DecisionTree_Feature_Importance.csv', index=False)