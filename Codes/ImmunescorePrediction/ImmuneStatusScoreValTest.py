#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.5.2
@author: beibei zhang
@license: Apache Licence 
@contact: 695193839@qq.com
@site: http://www.baidu.com
@software: PyCharm
@file: ImmuneStatusScoreTest.py
@time: 2024/2/29 20:06
"""
# # ##*******************************************1.XGBoost********************************
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
# 读入训练集数据
data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\HealthScore.csv')
# # 提取特征列（183）
selected_features=["ACE2", "ADAMTS13", "AGER", "AIF1", "AREG", "ARG1", "C2", "CCL11", "CCL13", "CCL15",
                   "CCL16", "CCL17", "CCL18", "CCL19", "CCL2", "CCL20", "CCL21", "CCL22", "CCL23", "CCL24",
                   "CCL25", "CCL26", "CCL27", "CCL28", "CCL3", "CCL5", "CCL7", "CCL8", "CD163", "CD200", "CD207",
                   "CD209", "CD27", "CD274", "CD33", "CD38", "CD4", "CD40LG", "CD48", "CD55", "CD59", "CD70",
                   "CD83", "CEBPB", "CHI3L1", "CLEC7A", "CSF1", "CSF3", "CST3", "CX3CL1", "CXCL1", "CXCL10",
                   "CXCL11", "CXCL12", "CXCL13", "CXCL16", "CXCL5", "CXCL6", "CXCL8", "CXCL9", "EGF", "ENG",
                   "EPO", "F3", "F7", "FAS", "FASLG", "FCGR2A", "FCGR2B", "FCGR3B", "FGF2", "FGF5", "GDF15",
                   "GH1", "GZMA", "GZMB", "HAVCR2", "HGF", "ICAM1", "IFNG", "IFNGR1", "IFNGR2", "IFNL1", "IGFBP3",
                   "IL10", "IL10RA", "IL10RB", "IL11", "IL12RB1", "IL13", "IL13RA1", "IL15RA", "IL16", "IL17A",
                   "IL17D", "IL17F", "IL17RA", "IL17RB", "IL18BP", "IL18R1", "IL19", "IL1A", "IL1B", "IL1R1",
                   "IL1R2", "IL1RAP", "IL1RL1", "IL1RL2", "IL1RN", "IL2", "IL20RA", "IL22RA1", "IL24", "IL2RA",
                   "IL34", "IL3RA", "IL4", "IL4R", "IL5", "IL5RA", "IL6", "IL6R", "IL6ST", "IL7", "IL7R", "KIT",
                   "LAG3", "LBP", "LCN2", "LEP", "LEPR", "LGALS3", "LGALS9", "LIFR", "LTA", "MASP1", "MMP1",
                   "MMP12", "MMP13", "MMP3", "MMP7", "MMP8", "MMP9", "MPO", "MSR1", "NCR1", "NGF", "OSM",
                   "PDCD1LG2", "PDGFB", "PDGFRA", "PDGFRB", "PECAM1", "PLAT", "PLAU", "PLAUR", "PRL", "PRTN3",
                   "RARRES2", "REN", "RETN", "S100A12", "SCARB2", "SELE", "SELP", "SERPINE1", "SFTPD", "SIGLEC1",
                   "SIRPA", "SPP1", "TGFB1", "THPO", "TIMP1", "TNF", "TNFRSF14", "TNFRSF1A", "TNFRSF1B", "TNFRSF4",
                   "TNFRSF8", "TNFRSF9", "TNFSF11", "TNFSF13B", "TSLP", "VCAM1", "XCL1"]
X = data[selected_features]
# 提取目标变量
y = data['Score']
# 提取Age列
age = data['Age']
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(X_scaled, y, age, test_size=0.2, random_state=42)
# 创建XGBoost模型
xgb_model = XGBRegressor()
# 拟合模型
xgb_model.fit(X_train, y_train)
# 在测试集上评估模型
y_pred = xgb_model.predict(X_test)
results = pd.DataFrame({ 'Age': age_test,'True_Values': y_test, 'Predictions': y_pred})
# 根据 Age 列的数值进行排序
results = results.sort_values('Age')
# 保存排序后的结果到另一个 CSV 文件
results.to_csv('XGBOOST_Test_results.csv', index=False)
mae = mean_absolute_error(y_test, y_pred)
print('xgboost result:')
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
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(y_test, y_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 绘制预测结果与真实值的图像
plt.plot(y_test, y_test, color='red', label='True Values')
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.text(0.2, 0.9, f'Pearson corr: {pearson_corr:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("XGBoost - True vs Predicted Values")
plt.legend(loc='lower right')
plt.savefig('XGBoost_test_prediction.svg', format='svg')
plt.savefig('XGBoost_test_prediction.pdf', format='pdf')
plt.show()  # 显示图像

#************************************独立验证集**********************************
# 读入测试集数据
val_data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day0.csv')
val_data1 = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day14.csv')
# 提取测试集特征列
X_val = val_data[selected_features]
X_val1 = val_data1[selected_features]
# 提取验证集标签列
y_val = val_data['Score']
y_val1 = val_data1['Score']
age_val = val_data['Age']
age_val1 = val_data1['Age']
# 在测试集上进行预测
y_val_pred = xgb_model.predict(X_val)
y_val_pred1 = xgb_model.predict(X_val1)
results1 = pd.DataFrame({'Age': age_val, 'True_Values': y_val, 'Predictions': y_val_pred})
# 根据Age列的数值进行排序
results1 = results1.sort_values('Age')
results1.to_csv('Day0_XGBOOST_val_results.csv', index=False)
results2 = pd.DataFrame({'Age': age_val1, 'True_Values': y_val1, 'Predictions': y_val_pred1})
# 根据Age列的数值进行排序
results2 = results2.sort_values('Age')
results2.to_csv('Day14_XGBOOST_val_results.csv', index=False)

###day0
# 计算Spearman相关性系数
print("XGBoost day0:")
spearman_corr, spearman_pval = spearmanr(age_val, y_val_pred)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val, y_val_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val, y_val_pred, 1)
polynomial = np.poly1d(coefficients)

# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val, polynomial(age_val), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val, y_val_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("XGBoost - Age vs Predicted/True Values(Day0)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day0_XGBoost_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()
# 打印拟合直线的系数
#print(f'xgboost拟合day0直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')

# day14
# 计算Spearman相关性系数
print("XGBoost day14:")
spearman_corr, spearman_pval = spearmanr(age_val1, y_val_pred1)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val1, y_val_pred1)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val1, y_val_pred1, 1)
polynomial = np.poly1d(coefficients)
# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val1, polynomial(age_val1), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val1, y_val_pred1, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("XGBoost - Age vs Predicted/True Values(Day14)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day14_XGBoost_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()
# 打印拟合直线的系数
#print(f'xgboost拟合day14直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')
# #
# #
# #
# # # ##********************************2.RandomForest*********************************
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split

# 读入数据
data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\HealthScore.csv')
# ###GCN结果(交集140个特征,0.95)
selected_features=["ACE2", "ADAMTS13", "AGER", "AIF1", "AREG", "ARG1", "C2", "CCL11", "CCL13", "CCL15",
                   "CCL16", "CCL17", "CCL18", "CCL19", "CCL2", "CCL20", "CCL21", "CCL22", "CCL23", "CCL24",
                   "CCL25", "CCL26", "CCL27", "CCL28", "CCL3", "CCL5", "CCL7", "CCL8", "CD163", "CD200", "CD207",
                   "CD209", "CD27", "CD274", "CD33", "CD38", "CD4", "CD40LG", "CD48", "CD55", "CD59", "CD70",
                   "CD83", "CEBPB", "CHI3L1", "CLEC7A", "CSF1", "CSF3", "CST3", "CX3CL1", "CXCL1", "CXCL10",
                   "CXCL11", "CXCL12", "CXCL13", "CXCL16", "CXCL5", "CXCL6", "CXCL8", "CXCL9", "EGF", "ENG",
                   "EPO", "F3", "F7", "FAS", "FASLG", "FCGR2A", "FCGR2B", "FCGR3B", "FGF2", "FGF5", "GDF15",
                   "GH1", "GZMA", "GZMB", "HAVCR2", "HGF", "ICAM1", "IFNG", "IFNGR1", "IFNGR2", "IFNL1", "IGFBP3",
                   "IL10", "IL10RA", "IL10RB", "IL11", "IL12RB1", "IL13", "IL13RA1", "IL15RA", "IL16", "IL17A",
                   "IL17D", "IL17F", "IL17RA", "IL17RB", "IL18BP", "IL18R1", "IL19", "IL1A", "IL1B", "IL1R1",
                   "IL1R2", "IL1RAP", "IL1RL1", "IL1RL2", "IL1RN", "IL2", "IL20RA", "IL22RA1", "IL24", "IL2RA",
                   "IL34", "IL3RA", "IL4", "IL4R", "IL5", "IL5RA", "IL6", "IL6R", "IL6ST", "IL7", "IL7R", "KIT",
                   "LAG3", "LBP", "LCN2", "LEP", "LEPR", "LGALS3", "LGALS9", "LIFR", "LTA", "MASP1", "MMP1",
                   "MMP12", "MMP13", "MMP3", "MMP7", "MMP8", "MMP9", "MPO", "MSR1", "NCR1", "NGF", "OSM",
                   "PDCD1LG2", "PDGFB", "PDGFRA", "PDGFRB", "PECAM1", "PLAT", "PLAU", "PLAUR", "PRL", "PRTN3",
                   "RARRES2", "REN", "RETN", "S100A12", "SCARB2", "SELE", "SELP", "SERPINE1", "SFTPD", "SIGLEC1",
                   "SIRPA", "SPP1", "TGFB1", "THPO", "TIMP1", "TNF", "TNFRSF14", "TNFRSF1A", "TNFRSF1B", "TNFRSF4",
                   "TNFRSF8", "TNFRSF9", "TNFSF11", "TNFSF13B", "TSLP", "VCAM1", "XCL1"]
X = data[selected_features]
# 提取目标变量
y = data['Score']
# 提取Age列
age = data['Age']
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(X_scaled, y, age, test_size=0.2, random_state=42)
# 创建随机森林模型
rf_model = RandomForestRegressor()
# 拟合模型
rf_model.fit(X_train, y_train)
# 在测试集上评估模型
y_pred = rf_model.predict(X_test)
results = pd.DataFrame({'True_Values': y_test, 'Predictions': y_pred, 'Age': age_test})
# 根据 Age 列的数值进行排序
results = results.sort_values('Age')
# 保存排序后的结果到另一个 CSV 文件
results.to_csv('RandomForest_Test_results.csv', index=False)
mae = mean_absolute_error(y_test, y_pred)
print('RandomForest result:')
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
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(y_test, y_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 绘制预测结果与真实值的图像
plt.plot(y_test, y_test, color='red', label='True Values')
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.text(0.2, 0.9, f'Pearson corr: {pearson_corr:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("RandomForest - True vs Predicted Values")
plt.legend(loc='lower right')
plt.savefig('RandomForest_test_prediction.svg', format='svg')
plt.savefig('RandomForest_test_prediction.pdf', format='pdf')
plt.show()  # 显示图像


#************************************独立验证集**********************************
# 读入测试集数据
val_data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day0.csv')
val_data1 = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day14.csv')
# 提取测试集特征列
X_val = val_data[selected_features]
X_val1 = val_data1[selected_features]
# 提取验证集标签列
y_val = val_data['Score']
y_val1 = val_data1['Score']
age_val = val_data['Age']
age_val1 = val_data1['Age']
# 在测试集上进行预测
y_val_pred = rf_model.predict(X_val)
y_val_pred1 = rf_model.predict(X_val1)
results1 = pd.DataFrame({ 'Age': age_val,'True_Values': y_val, 'Predictions': y_val_pred})
# 根据Age列的数值进行排序
results1 = results1.sort_values('Age')
results1.to_csv('Day0_RandomForest_val_results.csv', index=False)
results2 = pd.DataFrame({ 'Age': age_val1,'True_Values': y_val1, 'Predictions': y_val_pred1})
# 根据Age列的数值进行排序
results2 = results2.sort_values('Age')
results2.to_csv('Day14_RandomForest_val_results.csv', index=False)

###day0
# 计算Spearman相关性系数
print("RF day0:")
spearman_corr, spearman_pval = spearmanr(age_val, y_val_pred)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val, y_val_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val, y_val_pred, 1)
polynomial = np.poly1d(coefficients)

# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val, polynomial(age_val), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val, y_val_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("RandomForest - Age vs Predicted/True Values(Day0)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day0_RandomForest_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()

# 打印拟合直线的系数
print(f'RandomForest拟合day0直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')

# day14
# 计算Spearman相关性系数
print("RF day14:")
spearman_corr, spearman_pval = spearmanr(age_val1, y_val_pred1)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val1, y_val_pred1)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val1, y_val_pred1, 1)
polynomial = np.poly1d(coefficients)
# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val1, polynomial(age_val1), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val1, y_val_pred1, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("RandomForest - Age vs Predicted/True Values(Day14)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day14_RandomForest_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()
# 打印拟合直线的系数
#print(f'xgboost拟合day14直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')



# ###******************************************3.lasso*********************************
#
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
# # 提取特征列（covid-19与健康人的交集特征）
###GCN结果
selected_features=["ACE2", "ADAMTS13", "AGER", "AIF1", "AREG", "ARG1", "C2", "CCL11", "CCL13", "CCL15",
                   "CCL16", "CCL17", "CCL18", "CCL19", "CCL2", "CCL20", "CCL21", "CCL22", "CCL23", "CCL24",
                   "CCL25", "CCL26", "CCL27", "CCL28", "CCL3", "CCL5", "CCL7", "CCL8", "CD163", "CD200", "CD207",
                   "CD209", "CD27", "CD274", "CD33", "CD38", "CD4", "CD40LG", "CD48", "CD55", "CD59", "CD70",
                   "CD83", "CEBPB", "CHI3L1", "CLEC7A", "CSF1", "CSF3", "CST3", "CX3CL1", "CXCL1", "CXCL10",
                   "CXCL11", "CXCL12", "CXCL13", "CXCL16", "CXCL5", "CXCL6", "CXCL8", "CXCL9", "EGF", "ENG",
                   "EPO", "F3", "F7", "FAS", "FASLG", "FCGR2A", "FCGR2B", "FCGR3B", "FGF2", "FGF5", "GDF15",
                   "GH1", "GZMA", "GZMB", "HAVCR2", "HGF", "ICAM1", "IFNG", "IFNGR1", "IFNGR2", "IFNL1", "IGFBP3",
                   "IL10", "IL10RA", "IL10RB", "IL11", "IL12RB1", "IL13", "IL13RA1", "IL15RA", "IL16", "IL17A",
                   "IL17D", "IL17F", "IL17RA", "IL17RB", "IL18BP", "IL18R1", "IL19", "IL1A", "IL1B", "IL1R1",
                   "IL1R2", "IL1RAP", "IL1RL1", "IL1RL2", "IL1RN", "IL2", "IL20RA", "IL22RA1", "IL24", "IL2RA",
                   "IL34", "IL3RA", "IL4", "IL4R", "IL5", "IL5RA", "IL6", "IL6R", "IL6ST", "IL7", "IL7R", "KIT",
                   "LAG3", "LBP", "LCN2", "LEP", "LEPR", "LGALS3", "LGALS9", "LIFR", "LTA", "MASP1", "MMP1",
                   "MMP12", "MMP13", "MMP3", "MMP7", "MMP8", "MMP9", "MPO", "MSR1", "NCR1", "NGF", "OSM",
                   "PDCD1LG2", "PDGFB", "PDGFRA", "PDGFRB", "PECAM1", "PLAT", "PLAU", "PLAUR", "PRL", "PRTN3",
                   "RARRES2", "REN", "RETN", "S100A12", "SCARB2", "SELE", "SELP", "SERPINE1", "SFTPD", "SIGLEC1",
                   "SIRPA", "SPP1", "TGFB1", "THPO", "TIMP1", "TNF", "TNFRSF14", "TNFRSF1A", "TNFRSF1B", "TNFRSF4",
                   "TNFRSF8", "TNFRSF9", "TNFSF11", "TNFSF13B", "TSLP", "VCAM1", "XCL1"]
X = data[selected_features]
# 提取目标变量
y = data['Score']
# 提取Age列
age = data['Age']
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(X_scaled, y, age, test_size=0.2, random_state=42)

# 创建LassoCV模型，cv参数为交叉验证折数
lasso_cv = LassoCV(cv=5)
# 拟合模型
lasso_cv.fit(X_train, y_train)
#print('lasso result:')
print("最优的 alpha 值：", lasso_cv.alpha_)

# 在测试集上评估模型
y_pred = lasso_cv.predict(X_test)
#y_pred = np.round(y_pred).astype(int)##整数
results = pd.DataFrame({'True_Values': y_test, 'Predictions': y_pred, 'Age': age_test})
# 根据 Age 列的数值进行排序
results = results.sort_values('Age')
# 保存排序后的结果到另一个 CSV 文件
results.to_csv('Lasso_Test_results.csv', index=False)
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
plt.savefig('Lasso_test_prediction.svg', format='svg')
plt.savefig('Lasso_test_prediction.pdf', format='pdf')
plt.show()  # 显示图像

#************************************独立验证集**********************************
# 读入测试集数据
val_data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day0.csv')
val_data1 = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day14.csv')
# 提取测试集特征列
X_val = val_data[selected_features]
X_val1 = val_data1[selected_features]
# 提取验证集标签列
y_val = val_data['Score']
y_val1 = val_data1['Score']
age_val = val_data['Age']
age_val1 = val_data1['Age']
# 在测试集上进行预测
y_val_pred = lasso_cv.predict(X_val)
y_val_pred1 = lasso_cv.predict(X_val1)
results1 = pd.DataFrame({ 'Age': age_val,'True_Values': y_val, 'Predictions': y_val_pred})
# 根据Age列的数值进行排序
results1 = results1.sort_values('Age')
results1.to_csv('Day0_Lasso_val_results.csv', index=False)
results2 = pd.DataFrame({ 'Age': age_val1,'True_Values': y_val1, 'Predictions': y_val_pred1})
# 根据Age列的数值进行排序
results2 = results2.sort_values('Age')
results2.to_csv('Day14_Lasso_val_results.csv', index=False)

###day0
# 计算Spearman相关性系数
print("Lasso day0:")
spearman_corr, spearman_pval = spearmanr(age_val, y_val_pred)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val, y_val_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val, y_val_pred, 1)
polynomial = np.poly1d(coefficients)

# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val, polynomial(age_val), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val, y_val_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("Lasso - Age vs Predicted/True Values(Day0)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day0_Lasso_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()

# 打印拟合直线的系数
print(f'Lasso拟合day0直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')

# day14
# 计算Spearman相关性系数
print("Lasso day14:")
spearman_corr, spearman_pval = spearmanr(age_val1, y_val_pred1)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val1, y_val_pred1)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val1, y_val_pred1, 1)
polynomial = np.poly1d(coefficients)
# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val1, polynomial(age_val1), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val1, y_val_pred1, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("Lasso - Age vs Predicted/True Values(Day14)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day14_Lasso_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()
# 打印拟合直线的系数
#print(f'xgboost拟合day14直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')

#**********************************4.LightGBM****************************************
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

# 读入数据
data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\HealthScore.csv')
ids = data['ID']
## # 提取特征列
selected_features=["ACE2", "ADAMTS13", "AGER", "AIF1", "AREG", "ARG1", "C2", "CCL11", "CCL13", "CCL15",
                   "CCL16", "CCL17", "CCL18", "CCL19", "CCL2", "CCL20", "CCL21", "CCL22", "CCL23", "CCL24",
                   "CCL25", "CCL26", "CCL27", "CCL28", "CCL3", "CCL5", "CCL7", "CCL8", "CD163", "CD200", "CD207",
                   "CD209", "CD27", "CD274", "CD33", "CD38", "CD4", "CD40LG", "CD48", "CD55", "CD59", "CD70",
                   "CD83", "CEBPB", "CHI3L1", "CLEC7A", "CSF1", "CSF3", "CST3", "CX3CL1", "CXCL1", "CXCL10",
                   "CXCL11", "CXCL12", "CXCL13", "CXCL16", "CXCL5", "CXCL6", "CXCL8", "CXCL9", "EGF", "ENG",
                   "EPO", "F3", "F7", "FAS", "FASLG", "FCGR2A", "FCGR2B", "FCGR3B", "FGF2", "FGF5", "GDF15",
                   "GH1", "GZMA", "GZMB", "HAVCR2", "HGF", "ICAM1", "IFNG", "IFNGR1", "IFNGR2", "IFNL1", "IGFBP3",
                   "IL10", "IL10RA", "IL10RB", "IL11", "IL12RB1", "IL13", "IL13RA1", "IL15RA", "IL16", "IL17A",
                   "IL17D", "IL17F", "IL17RA", "IL17RB", "IL18BP", "IL18R1", "IL19", "IL1A", "IL1B", "IL1R1",
                   "IL1R2", "IL1RAP", "IL1RL1", "IL1RL2", "IL1RN", "IL2", "IL20RA", "IL22RA1", "IL24", "IL2RA",
                   "IL34", "IL3RA", "IL4", "IL4R", "IL5", "IL5RA", "IL6", "IL6R", "IL6ST", "IL7", "IL7R", "KIT",
                   "LAG3", "LBP", "LCN2", "LEP", "LEPR", "LGALS3", "LGALS9", "LIFR", "LTA", "MASP1", "MMP1",
                   "MMP12", "MMP13", "MMP3", "MMP7", "MMP8", "MMP9", "MPO", "MSR1", "NCR1", "NGF", "OSM",
                   "PDCD1LG2", "PDGFB", "PDGFRA", "PDGFRB", "PECAM1", "PLAT", "PLAU", "PLAUR", "PRL", "PRTN3",
                   "RARRES2", "REN", "RETN", "S100A12", "SCARB2", "SELE", "SELP", "SERPINE1", "SFTPD", "SIGLEC1",
                   "SIRPA", "SPP1", "TGFB1", "THPO", "TIMP1", "TNF", "TNFRSF14", "TNFRSF1A", "TNFRSF1B", "TNFRSF4",
                   "TNFRSF8", "TNFRSF9", "TNFSF11", "TNFSF13B", "TSLP", "VCAM1", "XCL1"]
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
# 创建LightGBM模型
lgb_model = lgb.LGBMRegressor( boosting_type='gbdt',  # 提升类型，可以是 'gbdt', 'dart', 'goss', 'rf'
    num_leaves=31,  # 叶子节点数
    max_depth=2,  # 最大树深度，-1 表示无限制
    learning_rate=0.1,  # 学习率
    n_estimators=100,  # 迭代次数
    objective='regression',  # 目标函数，这里是回归任务
    metric='l2'  # 评估指标，这里是均方误差

)
# 拟合模型
lgb_model.fit(X_train, y_train)
# 在测试集上评估模型
y_pred = lgb_model.predict(X_test)
#y_pred = np.round(y_pred).astype(int)
mae = mean_absolute_error(y_test, y_pred)
print('lightgbm result:')
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
# 1.绘制预测结果与真实值的图像
plt.plot(y_test, y_test, color='red', label='True Values')
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("LightGBM - True vs Predicted Values")
plt.legend(loc='lower right')
plt.text(0.2, 0.9, f'Pearson corr: {pearson_corr:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
# # 保存图片为SVG格式保存图片为PDF格式
plt.savefig('LightGBM_true_prediction.svg', format='svg')
plt.savefig('LightGBM_true_prediction.pdf', format='pdf')
plt.show()  # 显示图像


#************************************独立验证集**********************************
# 读入测试集数据
val_data = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day0.csv')
val_data1 = pd.read_csv(r'E:\zhangbeibei\Desktop\recent work\Plasma proteome\Data\COVID19_Day14.csv')
# 提取测试集特征列
X_val = val_data[selected_features]
X_val1 = val_data1[selected_features]
# 提取验证集标签列
y_val = val_data['Score']
y_val1 = val_data1['Score']
age_val = val_data['Age']
age_val1 = val_data1['Age']
# 在测试集上进行预测
y_val_pred = lgb_model.predict(X_val)
y_val_pred1 = lgb_model.predict(X_val1)
results1 = pd.DataFrame({ 'Age': age_val,'True_Values': y_val, 'Predictions': y_val_pred})
# 根据Age列的数值进行排序
results1 = results1.sort_values('Age')
results1.to_csv('Day0_LightGBM_val_results.csv', index=False)
results2 = pd.DataFrame({ 'Age': age_val1,'True_Values': y_val1, 'Predictions': y_val_pred1})
# 根据Age列的数值进行排序
results2 = results2.sort_values('Age')
results2.to_csv('Day14_LightGBM_val_results.csv', index=False)

###**********************************day0***********************************
# 计算Spearman相关性系数
print("LightGBM day0:")
spearman_corr, spearman_pval = spearmanr(age_val, y_val_pred)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val, y_val_pred)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val, y_val_pred, 1)
polynomial = np.poly1d(coefficients)

# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val, polynomial(age_val), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val, y_val_pred, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("LightGBM - Age vs Predicted/True Values(Day0)")
#plt.savefig('XGBoost_val_age_score.svg', format='svg')
plt.savefig('Day0_LightGBM_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()
# 打印拟合直线的系数
print(f'LightGBM拟合day0直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')


#*******************************day14*************************
# 计算Spearman相关性系数
print("LightGBM day14:")
spearman_corr, spearman_pval = spearmanr(age_val1, y_val_pred1)
print("Spearman相关性系数:", spearman_corr)
print("Spearman p-value:", spearman_pval)
# 计算Pearson相关系数
pearson_corr, pearson_pval= pearsonr(age_val1, y_val_pred1)
print("Pearson相关系数:", pearson_corr)
print("Pearson p-value:", pearson_pval)
# 使用numpy的polyfit函数拟合直线
coefficients = np.polyfit(age_val1, y_val_pred1, 1)
polynomial = np.poly1d(coefficients)
# 绘制预测值与年龄的图像
#plt.figure(figsize=(10, 6))
# 绘制拟合的直线
plt.plot(age_val1, polynomial(age_val1), color='red', linewidth=2, label='Fitted Line')
# 在图上标注拟合直线的系数
#plt.text(40, 80, f'y = {round(coefficients[0], 2)} * x + {round(coefficients[1], 2)}', fontsize=12, color='k')
plt.text(0.6, 0.8, f'Pearson corr: {pearson_corr:.2f} ,p-value: {pearson_pval:.2e}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
plt.scatter(age_val1, y_val_pred1, color='steelblue', alpha=0.7, s=100, label='Predictions')
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("LightGBM - Age vs Predicted/True Values(Day14)")
#plt.savefig('LightGBM_val_age_score.svg', format='svg')
plt.savefig('Day14_LightGBM_val_age_score.pdf', format='pdf')
plt.legend()
plt.show()
# 打印拟合直线的系数
#print(f'LightGBM拟合day14直线的系数为：斜率={coefficients[0]}, 截距={coefficients[1]}')