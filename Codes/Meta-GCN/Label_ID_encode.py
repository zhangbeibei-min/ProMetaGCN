#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.5.2
@author: beibei zhang
@license: Apache Licence 
@contact: 695193839@qq.com
@site: http://www.baidu.com
@software: PyCharm
@file: Label_ID_encode.py
@time: 2024/3/15 17:06
"""
# ###Label1已经code注释掉
import csv

# 读取Label1.csv的第一列内容
label1_values = []
with open(r'.\Meta-GCN\Data\Label1.csv', 'r') as label1_file:
    label1_reader = csv.reader(label1_file)
    for row in label1_reader:
        label1_values.append(row[0])

# 读取Cytokine_name_code_Idnumber.csv的第二列和第三列内容
cytokine_dict = {}
with open(r'.\Meta-GCN\Data\LabelClass\Cytokine_name_code_Idnumber.csv', 'r') as cytokine_file:
    cytokine_reader = csv.reader(cytokine_file)
    for row in cytokine_reader:
        cytokine_dict[row[2]] = row[0]

# 比较并输出结果
for value in label1_values:
    if value in cytokine_dict:
        print(f"找到匹配：{value} 对应的值为 {cytokine_dict[value]}")

# 存储匹配结果的列表
matched_results = []

# 读取Label1.csv的第一列内容
with open(r'.\Meta-GCN\Data\Label1.csv', 'r') as label1_file:
    label1_reader = csv.reader(label1_file)
    for row in label1_reader:
        value = row[0]
        if value in cytokine_dict:
            matched_results.append([value, cytokine_dict[value]])

# 将匹配结果写入label1encode.csv文件
with open(r'.\Meta-GCN\Data\label1encode.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Value', 'Matched Value'])
    for result in matched_results:
        csv_writer.writerow(result)

print("匹配结果已保存到label1encode.csv文件中。")



# ###保存除了正标签以外的（随机筛选数据集）
import csv

matched_values = set()

# 读取label1encode.csv中的匹配结果
with open(r'.\Meta-GCN\Data\label1encode.csv',
          'r') as label1encode_file:
    label1encode_reader = csv.reader(label1encode_file)
    next(label1encode_reader)  # 跳过标题行
    for row in label1encode_reader:
        matched_values.add(row[1])

# 找出不在匹配结果中的所有值
cytokine_data = []
with open(
        r'.\Meta-GCN\Data\LabelClass\Cytokine_name_code_Idnumber.csv',
        'r') as cytokine_file:
    for row in csv.reader(cytokine_file):
        cytokine_data.append(row)

unmatched_values = [row[0] for row in cytokine_data if row[0] not in matched_values]

# 剩余的内容保存到othersLabel.csv
with open(r'.\Meta-GCN\Data\othersLabel.csv', 'w',
          newline='') as others_label_file:
    others_label_writer = csv.writer(others_label_file)
    others_label_writer.writerow(['Value', 'Corresponding Value'])

    for row in cytokine_data:
        if row[0] in unmatched_values:
            others_label_writer.writerow([row[0], row[2]])

print("已保存到othersLabel.csv文件中。")
