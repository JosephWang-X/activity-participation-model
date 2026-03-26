# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

start = datetime.datetime.now()
#读取数据 
df = pd.read_csv("aup_n.csv",sep= ',',header= None)
df.columns = [
'area',
'date',
'time',
'p_flow',
'age_1',
'age_2',
'age_3',
'first_visit',
'man',
'woman',
'dur_1',
'dur_2',
'dur_3',
'dur_4',
'label'
]
df = df.replace(r'\N',np.nan)
df[['area','date']] =  df[['area','date']].astype(str)

df[['time',
'p_flow',
'age_1',
'age_2',
'age_3',
'first_visit',
'man',
'woman',
'dur_1',
'dur_2',
'dur_3',
'dur_4',
'label']] =  df[['time',
'p_flow',
'age_1',
'age_2',
'age_3',
'first_visit',
'man',
'woman',
'dur_1',
'dur_2',
'dur_3',
'dur_4',
'label']].astype(float)



#将数据分为活动时间和非活动时间
df_1 = df.loc[df.label==1]
df_0 = df.loc[df.label==0]

# 获取唯一的area值
area_ids = df['area'].unique()
# 创建一个字典，用于存储分割后的数据框
area_dataframes = {}


# 根据area_id分割数据框   
for area_id in area_ids:
    area_dataframes[area_id] = df_0[df_0['area'] == area_id]
    
    
# 创建一个空的 DataFrame 用于存放合并后的结果
df_merged = pd.DataFrame()

# 遍历每个子区域的 df_0 数据框，计算前七天的平均值，并与活动当天数据合并
for area, df_0_subarea in area_dataframes.items():
    
    df_0_avg_subarea = df_0_subarea.groupby('time').mean().reset_index()
    # 获取子区域活动当天的数据
    df_1_subarea = df[(df['area'] == area) & (df['label'] == 1)]

    
    # 合并活动当天和平均值数据
    df_merged_subarea = pd.merge(df_1_subarea, df_0_avg_subarea, on='time', suffixes=('_act', '_avg'))
    
    # 将合并后的数据添加到 df_merged 中
    df_merged = pd.concat([df_merged, df_merged_subarea], ignore_index=True)


# 计算增长率
metrics = ['p_flow']
for metric in metrics:
    df_merged[f'{metric}_增长率'] = (df_merged[f'{metric}_act'] - df_merged[f'{metric}_avg']) / df_merged[f'{metric}_avg']

# 对需要归一化的列进行选择
columns_to_scale = [
'p_flow_act',
'age_1_act',
'age_2_act',
'age_3_act',
'first_visit_act',
'man_act',
'woman_act',
'dur_1_act',
'dur_2_act',
'dur_3_act',
'dur_4_act',
]

# 使用 MinMaxScaler 进行归一化
scaler = MinMaxScaler()
df_merged[columns_to_scale] = scaler.fit_transform(df_merged[columns_to_scale])

# 定义权重
weights = {
    'p_flow': {'growth_rate': 0.05 , 'actual': 0.37},
    'dur_1': { 'actual': 0.02},
    'dur_2': { 'actual': 0.02},
    'dur_3': {'growth_rate': 0.01, 'actual': 0.05},
    'dur_4': {'growth_rate': 0.01, 'actual': 0.05},
    'first_visit': {'actual': 0.27},
    '比例均衡度': {'actual': 0.05}
}

# 计算性别和年龄段均衡得分
def balance_score(male, female, age_0_25, age_26_44, age_45_60):
    # 计算性别比例均衡得分
    if male + female == 0:
        gender_balance = 1  # 如果没有男性和女性人数，视为完全均衡
    else:
        gender_balance = 1 - abs(male - female) / (male + female)
    
    # 计算年龄段均衡得分
    total_age = age_0_25 + age_26_44 + age_45_60
    if total_age == 0:
        age_balance = 1  # 如果没有年龄段人数，视为完全均衡
    else:
        age_balance = 1 - abs(age_0_25 - (age_26_44 + age_45_60) / 2) / total_age

    return (gender_balance + age_balance) / 2

# 计算参与度得分的函数
def calculate_participation_score(row):
    score = 0
    
    # 计算基于增长率的得分
    for metric, weight in weights.items():
        if 'growth_rate' in weight and f'{metric}_增长率' in row:
            score += row[f'{metric}_增长率'] * weight['growth_rate']
        if 'actual' in weight and f'{metric}_act' in row:  # 对于实际值
            score += row[f'{metric}_act'] * weight['actual']
    
    # 计算性别和年龄的均衡得分
    balance = balance_score(
        row['man_act'],
        row['woman_act'],
        row['age_1_act'],
        row['age_2_act'],
        row['age_3_act']
    )
    score += balance * weights['比例均衡度']['actual']
    
    return score

df_merged['score'] = df_merged.apply(calculate_participation_score, axis=1)

df_result = df_merged[['area','date','time','score']]
#保留三位有效数字
df_result.set_index(['area'],inplace=True)
df_result['score'] = round(df_result['score'], 3)  
# 输出结果
# df_result.to_csv("C:/Users/19124/Desktop/OUTPUT/AUP_results.txt",header = True,encoding = 'gbk')
print("活动当天各时段的参与度得分:")
print(df_merged[['area','date','time', 'score']])

end = datetime.datetime.now()
print (end-start)
