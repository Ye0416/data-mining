import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


ATM_df = pd.read_table("train.tsv")

# # 将分类特征转换为数值特征
# object_cols = ATM_df.select_dtypes(include='object').columns
# ordinal_encoder = OrdinalEncoder()
# ATM_df[object_cols] = ordinal_encoder.fit_transform(ATM_df[object_cols])
#
# # 计算相关性矩阵
# corr_matrix = ATM_df.corr()
#
# # 绘制热力图
# fig, ax = plt.subplots(figsize=(12, 10))
# cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
#
# # 设置刻度和标签
# ax.set_xticks(np.arange(len(corr_matrix.columns)))
# ax.set_yticks(np.arange(len(corr_matrix.columns)))
# ax.set_xticklabels(corr_matrix.columns)
# ax.set_yticklabels(corr_matrix.columns)
#
# # 设置刻度标签旋转角度
# plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
#
# # 添加颜色条
# cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
#
# plt.show()


# -------------------- 2 -----------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
#
# # 提取数值特征和分类特征
# numerical_features = ATM_df.select_dtypes(include=['int64']).columns
# categorical_features = ATM_df.select_dtypes(include=['object']).columns
#
# # 散点图：数值特征与标签之间的关系
# for feature in numerical_features:
#     plt.figure()
#     plt.scatter(ATM_df[feature], ATM_df['revenue'], alpha=0.3)
#     plt.xlabel(feature)
#     plt.ylabel('revenue')
#     plt.title(f'Scatter plot: {feature} vs revenue')
#
# # 箱线图：分类特征与标签之间的关系
# for feature in categorical_features:
#     plt.figure()
#     sns.boxplot(x=ATM_df[feature], y=ATM_df['revenue'])
#     plt.xlabel(feature)
#     plt.ylabel('revenue')
#     plt.title(f'Box plot: {feature} vs revenue')
#
# # 直方图：数值特征的分布
# for feature in numerical_features:
#     plt.figure()
#     plt.hist(ATM_df[feature], bins=30, alpha=0.7)
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram: Distribution of {feature}')
#
# plt.show()


# ------------------- 3 -----------------
import plotly.express as px
# 提取数值特征和分类特征
numerical_features = ATM_df.select_dtypes(include=['int64']).columns
categorical_features = ATM_df.select_dtypes(include=['object']).columns

# 散点图矩阵：数值特征之间的关系
fig1 = px.scatter_matrix(ATM_df, dimensions=numerical_features, color='revenue')
fig1.show()

# 平行坐标图：数值特征与revenue之间的关系
fig2 = px.parallel_coordinates(ATM_df, color='revenue', dimensions=numerical_features, color_continuous_scale=px.colors.diverging.Tealrose)
fig2.show()

# 分类箱线图：分类特征与revenue之间的关系
for cat_feature in categorical_features:
    fig3 = px.box(ATM_df, x=cat_feature, y='revenue', title=f'{cat_feature} vs Revenue')
    fig3.show()

for cat_feature in numerical_features:
    fig3 = px.box(ATM_df, x=cat_feature, y='revenue', title=f'{cat_feature} vs Revenue')
    fig3.show()