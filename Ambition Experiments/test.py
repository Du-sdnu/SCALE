import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 数据定义 - 将batch_sizes改为字符集合
batch_sizes = ['$SCALE_{-proto}$', '$SCALE_{-vib}$', '$SCALE_{-gse}$', 'SCALE']
epochs = [10, 20, 30, 40, 50, 60, 70, 80]
show_epochs = [20, 50, 80]

# 1.画AUC-ROC在ieee-cis数据集
# accuracies = [
#     [0.73, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.88],
#     [0.70, 0.74, 0.77, 0.78, 0.80, 0.81, 0.83, 0.84],
#     [0.74, 0.79, 0.81, 0.83, 0.85, 0.86, 0.88, 0.89],
#     [0.76, 0.81, 0.84, 0.86, 0.87, 0.89, 0.91, 0.92]
# ]


# # 2. 画AUC-ROC在ethereum数据集
# accuracies = [
#     [0.78, 0.82, 0.85, 0.86, 0.88, 0.89, 0.91, 0.92],
#     [0.77, 0.81, 0.84, 0.85, 0.87, 0.88, 0.90, 0.91],
#     [0.72, 0.76, 0.79, 0.80, 0.82, 0.83, 0.85, 0.86],
#     [0.80, 0.85, 0.87, 0.89, 0.91, 0.92, 0.94, 0.95]
# ]
#
# #3.画准确率AUC-ROC在digraph-fin数据集
# accuracies = [
#     [0.70, 0.74, 0.76, 0.78, 0.79, 0.81, 0.82, 0.83],
#     [0.76, 0.80, 0.82, 0.84, 0.85, 0.87, 0.88, 0.89],
#     [0.75, 0.79, 0.81, 0.83, 0.84, 0.86, 0.87, 0.88],
#     [0.78, 0.82, 0.85, 0.86, 0.88, 0.89, 0.91, 0.92]
# ]
#
# #4.画Macro-F1在ieee-cis数据集
# accuracies = [
#     [0.62, 0.67, 0.69, 0.71, 0.73, 0.74, 0.76, 0.77],
#     [0.58, 0.63, 0.65, 0.67, 0.69, 0.70, 0.72, 0.73],
#     [0.63, 0.68, 0.70, 0.72, 0.74, 0.75, 0.77, 0.78],
#     [0.65, 0.70, 0.73, 0.75, 0.76, 0.78, 0.80, 0.81]
# ]
#
#
# # 5.画Macro-F1在ethereum数据集
# accuracies = [
#     [0.72, 0.77, 0.79, 0.81, 0.83, 0.84, 0.86, 0.87],
#     [0.73, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.88],
#     [0.68, 0.72, 0.75, 0.76, 0.78, 0.79, 0.81, 0.82],
#     [0.75, 0.80, 0.82, 0.84, 0.86, 0.87, 0.89, 0.90]
# ]
#
# # 6. 画Macro-F1在digraph-fin数据集
accuracies = [
    [0.64, 0.69, 0.72, 0.74, 0.76, 0.78, 0.79, 0.81],
    [0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87],
    [0.69, 0.74, 0.77, 0.79, 0.81, 0.83, 0.84, 0.86],
    [0.72, 0.77, 0.80, 0.82, 0.84, 0.86, 0.87, 0.89]
]

# 创建数值映射，用于在Y轴上定位字符标签的位置
batch_size_numeric = list(range(len(batch_sizes)))  # [0, 1, 2, 3, 4]
batch_size_mapping = {char: num for num, char in enumerate(batch_sizes)}

# 绘制3D图形
fig = plt.figure(figsize=(9, 7),dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 使用离散的颜色方案
#colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7']  # 明显不同的颜色
colors=['#02AFAC','#4D97CD','#ED8828','#8D73BA']
# 设置统一的线宽
line_width = 2

# 绘制每个batch size的折线
for i, batch_char in enumerate(batch_sizes):
    # 获取当前batch size的所有准确率
    z_vals = accuracies[i]
    numeric_y = batch_size_mapping[batch_char]  # 对应的数值位置

    # 选择颜色
    line_color = colors[i]

    # 绘制3D曲线 - 使用数值位置
    ax.plot(epochs, [numeric_y] * len(epochs), z_vals,
            label=f'{batch_char}',
            color=line_color,
            linewidth=line_width)

    # 画出折线在平面上的投影
    ax.plot(epochs, [numeric_y] * len(epochs), np.zeros(len(epochs)),
            color=line_color,
            linewidth=line_width+1, alpha=0.5)

    # 创建用于填充的多边形顶点
    verts = list(zip(epochs, [numeric_y] * len(epochs), z_vals))
    verts += list(zip(epochs[::-1], [numeric_y] * len(epochs), [0.6] * len(epochs)))

    # 使用 Poly3DCollection 填充曲线与 XY 平面之间的区域
    poly = Poly3DCollection([verts], color=line_color, alpha=0.3)
    ax.add_collection3d(poly)

# 添加颜色条 - 基于准确率
'''
Z = np.array(accuracies)
accuracy_norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
accuracy_cmap = cm.viridis
mappable = cm.ScalarMappable(norm=accuracy_norm, cmap=accuracy_cmap)
mappable.set_array(Z)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Accuracy')
'''
# 在特定的epoch上展示点和数值
for epoch in show_epochs:
    data_list_x = []
    data_list_y = []  # 存储数值位置
    data_list_z = []

    # 找到epoch在列表中的索引
    epoch_idx = epochs.index(epoch)

    for i, batch_char in enumerate(batch_sizes):
        accuracy_val = accuracies[i][epoch_idx]
        numeric_y = batch_size_mapping[batch_char]  # 对应的数值位置

        # 使用与对应折线相同的颜色
        point_color = colors[i]

        # 绘制重点的散点图，突出显示
        ax.scatter(epoch, numeric_y, accuracy_val,
                   color=point_color, s=50,
                   edgecolor='black', zorder=15)

        # 在散点旁边显示数值
        ax.text(epoch + 1, numeric_y, accuracy_val + 0.02,
                f'{accuracy_val:.2f}', color='black',
                fontsize=15, zorder=10)

        data_list_x.append(epoch)
        data_list_y.append(numeric_y)  # 存储数值位置
        data_list_z.append(accuracy_val)

    # 绘制每个特定epoch下的虚线连接不同batch size的准确率
    ax.plot(data_list_x, data_list_y, data_list_z,
            color='black', linewidth=line_width+1, linestyle='--', alpha=0.7)

# 设置坐标轴标签
ax.set_xlabel('Epochs', fontsize=16,labelpad=10)
#ax.set_ylabel('Methods', fontsize=15,labelpad=15)
ax.set_zlabel('AUC-ROC', fontsize=16, labelpad=10)
ax.set_zlim(0.6, 1.0)
z_ticks = np.arange(0.7, 1.01, 0.05)  # 从0.7到1.0，步长为0.05
ax.set_zticks(z_ticks)
ax.set_zticklabels([f'{tick:.2f}' for tick in z_ticks], fontsize=15)

# 设置Y轴刻度为字符标签
ax.set_yticks(batch_size_numeric)
ax.set_yticklabels(batch_sizes,fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='z', labelsize=16)
# 设置标题
#ax.set_title('Model Accuracy by Batch Size and Epoch', fontsize=14, pad=20)

# 添加图例
#ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)
#ax.legend(fontsize=12)

# 调整视角
ax.view_init(elev=20, azim=230)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.02, top=0.98, wspace=0, hspace=0)

#plt.tight_layout()
#plt.savefig('Precision-ablation-IMIOT.png', dpi=400, bbox_inches='tight')
plt.show()