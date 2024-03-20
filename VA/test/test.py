

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化画布和子图
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
arousal_line, = ax[0].plot([], [], label='Arousal')
valence_line, = ax[1].plot([], [], label='Valence')

# 设置图形属性
ax[0].set_title('Arousal')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Value')
ax[0].legend()

ax[1].set_title('Valence')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Value')
ax[1].legend()

# 准备数据
label_path = '/data/wenzhuofan/Data/ABAW/Split_Feature_Label/split_label/Validation_Set/o_800s_1000'
files = sorted(os.listdir(label_path))[:300]

# 更新函数，用于更新曲线
def update(frame):
    # ax[0].clear()
    # ax[1].clear()

    # 读取标签数据
    with open(os.path.join(label_path, files[frame]), 'r') as f:
        label = f.readlines()[1:]

    label = [[float(j) for j in i.split(',')] for i in label]
    arousal = [i[1] for i in label]
    valence = [i[0] for i in label]

    # 绘制曲线
    arousal_line.set_data(range(len(arousal)), arousal)
    valence_line.set_data(range(len(valence)), valence)

    # 设置曲线范围
    ax[0].set_xlim(0, len(arousal))
    ax[0].set_ylim(min(arousal), max(arousal))
    ax[1].set_xlim(0, len(valence))
    ax[1].set_ylim(min(valence), max(valence))

    # 返回曲线对象
    return arousal_line, valence_line

# 创建动画
ani = FuncAnimation(fig, update, frames=len(files), blit=True,interval=1000)
ani.save('arousal_valence_animation.gif', writer='pillow')
# 显示动画
plt.show()






