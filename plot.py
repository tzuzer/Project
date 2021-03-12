import matplotlib.pyplot as plt
import numpy as np

def plot_function(ax, ay, az, am,an):

    # plt.rcParams['savefig.dpi'] = 200 #图片像素
    # plt.rcParams['figure.dpi'] = 200 #分辨率
    plt.rcParams['figure.figsize'] = (10, 10)        # 图像显示大小
    #  plt.rcParams['font.sans-serif']=['SimHei']   # 防止中文标签乱码，还有通过导入字体文件的方法
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 0.5   # 设置曲线线条宽度

    plt.clf()    # 清除刷新前的图表，防止数据量过大消耗内存
    #plt.suptitle("Accuracy and Loss",fontsize=30)             # 添加总标题，并设置文字大小

    # 图表1
    agraphic = plt.subplot(2,1,1)  # plt.subplot(2,1,1)
    agraphic.set_title('Accuracy and Loss with 50 epochs')      # 添加子标题
    plt.rcParams.update({'font.size': 15})            # 设置标题大小
    agraphic.set_xlabel('epoches', fontsize=15)   # 添加轴标签
    agraphic.set_ylabel('Accuracy', fontsize=15)
    plt.tick_params(labelsize=13)
    plt.axis([0,50,0,100])
    plt.plot(ax,ay,'g-',label='training acc', linewidth=2)
    plt.plot(ax,am,'b-',label='testing acc', linewidth=2)

    plt.legend()
    # 等于agraghic.plot(ax,ay,'g-')

    # # 图表2
    agraphic = plt.subplot(2, 1, 2)  # plt.subplot(2,1,1)
    agraphic.set_xlabel('epoches', fontsize=15)  # 添加轴标签
    agraphic.set_ylabel('loss', fontsize=15)
    plt.tick_params(labelsize=13)
    plt.axis([0, 50, 0, 50])
    plt.plot(ax, az, 'g-', label='training loss', linewidth=2)
    plt.plot(ax, an, 'b-', label='testing loss', linewidth=2)

    plt.legend()

    plt.pause(0.6)     # 设置暂停时间，太快图表无法正常显示
    #plt.savefig('picture.png', dpi=300)  # 设置保存图片的分辨率

