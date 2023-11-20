from MyClasses import *
from models import 贪吃蛇训练模型
from GA import *
from datetime import datetime
import numpy as np
import torch

def 玩家_开始游戏(保存数据=False):
    '''
    用WASD控制移动，按键释放后才能进行下次移动
    :param 保存数据:
    :return:
    '''
    ui = UI()
    x = 蛇(窗口大小=10)

    running = True
    while running:
        ui(x)
        移动方向 = 键盘控制移动方向()
        running = x(移动方向)
    print("得分：", x.分数)
    if x.是否胜利:
        print('胜利！！')
    if 保存数据:
        当前时间 = datetime.now()
        np.savez(f'./{当前时间.strftime("%m_%d_%H_%M_%S")}.npz', data=x.datas, label=x.labels)  # 保存位置需要调整以下
        print(x.datas.shape, x.labels.shape)

def 训练(窗口大小=10,个体数量=100,保存轮次=100):
    ### 选择运行位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"程序运行位置：{torch.cuda.get_device_name(device)}.")
    # device = torch.device('cpu')
    # print(f"程序运行位置：{cpuinfo.get_cpu_info()['brand_raw']}")

    # ### 导入预训练模型
    # model_weights = torch.load("你的模型保存路径")

    ### 建立种群
    种群 = []  # 【蛇，模型】
    for _ in range(个体数量):
        x = 蛇(窗口大小=窗口大小)
        model = 贪吃蛇训练模型()
        model = model.to(device).eval()
        # model.load_state_dict(model_weights, strict=True)  # 导入预训练模型
        种群.append([x, model])

    ### 初始化实例
    ui = UI()
    导入不变信息(窗口大小=窗口大小, 个体数量=个体数量)

    ### 开始训练
    分数 = [None for _ in range(个体数量)]
    循环次数 = 1
    while True:
        for i in range(个体数量):
            if 分数[i] is not None:
                continue

            # 可以不显示
            if i == 0:
                ui(种群[i][0])
            if 分数[0] is not None:
                ui.黑屏()

            data = torch.Tensor(种群[i][0].data).to(device)
            移动方向 = 网络控制移动方向(model=种群[i][1], data=data)
            状态 = 种群[i][0](移动方向)

            if not 状态:
                分数[i] = 种群[i][0].吃到果实数量
        if None not in 分数:
            ui.黑屏()
            最佳分数 = max(分数)
            print(f"循环{循环次数}次  本次最佳分数为{最佳分数}")
            ### 保存
            if 循环次数 % 保存轮次 == 0:
                最佳索引 = 分数.index(最佳分数)
                torch.save(种群[最佳索引][1].state_dict(), "./model.pth")
            ### 遗传和变异
            种群 = 遗传(种群=种群)
            分数 = []
            for i in range(个体数量):
                分数.append(None)
                种群[i][0].初始化()
            循环次数 += 1

def 网络_开始游戏(模型路径):
    ### 选择运行位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"程序运行位置：{torch.cuda.get_device_name(device)}.")
    # device = torch.device('cpu')
    # print(f"程序运行位置：{cpuinfo.get_cpu_info()['brand_raw']}")

    ui = UI()
    x = 蛇(窗口大小=10)
    model = 贪吃蛇训练模型()

    model_weights = torch.load(模型路径)
    model.load_state_dict(model_weights, strict=True)
    model = model.to(device).eval()

    running = True
    while running:
        ui(x)
        data = torch.Tensor(x.data).to(device)
        移动方向 = 网络控制移动方向(model=model, data=data)
        running = x(移动方向)
    print(f"吃到果实数量：{x.吃到果实数量}\n走动步数：{x.步数}\n得分：{x.分数}")
    if x.是否胜利:
        print('胜利！！')

if __name__ == "__main__":
    玩家_开始游戏()
    # 训练()
    # 网络_开始游戏(模型路径="./model.pth")