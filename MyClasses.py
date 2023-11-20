import random
import numpy as np
import pygame
from pygame.locals import *
import keyboard
import time
import torch
from datetime import datetime

__all__ = [
    '蛇',
    'UI',
    '键盘控制移动方向',
    '网络控制移动方向',
]

方向s = ["w", 's', 'a', 'd']

class 蛇:
    def __init__(self, 窗口大小=10):
        ### 声明变量
        self.窗口大小: int
        self.果实位置: list
        self.蛇位置: list
        self.分数: int
        self.吃到果实数量: int
        self.步数: int
        self.额外扣分: int
        self.是否胜利: bool
        self.data: np.ndarray
        self.场地: np.ndarray
        self.datas: np.ndarray
        self.labels: np.ndarray

        ### 初始化数据
        # 用于保存全部信息
        self.datas = np.zeros((0, 32))
        self.labels = np.zeros((0,))
        # 其他数据
        self.窗口大小 = 窗口大小
        self.初始化()

    def __call__(self, 移动方向):
        return self.贪吃蛇的一步(移动方向)

    def __生成果实(self):
        场地大小 = len(self.场地)
        x, y = random.randint(0, 场地大小 - 1), random.randint(0, 场地大小 - 1)
        while True:
            if self.场地[x, y] == 0:
                break
            x, y = random.randint(0, 场地大小 - 1), random.randint(0, 场地大小 - 1)
        self.果实位置 = [x, y]

    def __蛇方向(self, 蛇片段) -> np.ndarray:
        方向 = 蛇片段[2]
        if 方向 == "w":
            return np.array([1, 0, 0, 0])
        elif 方向 == "a":
            return np.array([0, 0, 0, 1])
        elif 方向 == "s":
            return np.array([0, 0, 1, 0])
        elif 方向 == "d":
            return np.array([0, 1, 0, 0])
    def __获取方位(self, 原点, 其他) -> list:
        x = 其他[0] - 原点[0]
        y = 其他[1] - 原点[1]
        ### 首先判断果实在蛇左边还是右边
        if x == 0:
            if y > 0:
                return [1, 0, 0, 0, 0, 0, 0, 0]
            elif y < 0:
                return [0, 0, 0, 0, 1, 0, 0, 0]
            else:
                raise Exception("果实与蛇首重叠")
        elif x > 0:
            p = y / x
            if p == 1:
                return [0, 1, 0, 0, 0, 0, 0, 0]
            elif p == -1:
                return [0, 0, 0, 1, 0, 0, 0, 0]
            elif p == 0:
                return [0, 0, 1, 0, 0, 0, 0, 0]
        elif x < 0:
            p = y / x
            if p == 1:
                return [0, 0, 0, 0, 0, 1, 0, 0]
            elif p == -1:
                return [0, 0, 0, 0, 0, 0, 0, 1]
            elif p == 0:
                return [0, 0, 0, 0, 0, 0, 1, 0]
        return [0, 0, 0, 0, 0, 0, 0, 0]
    def __果实相对方向(self) -> np.ndarray:
        return np.array(self.__获取方位(self.蛇位置[0], self.果实位置))
    def __身子相对方向(self) -> np.ndarray:
        data = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        for i in range(1, len(self.蛇位置)):
            方位 = self.__获取方位(self.蛇位置[0], self.蛇位置[i])
            try:
                x = 方位.index(1)
                data[x] = 1
            except:
                pass
        return data
    def __与墙距离(self) -> np.ndarray:
        data = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        x, y, _ = self.蛇位置[0]
        ### 计算垂直水平方向
        try:
            data[0] = 1 / (self.窗口大小 - y - 1)
        except:
            data[0] = 0
        try:
            data[2] = 1 / (self.窗口大小 - x - 1)
        except:
            data[2] = 0
        try:
            data[4] = 1 / y
        except:
            data[4] = 0
        try:
            data[6] = 1 / x
        except:
            data[6] = 0
        ### 计算斜向方向距离
        for i in range(4):
            i = (i * 2) + 1
            p = 0
            xx, yy = x, y
            while True:
                if i == 1:
                    xx += 1
                    yy += 1
                    if xx == self.窗口大小 or yy == self.窗口大小:
                        break
                elif i == 3:
                    xx += 1
                    yy -= 1
                    if xx == self.窗口大小 or yy == -1:
                        break
                elif i == 5:
                    xx -= 1
                    yy -= 1
                    if xx == -1 or yy == -1:
                        break
                elif i == 7:
                    xx -= 1
                    yy += 1
                    if xx == -1 or yy == self.窗口大小:
                        break
                else:
                    raise Exception("陷入死循环")
                if p > 10000:
                    print(f"i:{i}   p:{p}   xx,yy:{xx, yy}   x,y:{x, y}   data:{data}")
                    raise Exception("陷入死循环")
                p += 1
            try:
                data[i] = 1 / p
            except:
                data[i] = 0
        return data

    def 贪吃蛇的一步(self, 方向):
        新各部位位置 = []
        if 方向 is None:
            return True
        self.分数 -= 1
        self.步数 += 1
        if 方向 == "w":
            新各部位位置.append((self.蛇位置[0][0], self.蛇位置[0][1] + 1, 'w'))
        elif 方向 == "s":
            新各部位位置.append((self.蛇位置[0][0], self.蛇位置[0][1] - 1, 's'))
        elif 方向 == "a":
            新各部位位置.append((self.蛇位置[0][0] - 1, self.蛇位置[0][1], 'a'))
        elif 方向 == "d":
            新各部位位置.append((self.蛇位置[0][0] + 1, self.蛇位置[0][1], 'd'))
        ## 更新各部位位置
        for i in range(len(self.蛇位置) - 1):
            新各部位位置.append(self.蛇位置[i])
        ## 判断死亡
        新各部位位置_仅坐标 = [(p[0], p[1]) for p in 新各部位位置]
        # 撞墙
        if max(新各部位位置_仅坐标[0]) >= self.窗口大小 or min(新各部位位置_仅坐标[0]) < 0:
            self.额外扣分 -= 50
            return False
        # 撞自己
        if 新各部位位置_仅坐标.count(新各部位位置_仅坐标[0]) > 1:
            self.额外扣分 -= 10
            return False
        # 打转
        if self.分数 <= -20:
            self.额外扣分 -= 100
            return False
        ### 判断胜利
        if 0 not in self.场地:
            self.是否胜利 = True
            self.额外扣分 += 1000
            return False
        ## 判断是否吃到果实
        是否需要生成新果实 = False
        if self.场地[新各部位位置[0][0], 新各部位位置[0][1]] == -1:
            新各部位位置.append(self.蛇位置[-1])
            # 记录是否需要生成新果实
            是否需要生成新果实 = True
        ## 重建场地
        # 清空场地
        self.场地 = np.zeros((self.窗口大小, self.窗口大小), dtype=int)
        # 绘制贪吃蛇
        for i, 位置 in enumerate(新各部位位置):
            x, y, _ = 位置
            self.场地[x, y] = len(新各部位位置) - i
        # 生成新果实
        if 是否需要生成新果实:
            self.__生成果实()
            self.分数 += 20
            self.吃到果实数量 += 1
        x, y = self.果实位置
        self.场地[x, y] = -1
        ## 更新各部位位置
        self.蛇位置 = 新各部位位置
        ### 更新data
        蛇首方向 = self.__蛇方向(self.蛇位置[0])
        蛇尾方向 = self.__蛇方向(self.蛇位置[-1])
        ## 蛇首八个方向(从垂直向上起，顺时针45°)上
        食物方向 = self.__果实相对方向()
        身子方向 = self.__身子相对方向()
        与墙距离 = self.__与墙距离()
        data = np.concatenate((蛇首方向, 蛇尾方向, 食物方向, 身子方向, 与墙距离), axis=0)
        self.datas = np.concatenate((self.datas, data[None, :]), axis=0)
        self.labels = np.append(self.labels, 方向s.index(方向))
        self.data = data
        return True

    def 初始化(self):
        self.果实位置 = []
        self.蛇位置 = []
        self.分数 = 10  # 初始为10，每吃一个果实加20分，每行动一次减1
        self.吃到果实数量 = 0
        self.步数 = 0
        self.额外扣分 = 0
        self.是否胜利 = False
        self.场地 = np.zeros((self.窗口大小, self.窗口大小), dtype=int)

        self.__初始化贪吃蛇()
    def __初始化贪吃蛇(self):
        ### 生成贪吃蛇
        x, y = random.randint(0, self.窗口大小 - 1), random.randint(0, self.窗口大小 - 1)
        self.蛇位置.append((x, y, 'w'))
        self.场地[x, y] = len(self.蛇位置)

        ### 生成果实
        self.__生成果实()
        x, y = self.果实位置
        self.场地[x, y] = -1

        ### 初始化data
        蛇首方向 = np.array([1, 0, 0, 0])  # 初始方向默认向上
        蛇尾方向 = np.array([1, 0, 0, 0])  # 只有一个身子时，默认蛇尾方向与蛇首方向相同
        ## 蛇首八个方向(从垂直向上起，顺时针45°)上
        食物方向 = self.__果实相对方向()
        身子方向 = self.__身子相对方向()
        与墙距离 = self.__与墙距离()
        self.data = np.concatenate((蛇首方向, 蛇尾方向, 食物方向, 身子方向, 与墙距离), axis=0)

class UI:
    def __init__(self):
        ### 声明变量
        self.__窗口大小: int
        self.__格大小: float
        self.刷新率: float = 20
        self.背景颜色: tuple = (255, 255, 255)
        self.__window_surface: pygame.surface.Surface

        ### 初始化
        pygame.init()
        pygame.display.set_caption('贪吃蛇')
        self.__window_surface = pygame.display.set_mode((1000, 1000))  # 设置背景大小

    def __call__(self, gameobject):
        ### 初始化
        pygame.display.update()  # 更新显示
        pygame.time.Clock().tick(self.刷新率)  # 设置刷新率
        self.__window_surface.fill(self.背景颜色)  # 设置背景
        self.__窗口大小 = gameobject.窗口大小
        self.__格大小 = 1000 / self.__窗口大小
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        ### 绘制
        ## 果实
        x, y = self.__坐标转换(gameobject.果实位置)
        pygame.draw.rect(self.__window_surface, (255, 0, 0), (x, y, self.__格大小, self.__格大小))
        ## 绘制蛇
        for i in range(len(gameobject.蛇位置)):
            x, y = self.__坐标转换(gameobject.蛇位置[i])
            ## 头
            if i == 0:
                pygame.draw.rect(self.__window_surface, (0, 0, 0), (x, y, self.__格大小, self.__格大小))
            else:
                pygame.draw.rect(self.__window_surface, (255, 255, 0), (x, y, self.__格大小, self.__格大小))
        ## 绘制成绩
        font = pygame.font.Font('myfont.ttf', 30)
        scoreSurf = font.render(f'得分: {gameobject.分数}', True, (0, 0, 0))
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (800, 10)
        self.__window_surface.blit(scoreSurf, scoreRect)

    def __坐标转换(self, 坐标):
        x = 坐标[0] * self.__格大小
        y = (self.__窗口大小 - 坐标[1] - 1) * self.__格大小
        return x, y

    def 黑屏(self):
        pygame.display.update()  # 更新显示
        self.__window_surface.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

def 键盘控制移动方向():
    '''
    只有抬起后才能继续键入第二次运动的方位
    :return:
    '''
    方向 = None
    if keyboard.is_pressed('w'):
        while True:
            if not keyboard.is_pressed('w'):
                方向 = "w"
                break
            time.sleep(0.01)
    elif keyboard.is_pressed('a'):
        while True:
            if not keyboard.is_pressed('a'):
                方向 = "a"
                break
            time.sleep(0.01)
    elif keyboard.is_pressed('s'):
        while True:
            if not keyboard.is_pressed('s'):
                方向 = "s"
                break
            time.sleep(0.01)
    elif keyboard.is_pressed('d'):
        while True:
            if not keyboard.is_pressed('d'):
                方向 = "d"
                break
            time.sleep(0.01)
    return 方向

def 网络控制移动方向(model, data):
    with torch.no_grad():  # 在评估模式下，不需要计算梯度
        output = model(data)
    # 获取预测结果
    predicted_label = torch.argmax(output, dim=1)
    x = predicted_label.item()
    return 方向s[x]