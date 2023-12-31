import numpy as np
import random


__all__ = [
    '导入不变信息',
    '遗传'
]


### 声明变量
__分数s: list
__模型权重s: list
__窗口大小: int
__最高分数: int
__转盘: list
__个体数量: int
__种群: list
父代精英数: int  # 父代中可以遗传后代的数量

def 导入不变信息(窗口大小, 个体数量):
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    __窗口大小 = 窗口大小
    __最高分数 = __窗口大小 * __窗口大小
    __个体数量 = 个体数量
    父代精英数 = __个体数量 // 5  # 可以修改

def 遗传(种群):
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    ### 初始化数据
    __种群 = 种群
    __获取基本信息()
    __筛选个体()

    ### 遗传与变异新种群
    __交叉遗传变异()
    __导入权重()

    return __种群

def __获取基本信息():
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    __分数s = []
    __模型权重s = []
    for 个体 in __种群:
        分数 = int((个体[0].吃到果实数量 + 1 / 个体[0].步数) * 100000) + 个体[0].额外扣分*1000
        __分数s.append(分数)
        __模型权重s.append(个体[1].导出权重())

def __筛选个体():
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    ### 按照分数大小排列
    分数与索引 = sorted(enumerate(__分数s), key=lambda x: x[1], reverse=True)  # 由大到小排列，格式为(原索引，数据)
    __分数s = [分数 for _, 分数 in 分数与索引]
    索引 = [i for i, _ in 分数与索引]
    __模型权重s = [__模型权重s[i] for i in 索引]
    __分数s = __分数s[:父代精英数]
    __模型权重s = __模型权重s[:父代精英数]

    ### 更新轮盘赌转盘
    __转盘 = []
    for i in range(len(__分数s)):
        if i == 0:
            __转盘.append(__分数s[i])
        else:
            __转盘.append(__分数s[i] + __转盘[-1])

def __轮盘赌() -> int:
    '''
    根据分数大小，返回选中的索引
    :return:
    '''
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    x = random.randint(__转盘[0], __转盘[-1])  # 筛选后，分数最低的那个基本没有概率被选择
    for i in range(len(__转盘)):
        xx = __转盘[i] - x
        if xx >= 0:
            return i

def __交叉遗传变异():
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    父代_模型权重s = __模型权重s
    子代_模型权重s = []
    for _ in range(__个体数量):
        ### 获取父代索引
        父代1_索引 = __轮盘赌()
        父代2_索引 = __轮盘赌()
        # while True:
        #     父代2_索引 = self.__轮盘赌()
        #     if 父代1_索引 != 父代2_索引:
        #         break
        ### 遗传(DNA长度为964)
        子代 = np.zeros((0, ))
        循环次数 = 0
        while True:
            if 循环次数 * 100 > 964:
                break
            if (循环次数 % 2) == 1:
                父代_索引 = 父代1_索引
            else:
                父代_索引 = 父代2_索引
            DNA = 父代_模型权重s[父代_索引][循环次数 * 100:(循环次数+1) * 100]
            分数 = __分数s[父代_索引]
            子代 = np.append(子代, __变异(DNA, 分数), axis=0)
            循环次数 += 1
        子代_模型权重s.append(子代)
    __模型权重s = 子代_模型权重s

def __变异率(分数) -> int:
    """
    变异的概率，百分之多少
    :return:
    """
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    变异率 = (__最高分数 - 分数) / __最高分数
    return int(变异率 * 100)
def __变异(DNA: np.ndarray, 分数: int) -> np.ndarray:
    '''
    高斯变异
    :param DNA:
    :param 分数:
    :return:
    '''
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    # ### 我自己写的变异算法，结果不好
    # 变异率 = self.__变异率(分数)
    # 子代 = np.zeros((0, ))
    # for 基因 in DNA:
    #     x = random.randint(1, 100)
    #     if x <= 变异率:
    #         基因 = np.random.normal(loc=0, scale=0.3)  # 生成一个满足正态分布的随机数
    #     子代 = np.append(子代, 基因)
    # return 子代

    ### 别人写的高斯遗传
    ## 变异率固定为0.2
    变异率 = 0.2
    ## 获取变异的索引
    mutation_array = np.random.random(DNA.shape) < 变异率
    ## 按照高斯分布（正态分布）随机一组数据，设置0.2倍标准差
    mutation = np.random.normal(size=DNA.shape)
    mutation[mutation_array] *= 0.2
    ## 变异的地方加到原本的DNA序列中
    DNA[mutation_array] += mutation[mutation_array]
    return DNA

def __导入权重():
    global __分数s, __模型权重s, __窗口大小, __最高分数, __转盘, __个体数量, __种群, 父代精英数
    for i in range(__个体数量):
        __种群[i][1].导入权重(__模型权重s[i])