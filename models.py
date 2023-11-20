import torch
import torch.nn as nn
import MY_model_training
from torchsummary import MYsummary
import numpy as np

class 贪吃蛇训练模型(nn.Module):
    def __init__(self):
        super(贪吃蛇训练模型, self).__init__()

        self.layer1 = nn.Linear(32, 20)
        self.layer2 = nn.Linear(20, 12)
        self.layer3 = nn.Linear(12, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.层名称 = [[name, self.state_dict()[name].cpu().numpy().shape]for name in self.state_dict()]

    def forward(self, x):
        x = x.view(-1, 32)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

    def 导出权重(self):
        data = np.zeros((0, ))
        权重 = self.state_dict()
        for name in 权重:
            D = 权重[name].cpu().numpy().copy()
            D = D.flatten()
            data = np.append(data, D, axis=0)
        return data
    def 导入权重(self, data):
        层名称 = self.层名称
        结点 = 0
        model_weights = self.state_dict()  # 获取原始网络模型的权重
        for name, shape in 层名称:
            try:
                i, j = shape
                x = i * j
                data_r = data[结点:结点 + x].reshape(i, j)
            except:
                x = shape[0]
                data_r = data[结点:结点 + x]
            data_r = torch.Tensor(data_r)
            结点 += x
            model_weights[name] = data_r
        self.load_state_dict(model_weights, strict=True)


if __name__ == "__main__":
    model = 贪吃蛇训练模型()
    data = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3]
    ], dtype=torch.float32)
    x = model(data)
    # print(x)
    # for name, param in model.named_parameters():
    #     print(name, param)
    data = model.导出权重()
    print(data)
    print(max(data), min(data))
    print(data.shape)
    # model.导入权重(data)