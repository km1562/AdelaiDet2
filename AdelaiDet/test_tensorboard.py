import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter


class NumNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.ReLU(),
        )

    def forward(self, batched_inputs):
        x = batched_inputs
        logits = self.classifier(x)
        return logits


model = NumNet()

# 把信息保存到logdir
writer = SummaryWriter('logdir')
# 测试传入16*5的feature_size时，模型的整个结构，这里测试会执行3次forward，为什么会执行3次我也不懂
writer.add_graph(model, torch.rand([16, 5]))
writer.close()