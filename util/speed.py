import torch
import torchvision.models as models
import torch.nn as nn
import time

# model = models.resnet101(pretrained=True)

# # 替换模型的最后一层为新的全连接层，将输出大小设置为7
# num_features = model.fc.in_features
# custom_fc_layer = nn.Linear(num_features, 7)  # 将输出大小设置为7
# model.fc = custom_fc_layer

# # 打印模型的参数数量，仅计算新的全连接层的参数
# num_parameters = sum(p.numel() for p in model.parameters())// 1000000
# print(f"Total number of parameters (custom FC layer only): {num_parameters}")

# # 创建一个随机输入张量，模拟一批数据
# input_tensor = torch.randn(1, 3, 224, 224).cuda()  # batch size为1，通道数为3，图像大小为224x224
# model = model.cuda()
# # 计时模型的前向传播
# num_iterations = 100  # 运行100次前向传播来获得平均速度
# total_time = 0

# for _ in range(num_iterations):
#     start_time = time.time()
#     output = model(input_tensor)
#     end_time = time.time()
#     total_time += (end_time - start_time)

# average_time = total_time / num_iterations
# print(f"Average inference time for {num_iterations} iterations: {average_time} seconds")

# import torch
# import torch.nn as nn
# import time

# # 自定义12层卷积神经网络
# class CustomConvNet(nn.Module):
#     def __init__(self):
#         super(CustomConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(2)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU()
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.relu4 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(2)
        
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.relu5 = nn.ReLU()
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.relu6 = nn.ReLU()
#         self.maxpool3 = nn.MaxPool2d(2)
        
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.relu7 = nn.ReLU()
#         self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.relu8 = nn.ReLU()
#         self.maxpool4 = nn.MaxPool2d(2)
        
#         self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.relu9 = nn.ReLU()
#         self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.relu10 = nn.ReLU()
#         self.maxpool5 = nn.MaxPool2d(2)
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(512 * 7 * 7, 512)  # 7x7是输入图像的大小，请根据实际情况调整
#         self.relu11 = nn.ReLU()
#         self.fc2 = nn.Linear(512, 10)  # 10是输出类别数量，请根据实际情况调整

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool1(x)
        
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.maxpool2(x)
        
#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.conv6(x)
#         x = self.relu6(x)
#         x = self.maxpool3(x)
        
#         x = self.conv7(x)
#         x = self.relu7(x)
#         x = self.conv8(x)
#         x = self.relu8(x)
#         x = self.maxpool4(x)
        
#         x = self.conv9(x)
#         x = self.relu9(x)
#         x = self.conv10(x)
#         x = self.relu10(x)
#         x = self.maxpool5(x)
        
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu11(x)
#         x = self.fc2(x)
#         return x

# # 创建自定义12层卷积神经网络
# model = CustomConvNet()

# # 打印模型的参数数量（以百万为单位）
# num_parameters = sum(p.numel() for p in model.parameters()) / 1_000_000  # 将结果除以1,000,000
# print(f"Total number of parameters (CustomConvNet): {num_parameters:.2f} million")

# # 创建一个随机输入张量，模拟一批数据
# input_tensor = torch.randn(1, 3, 224, 224)  # batch size为1，通道数为3，图像大小为224x224

# # 计时模型的前向传播
# num_iterations = 100  # 运行100次前向传播来获得平均速度
# total_time = 0

# for _ in range(num_iterations):
#     start_time = time.time()
#     output = model(input_tensor)
#     end_time = time.time()
#     total_time += (end_time - start_time)

# average_time = total_time / num_iterations
# print(f"Average inference time for {num_iterations} iterations: {average_time:.4f} seconds")


# import torch
# import torch.nn as nn
# import dgl
# import dgl.nn as dglnn
# import time

# # 自定义VGG16
# class VGG16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d(7)
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

# # 自定义6层GCN（图卷积神经网络）
# class CustomGCN(nn.Module):
#     def __init__(self, in_feats, out_feats, num_layers=6):
#         super(CustomGCN, self).__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(dglnn.GraphConv(in_feats, out_feats, activation=nn.ReLU()))
#         for _ in range(1, num_layers):
#             self.layers.append(dglnn.GraphConv(out_feats, out_feats, activation=nn.ReLU()))

#     def forward(self, g, features):
#         for layer in self.layers:
#             features = layer(g, features)
#         return features

# # 自定义整合VGG16、GCN和全连接层的网络
# class CustomNetwork(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CustomNetwork, self).__init__()
#         self.vgg16 = VGG16(num_classes=num_classes)
#         self.gcn = CustomGCN(in_feats=num_classes, out_feats=num_classes)
#         self.fc = nn.Linear(num_classes, num_classes)

#     def forward(self, x, g):
#         x_vgg = self.vgg16(x)
#         x_gcn = self.gcn(g, x_vgg)
#         x_combined = x_vgg + x_gcn
#         x_output = self.fc(x_combined)
#         return x_output

# # 创建自定义网络
# num_classes = 10  # 示例中使用的类别数量为10
# model = CustomNetwork(num_classes=num_classes)

# # 打印模型的参数数量（以百万为单位）
# num_parameters = sum(p.numel() for p in model.parameters()) / 1_000_000  # 将结果除以1,000,000
# print(f"Total number of parameters (CustomNetwork): {num_parameters:.2f} million")

# # 创建一个随机输入张量，模拟一批数据
# input_tensor = torch.randn(1, 3, 224, 224)  # batch size为1，通道数为3，图像大小为224x224

# # 创建一个随机图形（用于GCN）
# num_nodes = 10  # 示例中使用的图中节点数为10
# g = dgl.DGLGraph()
# g.add_nodes(num_nodes)
# edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]  # 创建所有节点之间的边
# g.add_edges(*zip(*edges))
# g = dgl.add_self_loop(g)

# # 计时模型的前向传播
# num_iterations = 100  # 运行100次前向传播来获得平均速度
# total_time = 0

# for _ in range(num_iterations):
#     start_time = time.time()
#     output = model(input_tensor, g)
#     end_time = time.time()
#     total_time += (end_time - start_time)

# average_time = total_time / num_iterations
# print(f"Average inference time for {num_iterations} iterations: {average_time:.4f} seconds")

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

# 加载预训练的VGG16模型
model = models.vgg19(pretrained=False)
model.cuda()  # 将模型设置为评估模式
input_batch = torch.randn(1, 3, 224, 224).cuda()

# 运行前向传播并计算时间
num_iterations = 100  # 运行100次前向传播来获得平均速度
total_time = 0

with torch.no_grad():
    for _ in range(num_iterations):
        start_time = time.time()
        output = model(input_batch)
        end_time = time.time()
        total_time += (end_time - start_time)

average_time = total_time / num_iterations
print(f"Average inference time for {num_iterations} iterations: {average_time:.4f} seconds")
