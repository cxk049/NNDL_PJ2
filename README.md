# NNDL_PJ2
A project for Neural Network and Deep Learning

## 项目主要结构

```
codes/
├── train_cifar10.py               # CIFAR-10训练主脚本
├── VGG_BatchNorm/                 # VGG与BatchNorm对比实验
│   ├── VGG_Loss_Landscape.py      # 损失景观可视化
│   ├── data/                      # 数据加载相关代码
│   │   ├── loaders.py             # 数据加载器
│   │   └── __init__.py
│   ├── models/                    # 模型定义
│   │   ├── vgg.py                 # VGG模型及其变种
│   │   └── __init__.py
│   └── utils/                     # 工具函数
│       ├── nn.py                  # 神经网络辅助函数
│       └── __init__.py
└── results/                       # 结果保存目录（运行后生成）
    ├── models/                    # 保存的模型权重
    └── figures/                   # 生成的图表和可视化
