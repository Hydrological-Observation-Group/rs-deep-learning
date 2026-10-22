# rsimg-segmentation-pytorch

We perform the fast use of the semantic segmentation models on the remote sensing image with pytorch.
<font color='darkorange'></font>

## Dataset

The surface water dataset built based on Sentinel-2 can be used for the quick testing in this repository, and the surface water dataset can be accessed at: https://zenodo.org/record/5205674.

## Models to be achieved

1. Unet (Simple)
2. DeeplabV3Plus [[Paper]](https://arxiv.org/abs/1802.02611)
3. DeeplabV3Plus with MobileNetV2 backbone
4. WatNet [[Paper]](https://www.sciencedirect.com/science/article/pii/S0303243421001793)
5. GMNet [[Paper]](https://www.tandfonline.com/doi/full/10.1080/17538947.2023.2198266) (Note: this model requires multiscale validation dataset)

## Features

1. Model training and validation Synchronously (scripts/trainer.py).
2. Generate validation patches from the whole dataset (`notebooks/3_valset_gen.ipynb`).
3. Plot and compare experiment results (`visual/methods_compare.ipynb`).

## 文件夹及文件介绍

```text
rs-deep-learning/
├── data/                  # 遥感影像、标签及验证集数据
├── dataloader/            # 旧版数据加载、预处理和增强模块
├── model/
│   ├── backbone/          # MobileNetV2、Xception65 等骨干网络
│   ├── seg/               # 通用语义分割模型
│   └── trained/           # 训练权重（运行训练后生成）
├── notebooks/             # 数据检查、归一化、验证集生成及训练示例
├── scripts/               # 训练配置与命令行训练入口
├── utils/                 # 数据读取、裁剪、增强、指标和可视化工具
├── visual/                # 模型结果对比与可视化 Notebook
├── LICENSE                # 项目许可证
└── README.md              # 项目说明
```

## Todo

1. using timm software for the classic model building, e.g., deeplabv3+, hrnet.
2. Built mamba-based baseline model.
3.
