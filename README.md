# Color Harmony Net

一个基于深度学习的图像颜色和谐度评估模型。

## 功能特点

- 使用 LDA 进行颜色主题建模
- 支持 RGB 和红外图像的特征提取
- 实时可视化训练过程
- 提供预训练模型

![不同数据集中图像组的示意图](./image/Images%20of%20different%20datasets.png)

**Table 1:** Performance Comparison on Different Datasets

| Methods | FLIR PLCC | FLIR SROCC | FLIR RMSE | KAIST PLCC | KAIST SROCC | KAIST RMSE | Our data PLCC | Our data SROCC | Our data RMSE |
| :------ | :-------: | :--------: | :-------: | :--------: | :---------: | :--------: | :-----------: | :------------: | :-----------: |
| NRSL    | 0.685     | 0.652      | 0.255     | 0.705      | 0.718       | 0.235      | 0.653         | 0.620          | 0.268         |
| HOSA    | 0.785     | 0.769      | 0.205     | 0.752      | 0.735       | 0.218      | 0.738         | 0.725          | 0.245         |
| NIMA    | 0.833     | 0.847      | 0.178     | 0.801      | 0.818       | 0.195      | 0.815         | 0.827          | 0.185         |
| NIQE    | 0.855     | 0.832      | 0.165     | 0.835      | 0.825       | 0.180      | 0.840         | 0.838          | 0.170         |
| LDANet  | **0.915** | **0.920**  | **0.105** | **0.895**  | **0.887**   | **0.115**  | **0.905**     | **0.910**      | **0.095**     |

## 安装

1. 克隆仓库:
```bash
git clone https://github.com/your-username/color-harmony-net.git
cd color-harmony-net
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python train.py
```

### 参数配置

在 `configs/config.py` 中设置:


## 可视化

训练过程中会生成:
- 损失曲线
- MSE 曲线
- 颜色分布图
- LDA 主题分布

## 预训练模型

提供以下预训练模型:
- `models/lda_model.pkl`

## 引用

如果您使用了本项目，请引用:

投稿中

## 联系方式

- 作者: Dian Sheng
- Email: shengdian970@163.com

