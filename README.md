
# Color Harmony Net

A deep learning-based model for evaluating the color harmony of images.

## Features

-   Uses LDA for color theme modeling
-   Supports feature extraction for both RGB and infrared images
-   Real-time visualization of the training process
-   Provides pre-trained models

![Illustrations of image groups in different datasets](./image/Images%20of%20different%20datasets.png)

**Table 1:** Performance Comparison on Different Datasets

| Methods | FLIR PLCC | FLIR SROCC | FLIR RMSE | KAIST PLCC | KAIST SROCC | KAIST RMSE | Our data PLCC | Our data SROCC | Our data RMSE |
| :------ | :-------: | :--------: | :-------: | :--------: | :---------: | :--------: | :-----------: | :------------: | :-----------: |
| NRSL    | 0.685     | 0.652      | 0.255     | 0.705      | 0.718       | 0.235      | 0.653         | 0.620          | 0.268         |
| HOSA    | 0.785     | 0.769      | 0.205     | 0.752      | 0.735       | 0.218      | 0.738         | 0.725          | 0.245         |
| NIMA    | 0.833     | 0.847      | 0.178     | 0.801      | 0.818       | 0.195      | 0.815         | 0.827          | 0.185         |
| NIQE    | 0.855     | 0.832      | 0.165     | 0.835      | 0.825       | 0.180      | 0.840         | 0.838          | 0.170         |
| LDANet  | **0.915** | **0.920**  | **0.105** | **0.895**  | **0.887**   | **0.115**  | **0.905**     | **0.910**      | **0.095**     |

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/color-harmony-net.git
    cd color-harmony-net
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Train the Model

```bash
python train.py
```

### Configuration

Set parameters in `configs/config.py`:

## Visualization

During training, the following will be generated:

-   Loss curve
-   MSE curve
-   Color distribution map
-   LDA topic distribution

## Pre-trained Models

The following pre-trained models are provided:

-   `models/lda_model.pkl`

## Citation

If you use this project, please cite:

Paper under review.

## Contact

-   Author: Dian Sheng
-   Email: shengdian970@163.com
```
