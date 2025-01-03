import torch


class FIDCalculator:
    def __init__(self, config):
        self.config = config
        self.eps = 1e-6  # 添加小扰动项，确保数值稳定性

    def calculate_fid_loss(self, features_real, features_generated):
        # 展平特征
        features_real = self.flatten_features(features_real)
        features_generated = self.flatten_features(features_generated)

        # 计算均值
        mu_real = torch.mean(features_real, dim=0)
        mu_gen = torch.mean(features_generated, dim=0)

        # 计算协方差矩阵
        cov_real = self.covariance_matrix(features_real)
        cov_gen = self.covariance_matrix(features_generated)

        # 添加小的扰动项，确保协方差矩阵是正定的
        cov_real += torch.eye(cov_real.size(0)).to(cov_real.device) * self.eps
        cov_gen += torch.eye(cov_gen.size(0)).to(cov_gen.device) * self.eps

        # 计算协方差矩阵的平方根
        cov_sqrt = self.matrix_sqrt(cov_real.mm(cov_gen))

        # 计算 FID 损失
        diff = mu_real - mu_gen
        diff_squared = torch.sum(diff ** 2)
        fid_loss = diff_squared + torch.trace(cov_real + cov_gen - 2 * cov_sqrt)

        return fid_loss

    def flatten_features(self, features):
        """
        将 [batch_size, channels, height, width] 展平为 [batch_size, feature_dim]
        """
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def covariance_matrix(self, features):
        """
        计算协方差矩阵 [feature_dim, feature_dim]
        """
        # 计算均值并中心化
        mean = torch.mean(features, dim=0, keepdim=True)
        features_centered = features - mean

        # 计算协方差矩阵
        cov = features_centered.t().mm(features_centered) / (features.size(0) - 1)
        return cov

    def matrix_sqrt(self, matrix):
        """
        使用 SVD 分解计算矩阵的平方根
        """
        # 使用 SVD 分解
        U, S, V = torch.svd(matrix)
        # 计算平方根
        sqrt_S = torch.sqrt(S)
        # 重构平方根矩阵
        return U.mm(torch.diag(sqrt_S)).mm(V.t())
