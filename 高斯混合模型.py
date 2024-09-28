import numpy as np
from sklearn.mixture import GaussianMixture

# 生成高斯混合模型数据
np.random.seed(0)
n_samples = 300

# 真实模型参数
true_means = np.array([[3, 1], [8, 10], [12, 2]])
true_covariances = np.array([[[1, -0.5], [-0.5, 1]], [[2, 0.8], [0.8, 2]], [[1, 0], [0, 1]]])
true_weights = np.array([1/3, 1/3, 1/3])
# 生成数据
X = np.concatenate([
    np.random.multivariate_normal(true_means[i], true_covariances[i], int(n_samples * true_weights[i]))
    for i in range(3)
])

# 拟合高斯混合模型并计算AIC和BIC
n_components_range = range(1, 4)  # 不同的高斯混合成分个数
best_aic = np.inf
best_bic = np.inf
best_gmm = None

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    print("Components: {}, AIC: {:.2f}, BIC: {:.2f}".format(n_components, aic, bic))

    if aic < best_aic:
        best_aic = aic
        best_gmm = gmm

    if bic < best_bic:
        best_bic = bic

# 模型参数
print("\nLearned Parameters:")
for i in range(best_gmm.n_components):
    print("component {} :均值 = {}, 协方差 = {}".format(i+1, best_gmm.means_[i], best_gmm.covariances_[i]))