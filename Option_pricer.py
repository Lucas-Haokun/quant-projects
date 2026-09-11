import numpy as np
import matplotlib.pyplot as plt


class OptionPricer:

    def __init__(self, S0, K, r, sigma, T, N=252, M=100000, seed=42):
        """
        参数:
        S0 : float - 初始资产价格
        K  : float - 行权价
        r  : float - 无风险利率（年化）
        sigma : float - 波动率（年化）
        T  : float - 到期时间（年）
        N  : int   - 时间步数
        M  : int   - 模拟路径数量
        seed : int - 随机种子(默认42, 保证可复现)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.M = M
        self.seed = seed
        self.price_paths = None   # 用于存储生成的路径
        self.S_T = None           # 用于存储到期价格

    def simulate(self):
        """
        生成 M 条 GBM 路径，并存储到期价格
        """
        np.random.seed(self.seed)
        dt = self.T / self.N

        # 生成随机冲击矩阵 (M x N)
        Z = np.random.normal(0, 1, (self.M, self.N))

        # 计算每日对数收益率（风险中性）
        # 注意：这里用 r - 0.5*sigma^2，因为我们要做的是风险中性定价
        daily_returns = (self.r - 0.5 * self.sigma**2) * \
            dt + self.sigma * np.sqrt(dt) * Z

        # 计算累计对数收益率和价格路径
        log_returns = np.cumsum(daily_returns, axis=1)
        self.price_paths = self.S0 * np.exp(log_returns)

        # 提取到期价格（所有路径的最后一个时间点）
        self.S_T = self.price_paths[:, -1]

    def price_call(self):
        """
        计算欧式看涨期权价格
        """
        if self.S_T is None:
            self.simulate()

        payoffs = np.maximum(self.S_T - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price

    def price_put(self):
        """
        计算欧式看跌期权价格
        """
        if self.S_T is None:
            self.simulate()

        payoffs = np.maximum(self.K - self.S_T, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price

    def plot_distribution(self):
        """
        绘制到期价格分布直方图，并标出行权价
        """
        if self.S_T is None:
            self.simulate()

        plt.figure(figsize=(10, 5))
        plt.hist(self.S_T, bins=100, alpha=0.7,
                 color='skyblue', edgecolor='black')
        plt.axvline(self.K, color='red', linestyle='--',
                    linewidth=2, label=f'Strike K = {self.K}')
        plt.axvline(np.mean(self.S_T), color='green', linestyle='-',
                    linewidth=2, label=f'Mean = {np.mean(self.S_T):.2f}')
        plt.title(f'Distribution of Terminal Prices (M={self.M}, N={self.N})')
        plt.xlabel('Price at Maturity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_convergence(self):
        """
        绘制期权价格随模拟路径数量增加而收敛的过程
        """
        if self.S_T is None:
            self.simulate()

        # 计算累积平均值
        payoffs = np.maximum(self.S_T - self.K, 0)
        cumulative_mean = np.cumsum(payoffs) / np.arange(1, self.M + 1)
        cumulative_price = np.exp(-self.r * self.T) * cumulative_mean

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_price, color='blue', linewidth=1.5)
        plt.axhline(cumulative_price[-1], color='red', linestyle='--', linewidth=2,
                    label=f'Final Price = {cumulative_price[-1]:.4f}')
        plt.title('Convergence of Monte Carlo Option Price')
        plt.xlabel('Number of Simulations')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
