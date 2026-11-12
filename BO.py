"""
贝叶斯优化（Bayesian Optimization）接入分子模拟环境。

环境接口与 SAC.py 完全一致：
    env = Env(polar_space=False)
    state = env.reset()
    next_state, reward, done, info = env.step(action)

其中 action = [x, y, bias_voltage, current]；
若 polar_space=True，则 action = [l, theta, bias_voltage, current]。

与 SAC 的区别：
    SAC 学习的是"策略"（状态 -> 动作），这里贝叶斯优化直接搜索
    "最佳参数"（x, y, v, a）。由于环境单步具有随机性，每个参数点会
    重复执行 n_repeats 次，用平均奖励（或成功率）作为该点的目标值，
    再用高斯过程 + Expected Improvement 决定下一个采样点。

主要参数：
    n_init / n_iter   初始拉丁超立方点数 / 贝叶斯优化迭代次数
    n_repeats         每个参数点重复执行环境的次数（降低随机噪声）
    objective         "reward"（平均奖励）或 "success"（成功率）
    xi                EI 的探索系数（相对于目标值标准差的倍数）
    refit_every       GP 超参数重拟合间隔（其余迭代只做廉价的 Cholesky 更新）
    max_train_points  GP 训练集上限（数据更多时保留最优/最近/均匀抽样点，控制 O(n^3) 开销）
    patience          早停耐心值（连续多少次没有刷新最优就停止；None 为不早停）
    progress          是否显示进度条（优先使用 tqdm，缺失时自动降级）
    record_trials     是否记录每次 reset/step 的原始反应结果
    seed              随机种子，便于复现

输出（bo_result_plot/）：
    bo_history_<时间>.txt          每次评估的参数、奖励、成功率
    bo_trials_<时间>.txt           每次 reset/step 的原始反应（success/failure/none）
    bo_best_<时间>.json            最优参数与对应指标
    bo_result_<时间>_<超参数>.png  参数分布与优化曲线

用法：
    python BO.py
"""

import json
import os
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from env import Env  # 导入自定义环境

try:
    from tqdm import tqdm
except ImportError:  # tqdm 为可选依赖，缺失时使用内置的简易进度条
    tqdm = None


def _format_time(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class _SimpleProgressBar:
    """tqdm 不可用时使用的极简进度条（接口与 tqdm 保持一致）。"""

    def __init__(self, total, desc="", width=30):
        self.total = max(total, 1)
        self.desc = desc
        self.width = width
        self.n = 0
        self.start = time.time()
        self.postfix = {}
        self._render()

    def set_postfix(self, **kwargs):
        self.postfix = kwargs
        self._render()

    def update(self, n=1):
        self.n += n
        self._render()

    def write(self, message):
        print()
        print(message)
        self._render()

    def _render(self):
        elapsed = time.time() - self.start
        ratio = min(self.n / self.total, 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        eta = elapsed / max(self.n, 1) * (self.total - self.n)
        line = (f"\r{self.desc} |{bar}| {self.n}/{self.total} "
                f"({ratio * 100:5.1f}%) {_format_time(elapsed)}<{_format_time(eta)}")
        if self.postfix:
            line += "  " + " ".join(f"{key}={value}" for key, value in self.postfix.items())
        print(line, end="", flush=True)

    def close(self):
        print()


def _make_progress_bar(total, desc):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)
    return _SimpleProgressBar(total, desc)


def latin_hypercube(n_samples, n_dims, rng):
    """
    拉丁超立方采样（Latin Hypercube Sampling）。

    返回 shape=(n_samples, n_dims) 的矩阵，每一维都在 [0, 1] 上分层，
    保证初始点比纯随机采样覆盖得更均匀。
    """
    intervals = np.arange(n_samples, dtype=np.float64) / n_samples
    points = np.zeros((n_samples, n_dims), dtype=np.float64)
    for dim in range(n_dims):
        jitter = rng.random(n_samples) / n_samples
        points[:, dim] = rng.permutation(intervals + jitter)
    return points


class GaussianProcess:
    """
    用于贝叶斯优化的高斯过程回归（Matern 5/2 核 + ARD）。

    - 输入为归一化到 [0, 1] 的参数点
    - 输出在内部做标准化，便于不同目标函数共用同一套超参数初值
    - 通过最大化对数边际似然自动调节每个维度的长度尺度和噪声
    """

    def __init__(self, length_scale=0.3, noise=1e-3, signal_var=1.0,
                 optimize=True, n_restarts=2, length_scale_bounds=(0.02, 0.5),
                 random_state=None):
        self.length_scale = np.atleast_1d(np.asarray(length_scale, dtype=np.float64))
        self.noise = float(noise)
        self.signal_var = float(signal_var)
        self.optimize = optimize
        self.n_restarts = n_restarts
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.rng = np.random.RandomState(random_state)

        self.X_train = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.L = None
        self.alpha = None

    def _kernel(self, X1, X2, length_scale):
        """Matern 5/2 核，length_scale 为每个维度一个值（ARD）。"""
        X1 = np.atleast_2d(np.asarray(X1, dtype=np.float64)) / length_scale
        X2 = np.atleast_2d(np.asarray(X2, dtype=np.float64)) / length_scale
        sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dist = np.maximum(sq1 + sq2 - 2.0 * X1 @ X2.T, 0.0)
        r = np.sqrt(sq_dist)
        sqrt5_r = np.sqrt(5.0) * r
        return self.signal_var * (1.0 + sqrt5_r + 5.0 / 3.0 * r ** 2) * np.exp(-sqrt5_r)

    def _kernel_and_gradients(self, X, length_scale):
        """返回训练核矩阵 K 以及 dK/dlog(length_scale_j)，用于解析梯度。"""
        X_scaled = X / length_scale
        sq = np.sum(X_scaled ** 2, axis=1)
        sq_dist = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X_scaled @ X_scaled.T, 0.0)
        r = np.sqrt(sq_dist)
        sqrt5_r = np.sqrt(5.0) * r
        exp_term = np.exp(-sqrt5_r)
        K = self.signal_var * (1.0 + sqrt5_r + 5.0 / 3.0 * r ** 2) * exp_term

        gradients = np.empty((X.shape[1], len(X), len(X)))
        for dim in range(X.shape[1]):
            diff = X[:, dim][:, None] - X[:, dim][None, :]
            gradients[dim] = ((5.0 / 3.0) * self.signal_var
                              * (1.0 + sqrt5_r) * exp_term
                              * (diff / length_scale[dim]) ** 2)
        return K, gradients

    def _negative_log_likelihood(self, theta, X, y):
        """
        theta = [log(length_scale_1..d), log(noise)]。

        同时返回解析梯度：dNLL/dtheta = 0.5 * tr((K^-1 - alpha alpha^T) dK/dtheta)。
        """
        length_scale = np.exp(theta[:-1])
        noise = np.exp(theta[-1])
        K, gradients = self._kernel_and_gradients(X, length_scale)
        K = K + (noise ** 2 + 1e-8) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(X) * np.log(2.0 * np.pi)

        K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        A = K_inv - np.outer(alpha, alpha)
        gradient = np.empty_like(theta)
        for dim in range(X.shape[1]):
            gradient[dim] = 0.5 * np.sum(A * gradients[dim])
        gradient[-1] = 0.5 * np.trace(A) * 2.0 * noise ** 2
        return nll, gradient

    def _set_training_data(self, X, y, length_scale, noise):
        """用给定的核超参数重算 L 和 alpha（不做超参数优化）。"""
        self.X_train = X
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) + 1e-8
        y_normalized = (y - self.y_mean) / self.y_std

        self.length_scale = length_scale
        self.noise = noise
        K = self._kernel(X, X, length_scale) + (noise ** 2 + 1e-8) * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_normalized))
        return self

    def fit(self, X, y):
        """拟合 GP，并通过最大化对数边际似然优化核超参数。"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.length_scale.size == 1:
            self.length_scale = np.full(X.shape[1], self.length_scale[0])
        length_scale = self.length_scale.copy()
        noise = self.noise

        if self.optimize and len(X) > 2:
            y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-8)
            low_ls, high_ls = self.length_scale_bounds
            bounds = ([(np.log(low_ls), np.log(high_ls))] * X.shape[1]
                      + [(np.log(1e-4), np.log(0.3))])
            theta0 = np.log(np.append(length_scale, noise))
            theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])

            best_theta = theta0
            best_nll = self._negative_log_likelihood(theta0, X, y_normalized)[0]
            for _ in range(self.n_restarts):
                x0 = np.clip(theta0 + self.rng.normal(0.0, 0.5, size=theta0.shape),
                             [b[0] for b in bounds], [b[1] for b in bounds])
                result = minimize(self._negative_log_likelihood, x0, args=(X, y_normalized),
                                  method="L-BFGS-B", bounds=bounds, jac=True,
                                  options={"maxiter": 100})
                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_theta = result.x
            length_scale = np.exp(best_theta[:-1])
            noise = float(np.exp(best_theta[-1]))

        return self._set_training_data(X, y, length_scale, noise)

    def update(self, X, y):
        """
        固定当前核超参数，只用最新数据重算 L 和 alpha。

        数据量较大时，每次都做超参数优化是 O(n^3)·(优化迭代数) 的开销；
        在两次重拟合之间调用本方法可以把单次迭代成本降到一次 Cholesky。
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.length_scale.size == 1:
            self.length_scale = np.full(X.shape[1], self.length_scale[0])
        return self._set_training_data(X, y, self.length_scale, self.noise)

    def predict(self, X, return_std=True):
        """返回后验均值（以及标准差）。"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Ks = self._kernel(X, self.X_train, self.length_scale)
        mean = Ks @ self.alpha * self.y_std + self.y_mean
        if not return_std:
            return mean

        v = np.linalg.solve(self.L, Ks.T)
        # Matern 核的 K(x, x) 恒等于 signal_var，无需构造 n x n 的核矩阵
        var = self.signal_var - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12)) * self.y_std
        return mean, std


def expected_improvement(gp, X, best, xi=0.01):
    """
    Expected Improvement 采集函数（越大越好）。

    best 为当前最优目标值（通常取 GP 后验均值的最优值），
    xi 控制探索强度（数值越大越偏向开发）。
    """
    mean, std = gp.predict(X)
    improvement = mean - best - xi
    z = improvement / (std + 1e-12)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-9] = 0.0
    return ei


class BOAgent:
    """
    贝叶斯优化代理，接口风格与 SACAgent 保持一致。

    每个 trial：
        1. 选择一个归一化参数点 x ∈ [0, 1]^4
        2. 映射到环境的动作空间 action = low + x * (high - low)
        3. 重复执行 n_repeats 次 reset/step，用平均奖励作为目标值
        4. 用全部历史 (x, score) 拟合 GP，再用 EI 选出下一个点

    xi 为相对值，实际使用的阈值是 xi * std(scores)，因此不随目标量纲变化。
    """

    def __init__(self, env: Env,
                 n_init=100,
                 n_iter=900,
                 n_repeats=40,
                 xi=0.02,
                 length_scale=0.3,
                 length_scale_bounds=(0.02, 0.5),
                 noise=1e-3,
                 objective="reward",
                 n_candidates=1024,
                 refit_every=50,
                 max_train_points=300,
                 patience=None,
                 final_refine=True,
                 progress=True,
                 record_trials=True,
                 seed=None):
        if objective not in ("reward", "success"):
            raise ValueError("objective must be 'reward' or 'success'")

        self.hyperparameters = {
            "n_init": n_init,
            "n_iter": n_iter,
            "n_repeats": n_repeats,
            "xi": xi,
            "length_scale": length_scale,
            "noise": noise,
            "objective": objective,
        }

        self.env = env
        self.action_low = np.asarray(env.action_space.low, dtype=np.float64)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float64)
        self.action_dim = self.action_low.shape[0]
        self.action_bound = [self.action_low, self.action_high]

        self.n_init = n_init
        self.n_iter = n_iter
        self.n_repeats = n_repeats
        self.xi = xi
        self.objective = objective
        self.n_candidates = n_candidates
        self.refit_every = max(1, int(refit_every))
        self.max_train_points = max_train_points
        self.patience = patience
        self.final_refine = final_refine
        self.progress = progress
        self.record_trials = record_trials
        self.rng = np.random.RandomState(seed)

        self.gp = GaussianProcess(length_scale=length_scale,
                                  length_scale_bounds=length_scale_bounds,
                                  noise=noise,
                                  n_restarts=1,
                                  random_state=seed)

        self.X = []          # 归一化后的参数点
        self.scores = []     # 每个点对应的目标值（越大越好）
        self.history = []    # 每次评估的完整记录

        self.best_action = None
        self.best_score = -np.inf
        self.best_result = None

        self.now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    # ------------------------------------------------------------------ #
    # 参数归一化 / 反归一化
    # ------------------------------------------------------------------ #
    def normalize(self, action):
        """将环境参数映射到 [0, 1]。"""
        return (np.asarray(action, dtype=np.float64) - self.action_low) / (self.action_high - self.action_low)

    def denormalize(self, x):
        """将 [0, 1] 内的点映射回环境参数空间。"""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
        return self.action_low + x * (self.action_high - self.action_low)

    def action_names(self):
        """参数名称，用于保存结果和打印。"""
        names = ["l", "theta"] if self.env.polar_space else ["x", "y"]
        return names + ["v", "a"]

    def action_to_xy(self, action):
        """把动作转换成笛卡尔坐标，便于绘图（兼容 polar_space）。"""
        if self.env.polar_space:
            length, theta, _, _ = action
            return length * np.cos(np.radians(theta)), length * np.sin(np.radians(theta))
        x, y, _, _ = action
        return x, y

    # ------------------------------------------------------------------ #
    # 环境评估
    # ------------------------------------------------------------------ #
    def evaluate(self, action, n_repeats=None):
        """
        在环境中重复执行同一个参数，返回统计后的目标值。

        注意：env.step 每次只给出一次随机反应结果，因此这里多次
        reset/step 取平均，降低随机性对贝叶斯优化的干扰。

        每次 reset/step 的原始结果都会记录在返回值的 "trials" 字段中，
        包括 reaction（success/failure/none）、reward、state 等。
        """
        n_repeats = self.n_repeats if n_repeats is None else n_repeats
        action = np.asarray(action, dtype=np.float32)

        rewards = []
        reaction_times = []
        reactions = {"success": 0, "failure": 0, "none": 0}
        trials = []

        for repeat in range(n_repeats):
            self.env.reset()
            next_state, reward, done, info = self.env.step(action)
            reaction = info.get("reaction", "none")
            reaction_time = info.get("reaction_time", 0.0)

            rewards.append(reward)
            reaction_times.append(reaction_time)
            reactions[reaction] = reactions.get(reaction, 0) + 1

            if self.record_trials:
                trials.append({
                    "repeat": repeat + 1,
                    "reaction": reaction,
                    "reward": float(reward),
                    "reaction_time": float(reaction_time),
                    "state": int(next_state),
                    "suc_factor": float(self.env.tip.suc_factor),
                    "fail_factor": float(self.env.tip.fail_factor),
                    "non_factor": float(self.env.tip.non_factor),
                })

        success_rate = reactions["success"] / n_repeats
        failure_rate = reactions["failure"] / n_repeats
        none_rate = reactions["none"] / n_repeats

        if self.objective == "reward":
            score = float(np.mean(rewards))
        else:
            score = float(success_rate)

        return {
            "action": np.asarray(action, dtype=np.float64),
            "reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "none_rate": none_rate,
            "reaction_time": float(np.mean(reaction_times)),
            "score": score,
            "trials": trials,
        }

    # ------------------------------------------------------------------ #
    # 用 EI 选择下一个采样点
    # ------------------------------------------------------------------ #
    def select_action(self):
        """
        在 [0, 1]^4 内最大化 EI，返回下一个待评估的归一化参数点。
        候选点 = 全局随机点 + 当前最优点附近的局部扰动点，
        再对 EI 最高的候选点做逐步收缩的局部随机搜索。
        """
        # 用 GP 后验均值（去噪后）的最优值作为 EI 的比较基准，
        # 避免把"运气好"的观测值当成最优，导致 EI 被远处的探索项主导
        train_X = self.gp.X_train
        means = self.gp.predict(train_X, return_std=False)
        best_index = int(np.argmax(means))
        best = float(means[best_index])
        best_x = np.asarray(train_X[best_index], dtype=np.float64)

        random_points = self.rng.random((self.n_candidates, self.action_dim))
        local_points = np.clip(
            best_x + self.rng.normal(0.0, 0.15, size=(self.n_candidates // 2, self.action_dim)),
            0.0, 1.0,
        )
        candidates = np.vstack([random_points, local_points, best_x])

        # xi 相对于目标值的标准差，使其不随目标函数量纲变化
        xi = self.xi * (np.std(self.scores) + 1e-8)
        ei = expected_improvement(self.gp, candidates, best, xi)

        if not np.isfinite(ei).all() or np.max(ei) <= 0.0:
            return self.rng.random(self.action_dim)

        ei = ei + 1e-12 * self.rng.random(len(ei))  # 打破平局
        order = np.argsort(ei)[::-1]

        x_next = candidates[order[0]].copy()
        best_ei = float(ei[order[0]])

        # 局部精修：在 EI 最高的候选点附近做逐步收缩的随机搜索
        step = 0.1
        for _ in range(4):
            perturbed = np.clip(
                x_next + self.rng.normal(0.0, step, size=(64, self.action_dim)),
                0.0, 1.0,
            )
            local_ei = expected_improvement(self.gp, perturbed, best, xi)
            idx = int(np.argmax(local_ei))
            if local_ei[idx] > best_ei:
                best_ei = float(local_ei[idx])
                x_next = perturbed[idx]
            step *= 0.5

        return x_next

    def _training_subset(self, scores):
        """
        限制 GP 训练集大小（保留最优的一部分 + 最近的一部分 + 其余均匀抽样）。

        精确 GP 的单次拟合是 O(n^3)。预算很大（如 1000 次评估）时，
        用几百个代表性点已经足够刻画 4 维目标函数，可以显著缩短运行时间。
        """
        n = len(scores)
        if self.max_train_points is None or n <= self.max_train_points:
            return np.arange(n)

        keep = int(self.max_train_points)
        n_best = keep // 2
        n_recent = keep // 4

        best_index = np.argsort(scores)[::-1][:n_best]
        recent_index = np.arange(n - n_recent, n)
        chosen = np.union1d(best_index, recent_index)

        rest = np.setdiff1d(np.arange(n), chosen)
        n_extra = keep - len(chosen)
        if n_extra > 0 and len(rest) > 0:
            positions = np.linspace(0, len(rest) - 1, n_extra).astype(int)
            chosen = np.union1d(chosen, rest[positions])
        return chosen.astype(int)

    # ------------------------------------------------------------------ #
    # 训练（优化）主循环
    # ------------------------------------------------------------------ #
    def train(self, visualize=False):
        """
        先做拉丁超立方初始化，再用贝叶斯优化迭代寻优。

        - 每 refit_every 次迭代重新优化一次 GP 超参数，其余迭代只做
          廉价的 Cholesky 更新，避免大数据量下 O(n^3) 反复优化
        - patience 不为 None 时，连续 patience 次没有刷新最优就提前停止
        参数:
            visualize: True 时实时刷新优化过程；False 时只在结束后保存图片。
        返回:
            (best_action, best_score)
        """
        total = self.n_init + self.n_iter
        init_points = latin_hypercube(self.n_init, self.action_dim, self.rng)

        fig = axes = None
        if visualize:
            plt.ion()
            fig, axes = self._create_figure()
            plt.show(block=False)

        pbar = _make_progress_bar(total, "Bayesian Optimization") if self.progress else None
        last_improve = 0
        start_time = time.time()

        def log(message):
            if pbar is not None:
                pbar.write(message)
            else:
                print(message)

        for step in range(total):
            if step < self.n_init:
                x = init_points[step]
                phase = "init"
            else:
                scores = np.array(self.scores)
                index = self._training_subset(scores)
                X_train = np.array(self.X)[index]
                y_train = scores[index]
                if (step - self.n_init) % self.refit_every == 0:
                    self.gp.fit(X_train, y_train)      # 重新优化 GP 超参数
                else:
                    self.gp.update(X_train, y_train)   # 固定超参数，只重算 L / alpha
                x = self.select_action()
                phase = "bo"

            action = self.denormalize(x)
            result = self.evaluate(action)
            result["phase"] = phase

            self.X.append(np.asarray(x, dtype=np.float64))
            self.scores.append(result["score"])
            self.history.append(result)

            if result["score"] > self.best_score:
                self.best_score = result["score"]
                self.best_action = result["action"]
                self.best_result = result
                last_improve = step
                log(f"[new best] step {step + 1}/{total} | {np.round(action, 3)} | "
                    f"reward={result['reward']:.3f} | success={result['success_rate']:.2f} | "
                    f"score={self.best_score:.3f}")

            if visualize:
                self._draw_figure(fig, axes)
                fig.canvas.draw()
                plt.pause(0.01)

            if pbar is not None:
                pbar.set_postfix(phase=phase, best=f"{self.best_score:.3f}",
                                 cur=f"{result['score']:.2f}")
                pbar.update(1)

            if self.patience is not None and step - last_improve >= self.patience:
                log(f"Early stop: no improvement in the last {self.patience} evaluations.")
                break

        if pbar is not None:
            pbar.close()

        if self.final_refine:
            self.refine_best()

        # 用最终历史刷新一次 GP（固定超参数，开销很小）
        scores = np.array(self.scores)
        index = self._training_subset(scores)
        self.gp.update(np.array(self.X)[index], scores[index])

        elapsed = time.time() - start_time
        names = self.action_names()
        best_str = ", ".join(f"{name}={value:.3f}" for name, value in zip(names, self.best_action))
        print(f"Best action: {best_str}")
        print(f"Best score: {self.best_score:.4f} | elapsed: {elapsed:.1f}s")

        if visualize:
            self._draw_figure(fig, axes)
            self._save_figure(fig)
            plt.ioff()
            plt.close(fig)
        else:
            self.plot_train(self.hyperparameters)

        self.save()
        return self.best_action, self.best_score

    def refine_best(self, top_k=5, repeat_factor=5):
        """
        用更多的重复次数重新评估表现最好的几个点。

        BO 过程中每个点的重复次数有限，最优点的排名可能受随机噪声影响；
        这里对 top-k 个点做更精确的估计，使最终给出的参数更可靠。
        """
        if not self.history:
            return

        selected = []
        for idx in np.argsort(self.scores)[::-1]:
            if len(selected) >= top_k:
                break
            x = np.asarray(self.X[idx])
            if all(np.linalg.norm(x - np.asarray(self.X[j])) > 0.02 for j in selected):
                selected.append(idx)

        best_score = -np.inf
        best_action = self.best_action
        best_result = self.best_result
        for idx in selected:
            action = self.denormalize(self.X[idx])
            result = self.evaluate(action, n_repeats=self.n_repeats * repeat_factor)
            result["phase"] = "refine"

            self.history.append(result)
            self.scores.append(result["score"])
            self.X.append(np.asarray(self.X[idx], dtype=np.float64))

            # 用高重复次数的估计值作为最终结果，避免"运气好"的观测被当成最优
            if result["score"] > best_score:
                best_score = result["score"]
                best_action = result["action"]
                best_result = result

            print(f"[refine] action={np.round(action, 3)} | reward={result['reward']:.3f} "
                  f"| success={result['success_rate']:.2f} | best={best_score:.3f}")

        self.best_score = best_score
        self.best_action = best_action
        self.best_result = best_result

    # ------------------------------------------------------------------ #
    # 绘图
    # ------------------------------------------------------------------ #
    def _create_figure(self):
        """创建与 SAC 相同的 2x2 布局：x-y 分布、v-a 分布、目标曲线。"""
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[5, 3], width_ratios=[5, 5])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        return fig, (ax1, ax2, ax3)

    def _draw_figure(self, fig, axes):
        """根据评估历史重绘整张图。"""
        ax1, ax2, ax3 = axes
        for ax in axes:
            ax.clear()

        action_bound = self.action_bound
        if self.env.polar_space:
            max_l = action_bound[1][0]
            ax1.set_xlim(-max_l, max_l)
            ax1.set_ylim(-max_l, max_l)
        else:
            ax1.set_xlim(action_bound[0][0], action_bound[1][0])
            ax1.set_ylim(action_bound[0][1], action_bound[1][1])
        ax1.set_title("x, y Distribution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2.set_xlim(action_bound[0][2], action_bound[1][2])
        ax2.set_ylim(action_bound[0][3], action_bound[1][3])
        ax2.set_title("v, a Distribution")
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")

        data_x, data_y, colors_xy = [], [], []
        data_v, data_a, colors_va = [], [], []
        for result in self.history:
            x, y = self.action_to_xy(result["action"])
            _, _, v, a = result["action"]

            if result["success_rate"] >= 0.5:
                color = (0, 1, 0, 0.5)      # 半透明绿色：成功为主
            elif result["failure_rate"] >= 0.5:
                color = (1, 0, 0, 0.5)      # 半透明红色：失败为主
            else:
                color = (0.5, 0.5, 0.5, 0.5)  # 半透明灰色：无明显反应

            data_x.append(x)
            data_y.append(y)
            colors_xy.append(color)
            data_v.append(v)
            data_a.append(a)
            colors_va.append(color)

        if data_x:
            ax1.scatter(data_x, data_y, c=colors_xy)
            ax2.scatter(data_v, data_a, c=colors_va)

        if self.best_action is not None:
            best_x, best_y = self.action_to_xy(self.best_action)
            ax1.scatter([best_x], [best_y], marker="*", s=260, facecolors="none",
                        edgecolors="black", linewidths=1.5, label="best")
            ax2.scatter([self.best_action[2]], [self.best_action[3]], marker="*", s=260,
                        facecolors="none", edgecolors="black", linewidths=1.5)
            ax1.legend(loc="upper right")

        if self.history:
            scores = np.array([result["score"] for result in self.history])
            best_so_far = np.maximum.accumulate(scores)
            steps = np.arange(1, len(scores) + 1)
            ax3.plot(steps, scores, color="blue", marker="o", markersize=3,
                     label="score per evaluation")
            ax3.plot(steps, best_so_far, color="green", label="best so far")
            ax3.axhline(self.best_score, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
            ax3.legend(loc="lower right")
        ax3.set_title(f"Optimization Progress ({self.objective})")
        ax3.set_xlabel("evaluation")
        ax3.set_ylabel("score")
        ax3.grid(alpha=0.3)

        fig.suptitle("Bayesian Optimization on Molecular Simulation", fontsize=12)
        plt.tight_layout()

    def _save_figure(self, fig, hyperparameters=None):
        if not os.path.exists("./bo_result_plot"):
            os.makedirs("./bo_result_plot")
        hyperparameters = self.hyperparameters if hyperparameters is None else hyperparameters
        str_hyperparameters = '_'.join([f"{key}={value}" for key, value in hyperparameters.items()])
        fig.savefig(f"bo_result_plot/bo_result_{self.now}_{str_hyperparameters}.png", dpi=150)

    def plot_train(self, hyperparameters):
        """训练结束后绘制并保存结果图（与 SAC 的 plot_train 对应）。"""
        fig, axes = self._create_figure()
        self._draw_figure(fig, axes)
        self._save_figure(fig, hyperparameters)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    def save(self):
        """把优化历史与最优参数保存到 bo_result_plot 目录。"""
        if not os.path.exists("./bo_result_plot"):
            os.makedirs("./bo_result_plot")

        names = self.action_names()
        history_path = f"bo_result_plot/bo_history_{self.now}.txt"
        with open(history_path, "w", encoding="utf-8") as f:
            f.write(f"best_score: {self.best_score:.6f}\n")
            f.write("best_action: " + ", ".join(
                f"{name}={value:.6f}" for name, value in zip(names, self.best_action)) + "\n")
            header = ["step", "phase"] + names + [
                "reward", "success_rate", "failure_rate", "none_rate", "score"]
            f.write("\t".join(header) + "\n")
            for step, result in enumerate(self.history, start=1):
                values = [str(step), result["phase"]]
                values += [f"{value:.6f}" for value in result["action"]]
                values += [f"{result['reward']:.6f}", f"{result['success_rate']:.6f}",
                           f"{result['failure_rate']:.6f}", f"{result['none_rate']:.6f}",
                           f"{result['score']:.6f}"]
                f.write("\t".join(values) + "\n")

        # 原始交互数据：每次 reset/step 的反应结果
        trials_path = None
        if self.record_trials:
            trials_path = f"bo_result_plot/bo_trials_{self.now}.txt"
            with open(trials_path, "w", encoding="utf-8") as f:
                header = (["evaluation", "phase", "repeat"] + names
                          + ["reaction", "reward", "reaction_time", "state",
                             "suc_factor", "fail_factor", "non_factor"])
                f.write("\t".join(header) + "\n")
                for step, result in enumerate(self.history, start=1):
                    action_values = [f"{value:.6f}" for value in result["action"]]
                    for trial in result.get("trials", []):
                        values = [str(step), result["phase"], str(trial["repeat"])]
                        values += action_values
                        values += [trial["reaction"], f"{trial['reward']:.6f}",
                                   f"{trial['reaction_time']:.6f}", str(trial["state"]),
                                   f"{trial['suc_factor']:.6f}",
                                   f"{trial['fail_factor']:.6f}",
                                   f"{trial['non_factor']:.6f}"]
                        f.write("\t".join(values) + "\n")

        best_path = f"bo_result_plot/bo_best_{self.now}.json"
        best_result = self.best_result if self.best_result is not None else self.history[-1]
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_score": self.best_score,
                "objective": self.objective,
                "hyperparameters": self.hyperparameters,
                "best_action": {name: float(value)
                                for name, value in zip(names, self.best_action)},
                "best_success_rate": best_result["success_rate"],
                "best_reward": best_result["reward"],
                "evaluations": len(self.history),
                "trials_file": trials_path,
                "trials": sum(len(result.get("trials", [])) for result in self.history),
            }, f, indent=4, ensure_ascii=False)

        print("Save the optimization history successfully!")
        print("History path:", history_path)
        if trials_path is not None:
            print("Trials path: ", trials_path)
        print("Best path:   ", best_path)


# 运行贝叶斯优化
if __name__ == "__main__":
    # 多种子重复实验：
    # for seed in range(5):
    #     env = Env(polar_space=False)
    #     agent = BOAgent(env, seed=seed)
    #     agent.train(visualize=False)

    env = Env(polar_space=False)
    agent = BOAgent(env,
                    n_init=6,          # 初始点（约占预算 10%）
                    n_iter=94,          # 贝叶斯优化迭代次数
                    n_repeats=10,        # 每个点重复 10 次，降低随机噪声
                    xi=0.02,             # 数据量大时更偏开发
                    n_candidates=1024,   # 降低 EI 预测开销（n=1000 时每轮约 0.3 s）
                    refit_every=50,      # 每 10 轮重拟合一次 GP 超参数
                    max_train_points=None,  # GP 训练集上限（1000 条历史中保留 300 个代表点）
                    patience=None,        # 连续 200 次没刷新最优就提前停止；想跑满 1000 次设为 None
                    seed=0)
    agent.train(visualize=False)
