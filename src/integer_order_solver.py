# integer_order_solver_advanced.py
"""
增强版整数阶 UAU-SIS 模型求解器
支持多种网络类型和灵活的参数配置
用于实验1和实验4
"""

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, List, Union, Optional
from enum import Enum
from tqdm import tqdm
import time


class NetworkType(Enum):
    """支持的网络类型"""
    BA = "barabasi_albert"  # 无标度网络
    ER = "erdos_renyi"  # 随机网络
    REGULAR = "regular"  # 规则网络
    CUSTOM = "custom"  # 自定义网络


class UAU_SIS_IntegerSolver:
    """增强版整数阶 UAU-SIS 模型求解器"""

    def __init__(self,
                 N: int = 2500,
                 network_type_consciousness: NetworkType = NetworkType.BA,
                 network_type_risk: NetworkType = NetworkType.BA,
                 network_params_consciousness: Dict = None,
                 network_params_risk: Dict = None,
                 custom_A: np.ndarray = None,
                 custom_B: np.ndarray = None,
                 seed: int = 42):
        """
        初始化求解器

        Parameters:
        -----------
        N : int
            网络节点数
        network_type_consciousness : NetworkType
            意识层网络类型
        network_type_risk : NetworkType
            风险层网络类型
        network_params_consciousness : Dict
            意识层网络参数
        network_params_risk : Dict
            风险层网络参数
        custom_A : np.ndarray
            自定义意识层邻接矩阵
        custom_B : np.ndarray
            自定义风险层邻接矩阵
        seed : int
            随机种子
        """
        self.N = N
        self.seed = seed
        np.random.seed(seed)

        # 设置默认网络参数
        if network_params_consciousness is None:
            network_params_consciousness = self._get_default_network_params(network_type_consciousness)
        if network_params_risk is None:
            network_params_risk = self._get_default_network_params(network_type_risk)

        # 生成或使用自定义网络
        if custom_A is not None:
            self.A = custom_A
            self.G_consciousness = None  # 自定义网络没有图对象
        else:
            self.G_consciousness = self._generate_network(network_type_consciousness,
                                                          network_params_consciousness)
            self.A = nx.to_numpy_array(self.G_consciousness)

        if custom_B is not None:
            self.B = custom_B
            self.G_risk = None
        else:
            self.G_risk = self._generate_network(network_type_risk, network_params_risk)
            self.B = nx.to_numpy_array(self.G_risk)

        # 验证网络维度
        if self.A.shape != (N, N) or self.B.shape != (N, N):
            raise ValueError(f"邻接矩阵维度不匹配: A{self.A.shape}, B{self.B.shape}, 期望 ({N},{N})")

        # 默认参数（根据论文实验设置）
        self.params = {
            'lambd': 0.3,  # 意识传播率
            'beta_U': 0.6,  # 未知者的感染率
            'gamma': 1 / 3,  # 意识者的感染率降低因子 β^A = γβ^U
            'delta': 0.5,  # 遗忘率
            'mu': 1.0,  # 恢复率
        }

        # 网络统计信息
        self.network_info = {
            'consciousness': {
                'type': network_type_consciousness,
                'params': network_params_consciousness,
                'avg_degree': np.mean(np.sum(self.A, axis=1))
            },
            'risk': {
                'type': network_type_risk,
                'params': network_params_risk,
                'avg_degree': np.mean(np.sum(self.B, axis=1))
            }
        }

        # 进度条相关
        self.progress_bar = None
        self.last_update_time = 0

    def _get_default_network_params(self, network_type: NetworkType) -> Dict:
        """获取默认网络参数"""
        if network_type == NetworkType.BA:
            return {'m': 4}  # BA网络连接数
        elif network_type == NetworkType.ER:
            return {'p': 0.008}  # ER网络连接概率 (N=2500时平均度≈20)
        elif network_type == NetworkType.REGULAR:
            return {'k': 10}  # 规则网络度
        else:
            return {}

    def _generate_network(self, network_type: NetworkType, params: Dict) -> nx.Graph:
        """生成指定类型的网络"""
        if network_type == NetworkType.BA:
            return nx.barabasi_albert_graph(self.N, params['m'], seed=self.seed)
        elif network_type == NetworkType.ER:
            return nx.erdos_renyi_graph(self.N, params['p'], seed=self.seed)
        elif network_type == NetworkType.REGULAR:
            return nx.random_regular_graph(params['k'], self.N, seed=self.seed)
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")

    def set_parameters(self, **kwargs):
        """设置模型参数"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            elif key == 'beta_A':  # 如果直接设置beta_A，计算对应的gamma
                self.params['gamma'] = value / self.params['beta_U']

    def get_network_info(self) -> Dict:
        """获取网络统计信息"""
        return self.network_info.copy()

    def set_custom_networks(self, A: np.ndarray, B: np.ndarray):
        """设置自定义网络"""
        if A.shape != (self.N, self.N) or B.shape != (self.N, self.N):
            raise ValueError(f"自定义网络维度不匹配: A{A.shape}, B{B.shape}, 期望 ({self.N},{self.N})")
        self.A = A
        self.B = B
        self.G_consciousness = None
        self.G_risk = None

        # 更新网络信息
        self.network_info['consciousness'] = {
            'type': NetworkType.CUSTOM,
            'params': {},
            'avg_degree': np.mean(np.sum(self.A, axis=1))
        }
        self.network_info['risk'] = {
            'type': NetworkType.CUSTOM,
            'params': {},
            'avg_degree': np.mean(np.sum(self.B, axis=1))
        }

    def _compute_neighbor_influence(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每个节点的邻居影响

        Parameters:
        -----------
        P : np.ndarray, shape (3N,)
            状态概率向量 [US_0, AS_0, AI_0, US_1, AS_1, AI_1, ...]

        Returns:
        --------
        M_A : np.ndarray, shape (N,)
            每个节点的意识邻居影响
        M_I : np.ndarray, shape (N,)
            每个节点的感染邻居影响
        """
        # 重塑为 (N, 3) 矩阵
        P_matrix = P.reshape(self.N, 3)

        # p^AS + p^AI 用于意识传播
        p_aware = P_matrix[:, 1] + P_matrix[:, 2]  # AS + AI

        # p^AI 用于感染传播
        p_infected = P_matrix[:, 2]  # AI

        # 计算邻居影响
        M_A = self.A @ p_aware  # 意识邻居影响
        M_I = self.B @ p_infected  # 感染邻居影响

        return M_A, M_I

    def derivative(self, t: float, P: np.ndarray) -> np.ndarray:
        """
        计算整数阶微分方程的导数

        Parameters:
        -----------
        t : float
            当前时间
        P : np.ndarray, shape (3N,)
            当前状态概率向量

        Returns:
        --------
        dP_dt : np.ndarray, shape (3N,)
            状态概率的导数
        """
        # 获取参数
        lambda_val = self.params['lambd']
        beta_U = self.params['beta_U']
        beta_A = self.params['gamma'] * beta_U
        delta = self.params['delta']
        mu = self.params['mu']

        # 计算邻居影响
        M_A, M_I = self._compute_neighbor_influence(P)

        # 重塑为 (N, 3) 矩阵便于处理
        P_matrix = P.reshape(self.N, 3)
        dP_dt = np.zeros_like(P_matrix)

        for i in range(self.N):
            p_US, p_AS, p_AI = P_matrix[i]

            # US 状态的导数
            dP_dt[i, 0] = -(lambda_val * M_A[i] + beta_U * M_I[i]) * p_US + delta * p_AS

            # AS 状态的导数
            dP_dt[i, 1] = (lambda_val * M_A[i]) * p_US - (delta + beta_A * M_I[i]) * p_AS + mu * p_AI

            # AI 状态的导数
            dP_dt[i, 2] = (beta_U * M_I[i]) * p_US + (beta_A * M_I[i]) * p_AS - mu * p_AI

        return dP_dt.flatten()

    def generate_initial_conditions(self,
                                    initial_state: str = 'US',
                                    infected_fraction: float = 0.02,
                                    aware_fraction: float = 0.03,
                                    custom_initial: np.ndarray = None) -> np.ndarray:
        """
        生成初始条件

        Parameters:
        -----------
        initial_state : str
            主导状态类型 ('US', 'AS', 'AI', 'mixed')
        infected_fraction : float
            初始感染节点比例（仅当initial_state='mixed'时使用）
        aware_fraction : float
            初始意识节点比例（仅当initial_state='mixed'时使用）
        custom_initial : np.ndarray
            自定义初始条件，形状为 (3N,)

        Returns:
        --------
        P0 : np.ndarray, shape (3N,)
            初始状态概率向量
        """
        if custom_initial is not None:
            if custom_initial.shape != (3 * self.N,):
                raise ValueError(f"自定义初始条件维度错误: {custom_initial.shape}, 期望 ({3 * self.N},)")
            return custom_initial

        P0 = np.zeros(3 * self.N)

        for i in range(self.N):
            if initial_state == 'US':
                p_vec = np.array([0.98, 0.01, 0.01])  # 大部分为US
            elif initial_state == 'AS':
                p_vec = np.array([0.01, 0.98, 0.01])  # 大部分为AS
            elif initial_state == 'AI':
                p_vec = np.array([0.01, 0.01, 0.98])  # 大部分为AI
            elif initial_state == 'mixed':
                # 按指定比例混合
                p_US = 1.0 - infected_fraction - aware_fraction
                p_AS = aware_fraction
                p_AI = infected_fraction
                p_vec = np.array([p_US, p_AS, p_AI])
            else:
                raise ValueError(f"不支持的初始状态类型: {initial_state}")

            # 添加小随机扰动
            perturbation = np.random.normal(0, 0.01, 3)
            p_vec = p_vec + perturbation
            p_vec = np.clip(p_vec, 0, 1)  # 确保概率在 [0,1] 范围内
            p_vec = p_vec / np.sum(p_vec)  # 重新归一化

            P0[3 * i:3 * i + 3] = p_vec

        return P0

    def _progress_callback(self, t: float, P: np.ndarray, t_span: Tuple[float, float]):
        """进度回调函数"""
        current_time = time.time()
        if current_time - self.last_update_time > 0.1:  # 每0.1秒更新一次
            progress = (t - t_span[0]) / (t_span[1] - t_span[0])
            if self.progress_bar is not None:
                self.progress_bar.n = int(progress * 100)
                self.progress_bar.refresh()
            self.last_update_time = current_time

    def solve(self,
              t_span: Tuple[float, float] = (0, 8.0),
              t_eval: np.ndarray = None,
              initial_conditions: np.ndarray = None,
              initial_conditions_kwargs: Dict = None,
              solver_kwargs: Dict = None,
              show_progress: bool = True) -> Dict:
        """
        求解整数阶微分方程

        Parameters:
        -----------
        t_span : tuple
            时间区间 (t_start, t_end)
        t_eval : np.ndarray
            需要输出的时间点
        initial_conditions : np.ndarray
            自定义初始条件
        initial_conditions_kwargs : Dict
            传递给generate_initial_conditions的参数
        solver_kwargs : Dict
            传递给 solve_ivp 的参数
        show_progress : bool
            是否显示进度条

        Returns:
        --------
        result : dict
            包含求解结果的字典
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 201)

        # 生成初始条件
        if initial_conditions is not None:
            P0 = initial_conditions
        else:
            if initial_conditions_kwargs is None:
                initial_conditions_kwargs = {}
            P0 = self.generate_initial_conditions(**initial_conditions_kwargs)

        # 默认求解器参数
        default_solver_kwargs = {
            'method': 'RK45',
            'rtol': 1e-6,
            'atol': 1e-8,
        }

        # 添加进度回调
        if show_progress:
            self.progress_bar = tqdm(total=100, desc="求解进度", unit="%")
            self.last_update_time = time.time()

            def wrapped_derivative(t, P):
                self._progress_callback(t, P, t_span)
                return self.derivative(t, P)

            default_solver_kwargs['fun'] = wrapped_derivative
        else:
            default_solver_kwargs['fun'] = self.derivative

        if solver_kwargs is not None:
            default_solver_kwargs.update(solver_kwargs)

        # 求解ODE
        sol = solve_ivp(
            default_solver_kwargs['fun'],
            t_span,
            P0,
            t_eval=t_eval,
            method=default_solver_kwargs['method'],
            rtol=default_solver_kwargs['rtol'],
            atol=default_solver_kwargs['atol']
        )

        # 关闭进度条
        if show_progress and self.progress_bar is not None:
            self.progress_bar.n = 100
            self.progress_bar.refresh()
            self.progress_bar.close()
            self.progress_bar = None

        # 计算总体统计量
        P_matrix = sol.y.reshape(self.N, 3, -1)  # (N, 3, time_steps)

        # 总体密度
        rho_US = np.mean(P_matrix[:, 0, :], axis=0)  # US 密度
        rho_AS = np.mean(P_matrix[:, 1, :], axis=0)  # AS 密度
        rho_AI = np.mean(P_matrix[:, 2, :], axis=0)  # AI 密度
        rho_A = rho_AS + rho_AI  # 总意识密度
        rho_I = rho_AI  # 总感染密度

        # 节点级别的统计（可选，用于更详细的分析）
        node_level_stats = {
            'final_US': P_matrix[:, 0, -1],
            'final_AS': P_matrix[:, 1, -1],
            'final_AI': P_matrix[:, 2, -1],
        }

        return {
            't': sol.t,
            'P': sol.y,
            'rho_US': rho_US,
            'rho_AS': rho_AS,
            'rho_AI': rho_AI,
            'rho_A': rho_A,
            'rho_I': rho_I,
            'node_level_stats': node_level_stats,
            'success': sol.success,
            'network_info': self.get_network_info(),
            'params': self.params.copy()
        }