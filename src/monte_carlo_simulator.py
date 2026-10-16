# monte_carlo_simulator_fixed.py
"""
修正版的UAU-SIS蒙特卡洛模拟器
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from enum import Enum


class NodeState(Enum):
    US = 0  # Unaware Susceptible
    AS = 1  # Aware Susceptible
    AI = 2  # Aware Infected


class UAU_SIS_MonteCarlo:
    """修正版的UAU-SIS模型蒙特卡洛模拟器"""

    def __init__(self,
                 N: int = 1000,
                 network_type_consciousness: str = "BA",
                 network_type_risk: str = "BA",
                 network_params_consciousness: Dict = None,
                 network_params_risk: Dict = None,
                 seed: int = 42):

        self.N = N
        self.seed = seed
        np.random.seed(seed)

        # 设置默认网络参数
        if network_params_consciousness is None:
            network_params_consciousness = self._get_default_network_params(network_type_consciousness)
        if network_params_risk is None:
            network_params_risk = self._get_default_network_params(network_type_risk)

        # 生成网络
        self.A = self._generate_adjacency_matrix(network_type_consciousness,
                                                 network_params_consciousness)  # 意识层
        self.B = self._generate_adjacency_matrix(network_type_risk,
                                                 network_params_risk)  # 风险层

        # 存储邻居列表以提高效率
        self.neighbors_consciousness = [np.where(self.A[i] > 0)[0] for i in range(N)]
        self.neighbors_risk = [np.where(self.B[i] > 0)[0] for i in range(N)]

        # 默认参数
        self.params = {
            'lambd': 0.3,  # 意识传播率
            'beta_U': 0.6,  # 未知者的感染率
            'gamma': 1 / 3,  # 意识者的感染率降低因子
            'delta': 0.5,  # 遗忘率
            'mu': 1.0,  # 恢复率
        }

        # 计算 beta_A
        self.params['beta_A'] = self.params['gamma'] * self.params['beta_U']

    def _get_default_network_params(self, network_type: str) -> Dict:
        """获取默认网络参数"""
        if network_type == "BA":
            return {'m': 4}
        elif network_type == "ER":
            return {'p': 0.008}
        elif network_type == "Regular":
            return {'k': 10}
        else:
            return {}

    def _generate_adjacency_matrix(self, network_type: str, params: Dict) -> np.ndarray:
        """生成邻接矩阵"""
        if network_type == "BA":
            G = nx.barabasi_albert_graph(self.N, params['m'], seed=self.seed)
        elif network_type == "ER":
            G = nx.erdos_renyi_graph(self.N, params['p'], seed=self.seed)
        elif network_type == "Regular":
            G = nx.random_regular_graph(params['k'], self.N, seed=self.seed)
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")

        return nx.to_numpy_array(G)

    def set_parameters(self, **kwargs):
        """设置模型参数"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # 更新 beta_A
        if 'beta_U' in kwargs or 'gamma' in kwargs:
            self.params['beta_A'] = self.params['gamma'] * self.params['beta_U']

    def initialize_states(self,
                          initial_infected: float = 0.02,
                          initial_aware: float = 0.03) -> np.ndarray:
        """
        初始化节点状态
        """
        states = np.full(self.N, NodeState.US.value)

        # 随机选择初始感染节点
        n_infected = int(self.N * initial_infected)
        infected_indices = np.random.choice(self.N, n_infected, replace=False)
        states[infected_indices] = NodeState.AI.value

        # 从剩余节点中选择初始意识节点
        remaining_indices = np.setdiff1d(np.arange(self.N), infected_indices)
        n_aware = int(self.N * initial_aware)
        if n_aware > 0:
            aware_indices = np.random.choice(remaining_indices, n_aware, replace=False)
            states[aware_indices] = NodeState.AS.value

        return states

    def _count_aware_neighbors(self, node: int) -> int:
        """计算节点的意识邻居数量（AS + AI）"""
        neighbors = self.neighbors_consciousness[node]
        return np.sum((self.states[neighbors] == NodeState.AS.value) |
                      (self.states[neighbors] == NodeState.AI.value))

    def _count_infected_neighbors(self, node: int) -> int:
        """计算节点的感染邻居数量（AI）"""
        neighbors = self.neighbors_risk[node]
        return np.sum(self.states[neighbors] == NodeState.AI.value)

    def _calculate_total_rate(self) -> Tuple[float, List[Tuple]]:
        """
        计算所有可能事件的总速率和事件列表
        """
        events = []
        total_rate = 0.0

        for i in range(self.N):
            current_state = self.states[i]

            if current_state == NodeState.US.value:
                # US -> AS: 意识传播
                num_aware_neighbors = self._count_aware_neighbors(i)
                rate_AS = self.params['lambd'] * num_aware_neighbors
                if rate_AS > 0:
                    events.append((i, NodeState.AS, rate_AS))
                    total_rate += rate_AS

                # US -> AI: 感染
                num_infected_neighbors = self._count_infected_neighbors(i)
                rate_AI = self.params['beta_U'] * num_infected_neighbors
                if rate_AI > 0:
                    events.append((i, NodeState.AI, rate_AI))
                    total_rate += rate_AI

            elif current_state == NodeState.AS.value:
                # AS -> US: 遗忘
                rate_US = self.params['delta']
                events.append((i, NodeState.US, rate_US))
                total_rate += rate_US

                # AS -> AI: 感染
                num_infected_neighbors = self._count_infected_neighbors(i)
                rate_AI = self.params['beta_A'] * num_infected_neighbors
                if rate_AI > 0:
                    events.append((i, NodeState.AI, rate_AI))
                    total_rate += rate_AI

            elif current_state == NodeState.AI.value:
                # AI -> AS: 恢复
                rate_AS = self.params['mu']
                events.append((i, NodeState.AS, rate_AS))
                total_rate += rate_AS

        return total_rate, events

    def _select_event(self, total_rate: float, events: List[Tuple]) -> Tuple[int, NodeState]:
        """
        根据速率比例随机选择发生的事件
        """
        if total_rate == 0:
            return None, None

        # 随机选择事件
        r = np.random.random() * total_rate
        cumulative_rate = 0.0

        for node, new_state, rate in events:
            cumulative_rate += rate
            if cumulative_rate >= r:
                return node, new_state

        # 如果由于浮点误差没有选中，返回最后一个事件
        return events[-1][0], events[-1][1]

    def simulate(self,
                 T: float = 10.0,
                 record_interval: float = 0.04,
                 initial_infected: float = 0.02,
                 initial_aware: float = 0.03,
                 progress_bar: bool = True) -> Dict:
        """
        运行单次蒙特卡洛模拟
        """
        # 初始化状态
        self.states = self.initialize_states(initial_infected, initial_aware)

        # 初始化记录
        time_points = np.arange(0, T + record_interval, record_interval)
        n_records = len(time_points)
        densities = {
            'US': np.zeros(n_records),
            'AS': np.zeros(n_records),
            'AI': np.zeros(n_records)
        }

        # 记录初始状态
        densities['US'][0] = np.sum(self.states == NodeState.US.value) / self.N
        densities['AS'][0] = np.sum(self.states == NodeState.AS.value) / self.N
        densities['AI'][0] = np.sum(self.states == NodeState.AI.value) / self.N

        # Gillespie算法
        current_time = 0.0
        record_idx = 1

        if progress_bar:
            pbar = tqdm(total=T, desc="Monte Carlo Simulation")

        while current_time < T and record_idx < n_records:
            # 计算总速率和事件列表
            total_rate, events = self._calculate_total_rate()

            if total_rate == 0:
                # 没有可能的事件，跳出循环
                break

            # 生成下一个事件时间
            dt = np.random.exponential(1.0 / total_rate)
            current_time += dt

            # 记录在到达下一个记录点之前的状态
            while record_idx < n_records and time_points[record_idx] <= current_time:
                densities['US'][record_idx] = np.sum(self.states == NodeState.US.value) / self.N
                densities['AS'][record_idx] = np.sum(self.states == NodeState.AS.value) / self.N
                densities['AI'][record_idx] = np.sum(self.states == NodeState.AI.value) / self.N
                record_idx += 1

            # 选择并执行事件
            node, new_state = self._select_event(total_rate, events)
            if node is not None:
                self.states[node] = new_state.value

            if progress_bar:
                pbar.update(min(dt, T - current_time + dt))

        # 填充剩余记录点
        while record_idx < n_records:
            densities['US'][record_idx] = np.sum(self.states == NodeState.US.value) / self.N
            densities['AS'][record_idx] = np.sum(self.states == NodeState.AS.value) / self.N
            densities['AI'][record_idx] = np.sum(self.states == NodeState.AI.value) / self.N
            record_idx += 1

        if progress_bar:
            pbar.close()

        return {
            'time_points': time_points,
            'densities': densities,
            'final_states': self.states.copy()
        }

    def run_multiple_simulations(self,
                                 n_simulations: int = 20,
                                 T: float = 10.0,
                                 record_interval: float = 0.04,
                                 initial_infected: float = 0.02,
                                 initial_aware: float = 0.03) -> Dict:
        """
        运行多次蒙特卡洛模拟并计算统计量
        """
        all_densities = {
            'US': [],
            'AS': [],
            'AI': []
        }

        print(f"Running {n_simulations} Monte Carlo simulations...")

        for i in range(n_simulations):
            if i > 0:
                # 改变随机种子以获得不同的模拟轨迹
                np.random.seed(self.seed + i)

            result = self.simulate(T, record_interval, initial_infected, initial_aware,
                                   progress_bar=(i == 0))

            all_densities['US'].append(result['densities']['US'])
            all_densities['AS'].append(result['densities']['AS'])
            all_densities['AI'].append(result['densities']['AI'])

        # 计算平均值和方差
        time_points = result['time_points']
        n_points = len(time_points)

        mean_densities = {}
        variance_densities = {}

        for state in ['US', 'AS', 'AI']:
            # 转换为 numpy 数组以便计算
            data = np.array(all_densities[state])
            mean_densities[state] = np.mean(data, axis=0)
            variance_densities[state] = np.var(data, axis=0)

        return {
            'time_points': time_points,
            'mean_densities': mean_densities,
            'variance_densities': variance_densities,
            'all_simulations': all_densities,
            'n_simulations': n_simulations,
            'parameters': self.params.copy()
        }


# 测试函数
def test_fixed_monte_carlo():
    """测试修正后的蒙特卡洛模拟器"""

    print("测试修正后的蒙特卡洛模拟器...")

    # 创建模拟器
    simulator = UAU_SIS_MonteCarlo_Fixed(
        N=1000,
        network_type_consciousness="BA",
        network_type_risk="BA",
        network_params_consciousness={'m': 4},
        network_params_risk={'m': 2},
        seed=42
    )

    # 设置参数
    simulator.set_parameters(
        lambd =0.3,
               beta_U=0.6,
               gamma=1 / 3,
               delta=0.5,
               mu=1.0
    )

    # 运行单次模拟
    print("运行单次模拟...")
    result = simulator.simulate(
        T=8.0,
        record_interval=0.04,
        initial_infected=0.02,
        initial_aware=0.03,
        progress_bar=True
    )

    print(f"最终状态: US={result['densities']['US'][-1]:.4f}, "
          f"AS={result['densities']['AS'][-1]:.4f}, AI={result['densities']['AI'][-1]:.4f}")

    # 运行多次模拟
    print("\n运行多次模拟...")
    mc_results = simulator.run_multiple_simulations(
        n_simulations=10,
        T=8.0,
        record_interval=0.04,
        initial_infected=0.02,
        initial_aware=0.03
    )

    print(f"平均最终状态: US={mc_results['mean_densities']['US'][-1]:.4f}, "
          f"AS={mc_results['mean_densities']['AS'][-1]:.4f}, AI={mc_results['mean_densities']['AI'][-1]:.4f}")

    return mc_results


if __name__ == "__main__":
    results = test_fixed_monte_carlo()