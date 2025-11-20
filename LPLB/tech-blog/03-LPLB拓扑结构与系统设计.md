# LPLB拓扑结构与系统设计：从Cube到Torus的负载均衡优化

在前两篇文章中，我们介绍了LPLB的基本概念和线性规划算法实现。现在让我们深入探讨LPLB的拓扑结构设计和系统架构，这些是LPLB实现高性能负载均衡的关键技术。

## 拓扑结构的重要性

在MoE模型的专家并行训练中，拓扑结构决定了专家之间的连接方式和通信模式。一个好的拓扑结构应该：

1. **最小化通信开销**：尽量将频繁通信的专家放在同一节点或高速连接的GPU上
2. **最大化负载均衡能力**：提供足够的冗余路径来分散负载
3. **适应硬件约束**：考虑GPU数量、内存限制、网络带宽等实际约束
4. **支持扩展性**：能够适应不同规模的模型和硬件配置

## LPLB支持的三种主要拓扑

### 1. Cube拓扑（立方体拓扑）

#### 基本概念

Cube拓扑是LPLB最常用的拓扑结构，适用于8-GPU的专家并行组。它形成一个三维立方体，每个顶点代表一个GPU，边代表GPU之间的连接。

#### 数学表示

```
Cube拓扑可以表示为图G = (V, E)，其中：
- V = {0, 1, 2, 3, 4, 5, 6, 7}：8个GPU
- E = {(i,j) | i XOR j = 1, 2, 或 4}：12条边

连接关系：
GPU0 ↔ GPU1, GPU2, GPU4
GPU1 ↔ GPU0, GPU3, GPU5
GPU2 ↔ GPU0, GPU3, GPU6
GPU3 ↔ GPU1, GPU2, GPU7
GPU4 ↔ GPU0, GPU5, GPU6
GPU5 ↔ GPU1, GPU4, GPU7
GPU6 ↔ GPU2, GPU4, GPU7
GPU7 ↔ GPU3, GPU5, GPU6
```

#### 实现代码

```python
import torch
import numpy as np

class CubeTopology:
    def __init__(self, n_gpus=8):
        self.n_gpus = n_gpus
        self.n_dim = 3  # 三维立方体

        # 验证GPU数量
        if n_gpus != 8:
            raise ValueError("Cube拓扑需要8个GPU")

        # 创建邻接矩阵
        self.adjacency_matrix = self._create_adjacency_matrix()

        # 创建GPU坐标映射
        self.gpu_coordinates = self._create_coordinate_mapping()

    def _create_adjacency_matrix(self):
        """创建立方体拓扑的邻接矩阵"""
        adj_matrix = torch.zeros(8, 8, dtype=torch.float32)

        for i in range(8):
            for j in range(8):
                if i != j and self._are_connected(i, j):
                    adj_matrix[i][j] = 1.0

        return adj_matrix

    def _are_connected(self, gpu1, gpu2):
        """判断两个GPU是否直接连接"""
        # 在立方体中，两个GPU连接当且仅当它们的二进制表示只有一位不同
        diff = gpu1 ^ gpu2
        return diff != 0 and (diff & (diff - 1)) == 0

    def _create_coordinate_mapping(self):
        """创建GPU到3D坐标的映射"""
        coordinates = {}
        for gpu in range(8):
            x = (gpu >> 0) & 1
            y = (gpu >> 1) & 1
            z = (gpu >> 2) & 1
            coordinates[gpu] = (x, y, z)

        return coordinates

    def get_neighbors(self, gpu_id):
        """获取指定GPU的邻居"""
        neighbors = []
        for j in range(8):
            if self.adjacency_matrix[gpu_id][j] > 0:
                neighbors.append(j)
        return neighbors

    def get_path_cost(self, source, destination):
        """计算两个GPU之间的最短路径成本"""
        if source == destination:
            return 0

        # 使用BFS计算最短路径
        from collections import deque

        queue = deque([(source, 0)])
        visited = {source}

        while queue:
            current, distance = queue.popleft()

            for neighbor in self.get_neighbors(current):
                if neighbor == destination:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # 不可达

# 使用示例
def demonstrate_cube_topology():
    """演示Cube拓扑的使用"""

    cube = CubeTopology()

    print("Cube拓扑邻接矩阵:")
    print(cube.adjacency_matrix.numpy())

    print("\n各GPU的邻居关系:")
    for gpu in range(8):
        neighbors = cube.get_neighbors(gpu)
        print(f"GPU{gpu}: 邻居 = {neighbors}")

    print("\n路径成本示例:")
    examples = [(0, 7), (1, 6), (3, 4)]
    for src, dst in examples:
        cost = cube.get_path_cost(src, dst)
        print(f"GPU{src} → GPU{dst}: 路径成本 = {cost}")

demonstrate_cube_topology()
```

#### Cube拓扑的优势

1. **高度对称性**：每个GPU都有相同的连接数（3个邻居）
2. **短直径**：任意两个GPU之间的最大距离为3
3. **良好的扩展性**：容易理解和实现
4. **适合8-GPU系统**：与常见的GPU服务器配置匹配

#### 实际配置示例

```python
def setup_cube_ep_configuration():
    """设置Cube拓扑的专家并行配置"""

    # 专家分配
    expert_assignments = {
        0: [0, 8],   # GPU0处理专家0和8
        1: [1, 9],   # GPU1处理专家1和9
        2: [2, 10],  # GPU2处理专家2和10
        3: [3, 11],  # GPU3处理专家3和11
        4: [4, 12],  # GPU4处理专家4和12
        5: [5, 13],  # GPU5处理专家5和13
        6: [6, 14],  # GPU6处理专家6和14
        7: [7, 15],  # GPU7处理专家7和15
    }

    # 冗余专家连接（基于Cube拓扑）
    redundant_connections = [
        (0, 1, 100),  # GPU0到GPU1可以重分配100个令牌
        (0, 2, 100),  # GPU0到GPU2可以重分配100个令牌
        (0, 4, 100),  # GPU0到GPU4可以重分配100个令牌
        (1, 0, 100),  # 双向连接
        (1, 3, 100),
        (1, 5, 100),
        # ... 其他连接
    ]

    return expert_assignments, redundant_connections

expert_assignments, connections = setup_cube_ep_configuration()
print("专家分配:", expert_assignments)
print("冗余连接:", connections[:6])  # 显示前6个连接
```

### 2. Hypercube拓扑（超立方体拓扑）

#### 基本概念

Hypercube是Cube拓扑的推广，适用于16-GPU的专家并行组。它形成一个四维超立方体，具有更多的连接和更好的扩展性。

#### 数学表示

```
Hypercube拓扑可以表示为图G = (V, E)，其中：
- V = {0, 1, ..., 15}：16个GPU
- E = {(i,j) | i XOR j = 2^k, for some k}：32条边

连接关系：
每个GPU有4个邻居，对应于二进制表示中的一位差异
```

#### 实现代码

```python
class HypercubeTopology:
    def __init__(self, n_gpus=16):
        self.n_gpus = n_gpus
        self.n_dim = 4  # 四维超立方体

        if n_gpus != 16:
            raise ValueError("Hypercube拓扑需要16个GPU")

        self.adjacency_matrix = self._create_adjacency_matrix()
        self.gpu_coordinates = self._create_coordinate_mapping()

    def _create_adjacency_matrix(self):
        """创建超立方体拓扑的邻接矩阵"""
        adj_matrix = torch.zeros(16, 16, dtype=torch.float32)

        for i in range(16):
            for j in range(16):
                if i != j and self._are_connected(i, j):
                    adj_matrix[i][j] = 1.0

        return adj_matrix

    def _are_connected(self, gpu1, gpu2):
        """判断两个GPU是否直接连接"""
        diff = gpu1 ^ gpu2
        # 检查是否是2的幂次方（只有一位不同）
        return diff != 0 and (diff & (diff - 1)) == 0

    def _create_coordinate_mapping(self):
        """创建GPU到4D坐标的映射"""
        coordinates = {}
        for gpu in range(16):
            x = (gpu >> 0) & 1
            y = (gpu >> 1) & 1
            z = (gpu >> 2) & 1
            w = (gpu >> 3) & 1
            coordinates[gpu] = (x, y, z, w)

        return coordinates

# 使用示例
def compare_cube_hypercube():
    """比较Cube和Hypercube拓扑的特性"""

    cube = CubeTopology()
    hypercube = HypercubeTopology()

    print("拓扑特性对比:")
    print("-" * 50)
    print(f"{'特性':<20} {'Cube(8GPU)':<15} {'Hypercube(16GPU)':<15}")
    print("-" * 50)
    print(f"{'GPU数量':<20} {8:<15} {16:<15}")
    print(f"{'维度':<20} {3:<15} {4:<15}")
    print(f"{'每个GPU的邻居数':<20} {3:<15} {4:<15}")
    print(f"{'图的直径':<20} {3:<15} {4:<15}")
    print(f"{'总边数':<20} {12:<15} {32:<15}")

compare_cube_hypercube()
```

### 3. Torus拓扑（环形拓扑）

#### 基本概念

Torus拓扑是一种环形结构，特别适合多节点部署。它将专家组织成环形，每个GPU连接到相邻的两个GPU。

#### 实现代码

```python
class TorusTopology:
    def __init__(self, n_gpus=8, nodes=2):
        self.n_gpus = n_gpus
        self.nodes = nodes
        self.gpus_per_node = n_gpus // nodes

        # 验证配置
        if n_gpus % nodes != 0:
            raise ValueError("GPU数量必须能被节点数整除")

        self.adjacency_matrix = self._create_adjacency_matrix()

    def _create_adjacency_matrix(self):
        """创建Torus拓扑的邻接矩阵"""
        adj_matrix = torch.zeros(self.n_gpus, self.n_gpus, dtype=torch.float32)

        for i in range(self.n_gpus):
            # 同一节点内的环形连接
            next_gpu = (i // self.gpus_per_node) * self.gpus_per_node + \
                      ((i % self.gpus_per_node) + 1) % self.gpus_per_node
            prev_gpu = (i // self.gpus_per_node) * self.gpus_per_node + \
                      ((i % self.gpus_per_node) - 1) % self.gpus_per_node

            adj_matrix[i][next_gpu] = 1.0
            adj_matrix[i][prev_gpu] = 1.0

            # 跨节点的连接（可选）
            other_node = (i // self.gpus_per_node + 1) % self.nodes
            cross_gpu = other_node * self.gpus_per_node + (i % self.gpus_per_node)
            adj_matrix[i][cross_gpu] = 0.5  # 跨节点连接权重较低

        return adj_matrix

def demonstrate_torus_topology():
    """演示Torus拓扑的使用"""

    torus = TorusTopology(n_gpus=8, nodes=2)

    print("Torus拓扑 (2节点，每节点4GPU):")
    print("节点内连接:")

    for node in range(2):
        start_gpu = node * 4
        end_gpu = (node + 1) * 4

        print(f"  节点{node} (GPU{start_gpu}-{end_gpu-1}):")
        for gpu in range(start_gpu, end_gpu):
            neighbors = [i for i in range(8) if torus.adjacency_matrix[gpu][i] > 0]
            print(f"    GPU{gpu}: {neighbors}")

demonstrate_torus_topology()
```

## LPLB系统架构设计

### 整体架构

```python
class LPLBSystem:
    def __init__(self, topology_type="cube", n_gpus=8, n_experts=16):
        """
        LPLB系统初始化

        参数：
        - topology_type: 拓扑类型 ("cube", "hypercube", "torus")
        - n_gpus: GPU数量
        - n_experts: 专家数量
        """
        self.topology_type = topology_type
        self.n_gpus = n_gpus
        self.n_experts = n_experts

        # 创建拓扑结构
        self.topology = self._create_topology()

        # 创建负载均衡器
        self.load_balancer = self._create_load_balancer()

        # 初始化统计信息
        self.statistics = {
            "total_tokens": 0,
            "load_imbalance_history": [],
            "communication_overhead": 0
        }

    def _create_topology(self):
        """根据配置创建拓扑结构"""
        if self.topology_type == "cube":
            return CubeTopology()
        elif self.topology_type == "hypercube":
            return HypercubeTopology()
        elif self.topology_type == "torus":
            return TorusTopology(nodes=self.n_gpus // 4)
        else:
            raise ValueError(f"不支持的拓扑类型: {self.topology_type}")

    def _create_load_balancer(self):
        """创建负载均衡器"""
        from lplb import Planner

        return Planner(
            redundant_to_original=self._create_redundant_mapping(),
            n_routed_experts=self.n_experts + self.n_experts // 2,  # 50%冗余
            n_logical_routed_experts=self.n_experts
        )

    def _create_redundant_mapping(self):
        """创建冗余专家映射"""
        n_redundant = self.n_experts // 2
        mapping = torch.zeros(self.n_experts, n_redundant, dtype=torch.float32)

        # 基于拓扑结构创建映射
        for i in range(self.n_experts):
            for j in range(n_redundant):
                # 简化策略：均匀分布冗余专家
                mapping[i][j] = 1.0

        return mapping

    def process_batch(self, expert_indices, token_counts):
        """
        处理一个batch的数据

        参数：
        - expert_indices: 专家索引 [batch_size, top_k]
        - token_counts: 每个专家处理的令牌数量 [batch_size, top_k]
        """
        # 更新统计信息
        self.statistics["total_tokens"] += token_counts.sum().item()

        # 执行负载均衡
        balanced_indices = self.load_balancer.run(expert_indices)

        # 计算负载均衡指标
        load_imbalance = self._calculate_load_imbalance(balanced_indices)
        self.statistics["load_imbalance_history"].append(load_imbalance)

        return balanced_indices

    def _calculate_load_imbalance(self, expert_indices):
        """计算负载不均衡指标"""
        # 统计每个专家的令牌数量
        expert_counts = torch.zeros(self.n_experts)
        for batch_idx in range(expert_indices.size(0)):
            for top_k_idx in range(expert_indices.size(1)):
                expert_id = expert_indices[batch_idx, top_k_idx].item()
                if expert_id < self.n_experts:
                    expert_counts[expert_id] += 1

        # 计算标准差作为不均衡指标
        return torch.std(expert_counts).item()

# 使用示例
def demonstrate_lplb_system():
    """演示LPLB系统的使用"""

    # 创建LPLB系统
    lplb_system = LPLBSystem(topology_type="cube", n_gpus=8, n_experts=16)

    print(f"LPLB系统配置:")
    print(f"  拓扑类型: {lplb_system.topology_type}")
    print(f"  GPU数量: {lplb_system.n_gpus}")
    print(f"  专家数量: {lplb_system.n_experts}")

    # 模拟处理多个batch
    for batch_idx in range(5):
        batch_size = 256
        top_k = 2

        # 生成模拟的专家选择和令牌数量
        expert_indices = torch.randint(0, 16, (batch_size, top_k))
        token_counts = torch.randint(1, 10, (batch_size, top_k))

        # 处理batch
        balanced_indices = lplb_system.process_batch(expert_indices, token_counts)

        load_imbalance = lplb_system.statistics["load_imbalance_history"][-1]
        print(f"  Batch {batch_idx + 1}: 负载不均衡指标 = {load_imbalance:.3f}")

demonstrate_lplb_system()
```

## 拓扑结构性能分析

### 性能指标

```python
def analyze_topology_performance():
    """分析不同拓扑结构的性能"""

    topologies = ["cube", "hypercube", "torus"]
    metrics = {}

    for topo in topologies:
        print(f"\n分析 {topo} 拓扑性能:")

        try:
            if topo == "cube":
                topology = CubeTopology()
                n_gpus = 8
            elif topo == "hypercube":
                topology = HypercubeTopology()
                n_gpus = 16
            else:
                topology = TorusTopology()
                n_gpus = 8

            # 计算拓扑指标
            avg_degree = torch.sum(topology.adjacency_matrix) / n_gpus
            diameter = max([topology.get_path_cost(i, j) for i in range(n_gpus) for j in range(n_gpus)])

            # 计算连通性
            connectivity = calculate_connectivity(topology.adjacency_matrix)

            # 估算通信开销
            comm_overhead = estimate_communication_overhead(topology)

            metrics[topo] = {
                "avg_degree": avg_degree.item(),
                "diameter": diameter,
                "connectivity": connectivity,
                "comm_overhead": comm_overhead
            }

            print(f"  平均连接数: {avg_degree.item():.2f}")
            print(f"  图直径: {diameter}")
            print(f"  连通性: {connectivity:.3f}")
            print(f"  通信开销估计: {comm_overhead:.2f}")

        except Exception as e:
            print(f"  错误: {e}")

    return metrics

def calculate_connectivity(adj_matrix):
    """计算图的连通性"""
    n = adj_matrix.size(0)

    # 使用Floyd-Warshall算法计算最短路径
    dist = adj_matrix.clone()
    dist[dist == 0] = float('inf')
    dist[torch.eye(n, dtype=torch.bool)] = 0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # 连通性：有限路径的比例
    finite_paths = torch.isfinite(dist).float().mean()
    return finite_paths.item()

def estimate_communication_overhead(topology):
    """估算通信开销"""
    # 简化模型：基于图的直径和平均连接数
    diameter = max([topology.get_path_cost(i, j) for i in range(topology.n_gpus) for j in range(topology.n_gpus)])
    avg_connections = torch.sum(topology.adjacency_matrix) / topology.n_gpus

    # 通信开销与直径成正比，与平均连接数成反比
    overhead = diameter / (avg_connections.item() + 1)
    return overhead

# 运行性能分析
topology_metrics = analyze_topology_performance()
```

### 实际部署建议

```python
def recommend_topology(n_gpus, n_nodes, workload_characteristics):
    """
    根据硬件配置和工作负载特征推荐拓扑结构

    参数：
    - n_gpus: 总GPU数量
    - n_nodes: 节点数量
    - workload_characteristics: 工作负载特征字典
    """

    print(f"\n拓扑配置推荐:")
    print(f"硬件配置: {n_gpus} GPU, {n_nodes} 节点")

    recommendations = []

    if n_gpus == 8 and n_nodes == 1:
        # 单节点8 GPU
        if workload_characteristics.get("high_communication", False):
            recommendations.append({
                "topology": "cube",
                "reason": "8 GPU单节点，Cube拓扑提供最优的通信性能"
            })
        else:
            recommendations.append({
                "topology": "cube",
                "reason": "标准8 GPU配置，Cube拓扑是默认选择"
            })

    elif n_gpus == 16 and n_nodes <= 2:
        # 16 GPU配置
        if n_nodes == 1:
            recommendations.append({
                "topology": "hypercube",
                "reason": "16 GPU单节点，Hypercube提供最佳连接性"
            })
        else:
            recommendations.append({
                "topology": "torus",
                "reason": "多节点16 GPU，Torus拓扑适合跨节点部署"
            })

    elif n_gpus > 16:
        # 大规模配置
        recommendations.append({
            "topology": "torus",
            "reason": "大规模部署，Torus拓扑提供最好的扩展性"
        })

    else:
        # 其他配置
        recommendations.append({
            "topology": "cube",
            "reason": "默认推荐，可以根据具体需求调整"
        })

    # 输出推荐结果
    for i, rec in enumerate(recommendations, 1):
        print(f"  推荐{i}: {rec['topology']}")
        print(f"    理由: {rec['reason']}")

    return recommendations

# 使用示例
config_recommendations = recommend_topology(
    n_gpus=8,
    n_nodes=1,
    workload_characteristics={
        "high_communication": True,
        "memory_intensive": False,
        "latency_sensitive": True
    }
)
```

## 总结

LPLB的拓扑结构设计体现了以下关键优势：

1. **灵活性**：支持多种拓扑结构，适应不同硬件配置
2. **高效性**：每种拓扑都经过优化，最小化通信开销
3. **可扩展性**：从小规模8-GPU到大规模多节点部署
4. **实用性**：与实际的硬件架构和MoE模型需求相匹配

通过合理选择拓扑结构，LPLB能够：

- **Cube拓扑**：为标准8-GPU服务器提供最优性能
- **Hypercube拓扑**：为16-GPU系统提供增强的连接性
- **Torus拓扑**：为多节点大规模部署提供良好的扩展性

在下一篇文章中，我们将探讨LPLB的实战应用，包括具体的部署案例、性能调优技巧，以及与Deep-EP框架的深度集成。

---

*下一篇：[LPLB实战应用与性能调优](./04-LPLB实战应用与性能调优.md)*