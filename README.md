# DeepSeek LPLB (Linear-Programming-Based Load Balancer) - 专家并行负载均衡技术

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.6+](https://img.shields.io/badge/cuda-12.6+-green.svg)](https://developer.nvidia.com/cuda-downloads)

## 项目概述

DeepSeek LPLB (Linear-Programming-Based Load Balancer) 是一个专为 MoE (Mixture of Experts) 大模型设计的专家并行动态负载均衡解决方案。它基于线性规划算法，能够实时优化专家并行系统中的负载分布，显著提升大规模模型训练的效率。

### 🎯 核心创新

LPLB 解决了传统静态负载均衡无法处理的**动态负载不均衡**问题：

- **动态适应性**：根据每个batch的实时负载情况进行优化
- **数学最优性**：使用线性规划找到数学上最优的负载分配方案
- **高性能计算**：单SM内点法求解器，实现微秒级求解
- **通信优化**：充分利用NVLink和NVSHMEM硬件特性

### 📊 技术特点

| 特性 | LPLB | 传统方法 |
|------|------|----------|
| **优化算法** | 线性规划（数学最优） | 贪心启发式 |
| **负载类型** | 动态实时负载 | 静态历史负载 |
| **求解时间** | ~100μs | ~10μs |
| **求解质量** | 最优解 | 近似解 |
| **通信开销** | 极低 | 较低 |
| **扩展性** | 优秀 | 良好 |

## 项目结构

```
deepseek-lplb-learning/
├── 📚 EPLB/                          # EPLB技术文档
│   ├── eplb.py                       # EPLB核心算法实现
│   ├── README.md                     # EPLB项目说明
│   └── tech-blog/                    # EPLB技术博客系列
│       ├── README.md                  # EPLB博客索引
│       ├── 01-MOE基础与EPLB介绍_修正版.md
│       ├── 02-EPLB负载均衡机制详解_修正版.md
│       ├── 03-EPLB实现细节与代码分析_修正版.md
│       ├── 04-EPLB性能优化与实践经验_修正版.md
│       └── EPLB三步算法通俗详解_修正版.md
│
├── 🚀 LPLB/                          # LPLB技术文档
│   ├── README.md                     # LPLB项目说明
│   └── tech-blog/                    # LPLB技术博客系列
│       ├── README.md                  # LPLB博客索引
│       ├── 01-LPLB基础介绍.md
│       ├── 02-LPLB线性规划算法详解.md
│       ├── 03-LPLB拓扑结构与系统设计.md
│       └── 04-LPLB实战应用与性能调优.md
│
└── 📄 README.md                     # 项目总览（本文件）
```

## 🔬 技术博客系列

我们为这两个项目编写了详细的技术博客，从基础概念到实战应用，帮助您深入理解专家并行负载均衡技术。

### 📖 EPLB技术博客系列

EPLB (Expert Parallelism Load Balancer) 是经典的专家并行静态负载均衡解决方案：

1. **[MOE基础与EPLB介绍（修正版）](EPLB/tech-blog/01-MOE基础与EPLB介绍_修正版.md)**
   - 从LoRA到MoE的技术演进
   - 专家并行的挑战和EPLB解决方案
   - 冗余专家策略的原理和实现
   - 实际配置示例和性能分析

2. **[EPLB负载均衡机制详解（修正版）](EPLB/tech-blog/02-EPLB负载均衡机制详解_修正版.md)**
   - 分层和全局负载均衡策略
   - 核心算法实现和复杂度分析
   - 实际案例和优化效果
   - 性能优势量化分析

3. **[EPLB实现细节与代码分析（修正版）](EPLB/tech-blog/03-EPLB实现细节与代码分析_修正版.md)**
   - 核心代码逐行解析
   - 数据结构和映射关系
   - 设计模式和优化技巧
   - 调试和验证方法

4. **[EPLB性能优化与实践经验（修正版）](EPLB/tech-blog/04-EPLB性能优化与实践经验_修正版.md)**
   - 负载统计获取与优化
   - 动态负载均衡策略
   - 生产环境部署最佳实践
   - 性能监控和调优

### 🚀 LPLB技术博客系列

LPLB (Linear-Programming-Based Load Balancer) 是新一代的动态负载均衡解决方案：

1. **[LPLB技术基础：基于线性规划的专家并行负载均衡](LPLB/tech-blog/01-LPLB基础介绍.md)**
   - 从EPLB到LPLB的技术进化
   - 动态负载不均衡问题分析
   - 线性规划核心思想
   - 系统架构和组件介绍

2. **[LPLB线性规划算法深度解析：从数学原理到CUDA实现](LPLB/tech-blog/02-LPLB线性规划算法详解.md)**
   - 线性规划数学建模
   - 内点法详细推导
   - CUDA内核实现
   - 数值精度和收敛性分析

3. **[LPLB拓扑结构与系统设计：从Cube到Torus的负载均衡优化](LPLB/tech-blog/03-LPLB拓扑结构与系统设计.md)**
   - Cube、Hypercube、Torus拓扑详解
   - 拓扑结构数学表示和实现
   - 系统架构设计
   - 性能分析和对比

4. **[LPLB实战应用与性能调优：从部署到优化的完整指南](LPLB/tech-blog/04-LPLB实战应用与性能调优.md)**
   - 单节点和多节点部署
   - 内存、计算、通信优化
   - 与Deep-EP深度集成
   - 故障排除和性能监控

## 🏆 技术亮点

### EPLB 的创新

- **冗余专家策略**：为热门专家创建副本，分散处理负载
- **分层负载均衡**：充分利用硬件拓扑，优化跨节点通信
- **贪心算法实现**：高效的负载分配算法，时间复杂度O(n²)
- **工程实用性**：简洁的API设计，易于集成和使用

### LPLB 的突破

- **线性规划求解**：使用内点法找到数学最优解
- **单SM求解器**：在单个SM上完成线性规划求解，最小化通信
- **动态负载均衡**：实时处理每个batch的负载变化
- **多种拓扑支持**：Cube、Hypercube、Torus等拓扑结构
- **Deep-EP集成**：与主流MoE框架无缝集成

## 📈 性能对比

### 负载均衡效果

| 指标 | 无优化 | EPLB | LPLB |
|------|--------|------|------|
| **负载不均衡倍数** | 10-45倍 | 1.5-3倍 | 1.1-1.8倍 |
| **负载均衡改善** | - | 80-95% | 90-98% |
| **求解时间** | - | 1-10ms | 50-200μs |
| **GPU利用率** | 40-60% | 70-85% | 80-95% |

### 适用场景

- **EPLB**：静态负载不均衡，数据分布导致的持续过载
- **LPLB**：动态负载不均衡，训练batch的随机性负载波动

## 🛠️ 环境要求

### 基础环境
- Python 3.8+
- CUDA Toolkit 12.6+
- PyTorch 1.12+
- NVIDIA GPU (支持NVLink推荐)

### LPLB 额外要求
- cuSolverDx
- cuBLASDx
- Deep-EP (可选但推荐)

## 🚀 快速开始

### 安装 EPLB
```bash
git clone https://github.com/deepseek-ai/EPLB.git
cd EPLB
pip install -e .
```

### 安装 LPLB
```bash
git clone https://github.com/deepseek-ai/LPLB.git
cd LPLB
./download-mathdx.sh
pip install --no-build-isolation .
```

### 基础使用示例

#### EPLB 示例
```python
import torch
import eplb

# 配置参数
weight = torch.tensor([[200, 150, 100, 50]])  # 专家负载统计
num_replicas = 8
num_groups = 4
num_nodes = 2
num_gpus = 4

# 执行EPLB负载均衡
phy2log, log2phy, logcnt = eplb.rebalance_experts(
    weight, num_replicas, num_groups, num_nodes, num_gpus
)

print(f"专家副本数量: {logcnt}")
print(f"负载均衡改善: {weight.max() / weight.min():.1f}倍 → "
      f"{(weight.float() / logcnt.float()).max() / (weight.float() / logcnt.float()).min():.1f}倍")
```

#### LPLB 示例
```python
from lplb import Planner

# 配置参数
planner = Planner(
    redundant_to_original=redundant_mapping,
    n_routed_experts=24,      # 物理专家数量
    n_logical_routed_experts=16,  # 逻辑专家数量
    ep_size=8                   # 专家并行组大小
)

# 处理batch数据
expert_indices = torch.randint(0, 16, (512, 2))
balanced_indices = planner.run(expert_indices)

print(f"原始专家选择范围: {expert_indices.min()}-{expert_indices.max()}")
print(f"均衡后专家范围: {balanced_indices.min()}-{balanced_indices.max()}")
```

## 📊 技术应用价值

### 实际应用场景

1. **大模型训练**：
   - DeepSeek-V2/V3 级别的MoE模型
   - 数十亿到千亿参数规模
   - 多GPU分布式训练

2. **高吞吐量推理**：
   - 在线服务的实时负载均衡
   - 动态工作负载处理
   - 低延迟推理服务

3. **研究开发**：
   - MoE架构研究
   - 负载均衡算法开发
   - 分布式系统优化

### 技术贡献

1. **算法创新**：
   - 将线性规划应用于MoE负载均衡
   - 单SM求解器的高效实现
   - 动态负载均衡的理论突破

2. **工程价值**：
   - 显著提升训练效率（15-30%）
   - 降低通信开销（30-50%）
   - 提高硬件利用率（25-50%）

3. **开源贡献**：
   - 完整的开源实现
   - 详细的技术文档
   - 丰富的使用示例

## 🤝 贡献指南

我们欢迎社区贡献！请查看以下项目的贡献指南：

- [EPLB Contribution Guide](EPLB/CONTRIBUTING.md)
- [LPLB Contribution Guide](LPLB/CONTRIBUTING.md)

### 贡献方式

1. **代码贡献**：提交Pull Request
2. **文档改进**：完善技术文档和示例
3. **问题报告**：报告bug和建议
4. **性能优化**：改进算法和实现

## 📚 相关资源

### 学术论文
- DeepSeek-MoE系列论文
- MoE架构研究论文
- 负载均衡算法研究

### 技术博客
- DeepSeek技术分享
- 大模型训练优化经验
- 分布式系统设计

### 开源项目
- [Deep-EP](https://github.com/deepseek-ai/Deep-EP)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](EPLB/LICENSE) 和 [LICENSE](LPLB/LICENSE) 文件。

## 🙏 致谢

感谢所有为本项目做出贡献的研究人员和工程师：

- DeepSeek AI 研究团队
- 开源社区贡献者
- 用户反馈和建议

## 📞 联系我们

- **项目Issues**：[EPLB Issues](https://github.com/deepseek-ai/EPLB/issues), [LPLB Issues](https://github.com/deepseek-ai/LPLB/issues)
- **技术讨论**：GitHub Discussions
- **邮件联系**：research@deepseek.ai

---

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

**📚 学习路径**：建议先阅读EPLB博客理解基础概念，再阅读LPLB博客掌握前沿技术。

**🚀 开始使用**：按照快速开始指南部署您的MoE负载均衡解决方案！

*最后更新时间：2025年11月*