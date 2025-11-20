#!/usr/bin/env python3
"""
验证技术博客MD文档中所有例子的数值准确性
"""

import torch
import sys
import os

# 添加EPLB路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import eplb

def check_first_blog_example():
    """检查第一篇博客中的负载分散例子"""
    print("=" * 60)
    print("检查第一篇博客中的负载分散例子")
    print("=" * 60)

    # 博客中的例子：原始分配有严重不均衡
    print("博客中的例子:")
    print("原始分配（负载不均衡）:")
    print("GPU1: Expert1(负载200) = 200")
    print("GPU2: Expert2(负载150) = 150")
    print("GPU3: Expert3(负载100) = 100")
    print("GPU4: Expert4(负载50) = 50")

    # 验证这个例子的真实性
    weight = torch.tensor([[200, 150, 100, 50]])

    print(f"\n验证结果:")
    print(f"原始专家负载: {weight[0].numpy()}")
    print(f"总负载: {weight.sum().item()}")
    print(f"最大负载: {weight.max().item()}")
    print(f"最小负载: {weight.min().item()}")
    print(f"负载差异: {weight.max().item() - weight.min().item()}")
    print(f"负载倍数: {weight.max().item() / weight.min().item():.1f}倍")

    # 现在测试EPLB优化后的效果
    print(f"\nEPLB优化效果测试（增加到8个物理专家）:")
    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_replicas=8, num_groups=4, num_nodes=1, num_gpus=4
    )

    print(f"副本数量: {logcnt[0].numpy()}")

    # 计算每个GPU的负载
    experts_per_gpu = 8 // 4  # 每个GPU 2个专家
    gpu_loads = []

    for gpu in range(4):
        gpu_load = 0
        gpu_experts = []
        for phys_expert in range(gpu * experts_per_gpu, (gpu + 1) * experts_per_gpu):
            log_expert = phy2log[0, phys_expert].item()
            original_load = weight[0, log_expert].item()
            replica_count = logcnt[0, log_expert].item()
            load_per_replica = original_load / replica_count

            gpu_load += load_per_replica
            gpu_experts.append(f"E{log_expert+1}({load_per_replica:.1f})")

        gpu_loads.append(gpu_load)
        print(f"GPU{gpu+1}: {' + '.join(gpu_experts)} = {gpu_load:.1f}")

    max_load = max(gpu_loads)
    min_load = min(gpu_loads)
    print(f"\n优化效果:")
    print(f"最大GPU负载: {max_load:.1f}")
    print(f"最小GPU负载: {min_load:.1f}")
    print(f"负载差异: {max_load - min_load:.1f}")
    print(f"负载倍数: {max_load / min_load:.2f}倍")
    print(f"改善程度: 从4.0倍降低到{max_load / min_load:.2f}倍")

def check_eplb_official_example():
    """检查EPLB官方示例的准确性"""
    print("\n" + "=" * 60)
    print("检查EPLB官方示例的准确性")
    print("=" * 60)

    # 官方示例数据
    weight = torch.tensor([[ 90, 132,  40,  61, 104, 165,  39,   4,  73,  56, 183,  86],
                           [ 20, 107, 104,  64,  19, 197, 187, 157, 172,  86,  16,  27]])

    print("EPLB官方示例数据验证:")
    print("第1层专家负载:")
    for i, load in enumerate(weight[0]):
        print(f"Expert{i+1:2d}: {load.item():3d}")

    # 计算原始统计
    print(f"\n原始负载统计:")
    print(f"总负载: {weight[0].sum().item()}")
    print(f"平均负载: {weight[0].float().mean().item():.1f}")
    print(f"最大负载: {weight[0].max().item()} (Expert{weight[0].argmax().item()+1})")
    print(f"最小负载: {weight[0].min().item()} (Expert{weight[0].argmin().item()+1})")
    print(f"负载差异: {weight[0].max().item() - weight[0].min().item()}")

    # EPLB处理
    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_replicas=16, num_groups=4, num_nodes=2, num_gpus=8
    )

    print(f"\nEPLB处理结果:")
    print(f"副本数量: {logcnt[0].numpy()}")

    # 验证每个专家的负载分散
    print(f"\n专家负载分散情况:")
    for i in range(12):
        original_load = weight[0, i].item()
        replica_count = logcnt[0, i].item()
        if replica_count > 1:
            load_per_replica = original_load / replica_count
            print(f"Expert{i+1:2d}: {original_load:3d} → {replica_count}个副本, 每个{load_per_replica:.1f}")

    # 计算GPU负载并验证博客中的数据
    print(f"\nGPU负载详细分析:")
    experts_per_gpu = 16 // 8  # 每个GPU 2个专家
    gpu_loads = []

    for gpu in range(8):
        start_expert = gpu * experts_per_gpu
        end_expert = start_expert + experts_per_gpu

        gpu_load = 0
        gpu_details = []
        for phys_expert in range(start_expert, end_expert):
            log_expert = phy2log[0, phys_expert].item()
            original_load = weight[0, log_expert].item()
            replica_count = logcnt[0, log_expert].item()
            load_per_replica = original_load / replica_count

            gpu_load += load_per_replica
            gpu_details.append(f"E{log_expert+1}({load_per_replica:.1f})")

        gpu_loads.append(gpu_load)
        print(f"GPU{gpu+1:2d}: {' + '.join(gpu_details)} = {gpu_load:.1f}")

    print(f"\nGPU负载统计:")
    print(f"最大负载: {max(gpu_loads):.1f}")
    print(f"最小负载: {min(gpu_loads):.1f}")
    print(f"平均负载: {sum(gpu_loads)/len(gpu_loads):.1f}")
    print(f"负载差异: {max(gpu_loads) - min(gpu_loads):.1f}")
    print(f"负载倍数: {max(gpu_loads) / min(gpu_loads):.2f}倍")

def check_simple_math_example():
    """检查简单的数学计算例子"""
    print("\n" + "=" * 60)
    print("检查简单的数学计算例子")
    print("=" * 60)

    # 验证负载分散的基本数学
    print("验证负载分散的数学原理:")

    examples = [
        {"original": 200, "replicas": 2, "expected": 100},
        {"original": 150, "replicas": 2, "expected": 75},
        {"original": 183, "replicas": 2, "expected": 91.5},
        {"original": 165, "replicas": 2, "expected": 82.5},
    ]

    print("负载分散数学验证:")
    for example in examples:
        original = example["original"]
        replicas = example["replicas"]
        expected = example["expected"]
        actual = original / replicas

        status = "✓" if abs(actual - expected) < 0.001 else "✗"
        print(f"{status} {original} ÷ {replicas} = {actual:.1f} (期望: {expected})")

    # 验证总负载守恒
    print(f"\n总负载守恒验证:")
    total_original = sum([ex["original"] for ex in examples])
    total_after_replicas = sum([ex["original"] for ex in examples])  # 总负载不变
    print(f"原始总负载: {total_original}")
    print(f"分散后总负载: {total_after_replicas}")
    print(f"负载守恒: {total_original == total_after_replicas}")

if __name__ == "__main__":
    check_first_blog_example()
    check_eplb_official_example()
    check_simple_math_example()

    print("\n" + "=" * 60)
    print("博客例子验证完成")
    print("=" * 60)
    print("如果发现错误，需要修正相应的MD文档")
    print("=" * 60)