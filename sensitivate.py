from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建 DBN 模型
dbn_model = DBN()

# 定义两个时间片的节点：A_t0, B_t0 和 A_t1, B_t1
time_slices = [(0, 1)]  # 时间片 0 和 1
nodes = ['A', 'B']

# 添加边：A_t0 -> A_t1, B_t0 -> B_t1
for t in time_slices:
    for node in nodes:
        dbn_model.add_edge((node, t[0]), (node, t[1]))

# 2. 定义初始状态的 CPD
cpd_a_t0 = TabularCPD(
    variable=('A', 0),
    variable_card=2,
    values=[[0.6], [0.4]],  # P(A=0) = 0.6, P(A=1) = 0.4
    state_names={'A': [0, 1]}
)

cpd_b_t0 = TabularCPD(
    variable=('B', 0),
    variable_card=2,
    values=[[0.7], [0.3]],  # P(B=0) = 0.7, P(B=1) = 0.3
    state_names={'B': [0, 1]}
)

# 3. 定义转移概率（时间片 0 → 1）
cpd_a_transition = TabularCPD(
    variable=('A', 1),
    variable_card=2,
    values=[[0.9, 0.1],  # P(A_t1=0 | A_t0=0), P(A_t1=0 | A_t0=1)
            [0.1, 0.9]],  # P(A_t1=1 | A_t0=0), P(A_t1=1 | A_t0=1)
    evidence=[('A', 0)],
    evidence_card=[2],
    state_names={'A': [0, 1]}
)

cpd_b_transition = TabularCPD(
    variable=('B', 1),
    variable_card=2,
    values=[[0.8, 0.2],
            [0.2, 0.8]],
    evidence=[('B', 0)],
    evidence_card=[2],
    state_names={'B': [0, 1]}
)

# 4. 将 CPD 添加到模型
dbn_model.add_cpds(cpd_a_t0, cpd_b_t0, cpd_a_transition, cpd_b_transition)

# 5. 初始化推理器
infer = DBNInference(dbn_model)


# 6. 敏感性分析函数：调整某个参数并比较输出变化
def sensitivity_analysis(target_node, parameter_to_change, base_value, perturbation_range):
    results = []
    for delta in perturbation_range:
        # 复制原始模型
        modified_dbn = DBN()
        modified_dbn.add_edges_from(dbn_model.edges())
        modified_dbn.add_cpds(*dbn_model.get_cpds())

        # 修改指定参数
        for cpd in modified_dbn.get_cpds():
            if cpd.variable == parameter_to_change:
                if cpd.variable_card == 2:  # 假设是二值变量
                    values = cpd.values.copy()
                    values[0] = max(0, min(1, values[0] + delta))  # 调整第一个值
                    values[1] = 1 - values[0]  # 保持概率和为 1
                    cpd.values = values
                    break

        # 重新推理
        query = infer.query(modified_dbn, variables=[target_node])
        posterior = query.values[1]  # 假设只关注状态 1 的概率
        results.append(posterior)

    return results


# 7. 执行敏感性分析：调整 A_t0 的初始概率，观察 B_t1 的变化
target_node = ('B', 1)
parameter_to_change = ('A', 0)
base_value = 0.6
perturbation_range = np.linspace(-0.2, 0.2, 21)  # 在 ±20% 范围内扰动

results = sensitivity_analysis(target_node, parameter_to_change, base_value, perturbation_range)

# 8. 可视化结果
plt.plot(perturbation_range, results, marker='o', linestyle='--')
plt.xlabel('Perturbation of P(A=0)')
plt.ylabel(f'P({target_node} = 1)')
plt.title('Sensitivity Analysis of P(B_t1=1) to P(A_t0=0)')
plt.grid(True)
plt.show()