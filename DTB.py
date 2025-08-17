from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

# 1. 定义网络结构
model = BayesianNetwork([('AB', 'C')])

# 2. 定义AB节点的联合概率 (4种状态)
# 假设先验概率分布 (需根据实际数据调整)
ab_cpd = TabularCPD(
    variable='AB',
    variable_card=4,  # 4种状态
    values=[[0.3], [0.2], [0.1], [0.4]],  # P(AB=00),P(AB=01),P(AB=10),P(AB=11)
    state_names={'AB': ['00', '01', '10', '11']}
)

# 3. 定义C的条件概率表
# 给定AB的联合状态时C的概率
c_cpd = TabularCPD(
    variable='C',
    variable_card=2,  # 二值变量
    values=[
        # C=0 的概率 | AB=00,01,10,11
        [0.9, 0.7, 0.4, 0.1],

        # C=1 的概率 | AB=00,01,10,11
        [0.1, 0.3, 0.6, 0.9]
    ],
    evidence=['AB'],
    evidence_card=[4],
    state_names={
        'C': ['False', 'True'],
        'AB': ['00', '01', '10', '11']
    }
)

# 4. 添加概率表到模型
model.add_cpds(ab_cpd, c_cpd)

# 5. 验证模型
assert model.check_model()

# 6. 推理示例
from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

# 示例查询1：当C发生时，A的概率
result = infer.query(variables=['AB'], evidence={'C': 'True'})
print("\n当C为真时AB的联合分布:")
print(result)

# 示例查询2：当A发生时，C的概率 (需提取A=1的状态)
# P(C|A=1) = P(C|AB=10) + P(C|AB=11)
prob_c_given_a1 = np.dot(
    [c_cpd.values[1, 2], c_cpd.values[1, 3]],  # AB=10和11时C=1的概率
    [ab_cpd.values[2], ab_cpd.values[3]]  # AB=10和11的权重
) / (ab_cpd.values[2] + ab_cpd.values[3])
print(f"\n当A发生时C为真的概率: {prob_c_given_a1:.3f}")

# 示例查询3：B对C的影响 (需提取B=1的状态)
# P(C|B=1) = P(C|AB=01) + P(C|AB=11)
prob_c_given_b1 = np.dot(
    [c_cpd.values[1, 1], c_cpd.values[1, 3]],  # AB=01和11时C=1的概率
    [ab_cpd.values[1], ab_cpd.values[3]]  # AB=01和11的权重
) / (ab_cpd.values[1] + ab_cpd.values[3])
print(f"当B发生时C为真的概率: {prob_c_given_b1:.3f}")