"""
幻觉分类体系定义

基于对 ground_truth.json 和 replies.json 的分析，将幻觉定义为：
"系统回复中包含与知识库事实不一致、知识库无法支撑、或超出系统实际能力的陈述。"

分类遵循两个维度：
1. 内容维度：编造了什么？（政策、参数、信息、优惠、能力）
2. 严重程度：对用户的影响多大？（Critical / High / Medium / Low）
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Severity(Enum):
    """幻觉严重程度"""
    CRITICAL = "Critical"  # 可能造成安全/健康风险，或严重法律/合规问题
    HIGH = "High"          # 严重误导消费决策，或承诺无法兑现的服务
    MEDIUM = "Medium"      # 事实性错误但不直接涉及安全/金钱
    LOW = "Low"            # 轻微偏差、信息遗漏，影响有限


class HallucinationType(Enum):
    """幻觉类型定义"""
    # --- 编造型：回复声称了知识库中不存在或相矛盾的事实 ---
    POLICY_FABRICATION = (
        "政策编造",
        "回复中的退换货、保修、售后政策与知识库矛盾",
        Severity.HIGH,
    )
    PARAMETER_FABRICATION = (
        "参数编造",
        "回复中的产品参数（材质、规格、接口、版本等）与知识库矛盾",
        Severity.HIGH,
    )
    PROMOTION_FABRICATION = (
        "优惠编造",
        "回复中声称的优惠活动、折扣、优惠券与知识库矛盾",
        Severity.MEDIUM,
    )
    INFORMATION_FABRICATION = (
        "信息编造",
        "回复中声称的事实信息（地址、门店、品牌关系等）知识库无法支撑",
        Severity.HIGH,
    )

    # --- 偏差型：部分正确但有关键错误 ---
    POLICY_DEVIATION = (
        "政策偏差",
        "回复中的政策描述部分正确但存在关键数据/流程偏差",
        Severity.MEDIUM,
    )

    # --- 能力型：声称系统不具备的能力 ---
    CAPABILITY_OVERREACH = (
        "能力越界",
        "回复声称执行了系统实际不具备的操作能力（查物流、改订单等）",
        Severity.HIGH,
    )

    # --- 安全型：忽视安全警告 ---
    SAFETY_MISLEADING = (
        "安全误导",
        "回复忽视或弱化了知识库中的安全警告，可能造成健康/安全风险",
        Severity.CRITICAL,
    )

    # --- 遗漏型：遗漏关键信息 ---
    INFORMATION_OMISSION = (
        "信息遗漏",
        "回复遗漏了知识库中对用户决策重要的信息，导致回复不准确",
        Severity.LOW,
    )

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def description(self) -> str:
        return self.value[1]

    @property
    def default_severity(self) -> Severity:
        return self.value[2]


# 严重程度排序（用于报告排序）
SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


@dataclass
class HallucinationResult:
    """单条检测结果"""
    id: str
    user_question: str
    system_reply: str
    knowledge_base: str
    is_hallucination: bool
    hallucination_type: Optional[str] = None  # HallucinationType.label
    severity: Optional[str] = None             # Severity.value
    detail: str = ""                           # 检测说明
    confidence: float = 1.0                    # 置信度 0-1


@dataclass
class DetectionReport:
    """完整检测报告"""
    results: list[HallucinationResult] = field(default_factory=list)
    total: int = 0
    hallucination_count: int = 0
    non_hallucination_count: int = 0
    type_distribution: dict = field(default_factory=dict)
    severity_distribution: dict = field(default_factory=dict)

    # 与 ground truth 对比指标
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positives: list = field(default_factory=list)
    false_negatives: list = field(default_factory=list)
    type_mismatches: list = field(default_factory=list)
