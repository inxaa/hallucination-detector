"""
幻觉检测引擎

支持两种模式：
- mock: 基于规则 + 结构化对比，不依赖外部 API
- llm:  调用真实 LLM API 做语义判断（需配置 API key）

检测流程：
1. 预检：规则匹配明显的幻觉模式（能力越界等）
2. 内容比对：提取回复中的关键声明，与知识库逐项对比
3. 综合判断：汇总所有不一致，分类并评估严重程度
"""

import json
import re
import os
from typing import Optional

from .classifier import (
    HallucinationResult,
    HallucinationType,
    Severity,
)


class HallucinationDetector:
    """幻觉检测器基类"""

    def detect(self, case: dict) -> HallucinationResult:
        raise NotImplementedError

    def detect_all(self, cases: list[dict]) -> list[HallucinationResult]:
        return [self.detect(c) for c in cases]


# ============================================================
# Mock 检测器：规则 + 结构化关键词对比
# ============================================================

class MockDetector(HallucinationDetector):
    """
    基于规则的 Mock 检测器。

    检测策略：
    1. 能力越界检测：KB 中包含"无（"表示系统不具备对应能力，
       若回复声称已执行该操作 → CAPABILITY_OVERREACH
    2. 政策/参数关键词对比：提取 KB 和回复中的关键数值/事实，对比差异
    3. 安全误导检测：KB 中有安全警告但回复说"放心"→ SAFETY_MISLEADING
    4. 信息遗漏检测：KB 中有重要补充信息但回复未提及
    """

    # 能力越界触发模式：KB 中这些模式表示"系统不具备该能力"
    CAPABILITY_GAP_PATTERNS = [
        r"无[（(].*未接入.*[）)]",
        r"无[（(].*不具备.*[）)]",
        r"无[（(].*需.*人工.*[）)]",
        r"无[（(].*需转人工.*[）)]",
        r"无[（(].*需由.*系统.*[）)]",
    ]

    # 安全警告关键词（KB 中出现 → 回复不能弱化）
    SAFETY_KEYWORDS = [
        "孕妇.*咨询医生", "建议咨询医生", "孕妇.*慎用", "孕妇.*禁用",
        "不建议", "禁止", "风险", "过敏", "副作用",
    ]

    # 回复中弱化安全警告的表达
    SAFETY_DISMISS_PATTERNS = [
        r"可以放心", r"放心使用", r"完全安全", r"没有问题",
        r"所有人都.*用", r"孕妇可以",
    ]

    def detect(self, case: dict) -> HallucinationResult:
        kb = case.get("knowledge_base", "")
        reply = case.get("system_reply", "")
        question = case.get("user_question", "")
        case_id = case.get("id", "unknown")

        # 逐项检测
        findings = []

        # 1. 能力越界检测
        cap_finding = self._check_capability_overreach(case_id, kb, reply)
        if cap_finding:
            findings.append(cap_finding)

        # 2. 事实矛盾检测（参数、政策、信息、优惠）
        fact_findings = self._check_factual_contradictions(case_id, kb, reply, question)
        findings.extend(fact_findings)

        # 3. 安全误导检测
        safety_finding = self._check_safety_misleading(case_id, kb, reply)
        if safety_finding:
            findings.append(safety_finding)

        # 4. 信息遗漏检测（仅在没有其他更严重问题时）
        if not findings:
            omission_finding = self._check_information_omission(case_id, kb, reply)
            if omission_finding:
                findings.append(omission_finding)

        # 汇总结果
        if not findings:
            return HallucinationResult(
                id=case_id,
                user_question=question,
                system_reply=reply,
                knowledge_base=kb,
                is_hallucination=False,
                detail="回复内容与知识库一致，未检测到幻觉。",
                confidence=0.9,
            )

        # 取最严重的 finding 作为主分类
        findings.sort(key=lambda f: self._severity_rank(f["severity"]))
        primary = findings[0]

        # 合并多个 finding 的详情
        detail_parts = []
        for i, f in enumerate(findings):
            detail_parts.append(f"[{f['type']}] {f['detail']}")
        combined_detail = " | ".join(detail_parts)

        return HallucinationResult(
            id=case_id,
            user_question=question,
            system_reply=reply,
            knowledge_base=kb,
            is_hallucination=True,
            hallucination_type=primary["type"],
            severity=primary["severity"],
            detail=combined_detail,
            confidence=primary.get("confidence", 0.85),
        )

    def _severity_rank(self, sev: str) -> int:
        order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        return order.get(sev, 99)

    # ---- 能力越界检测 ----

    def _check_capability_overreach(self, case_id: str, kb: str, reply: str) -> Optional[dict]:
        """检测 KB 标明"无"的能力，回复是否声称已执行"""
        has_gap = False
        gap_descriptions = []

        for pattern in self.CAPABILITY_GAP_PATTERNS:
            matches = re.findall(pattern, kb)
            for m in matches:
                has_gap = True
                gap_descriptions.append(m)

        if not has_gap:
            return None

        # 检查回复是否声称执行了操作（"已帮您"、"查到了"等）
        action_patterns = [
            r"已[帮为].*[改查升发处]",
            r"查到[了]",
            r"已经.*处理",
            r"包裹目前",
            r"已在处理",
            r"已升级",
        ]

        claimed_action = False
        for pattern in action_patterns:
            if re.search(pattern, reply):
                claimed_action = True
                break

        if claimed_action:
            return {
                "type": HallucinationType.CAPABILITY_OVERREACH.label,
                "severity": HallucinationType.CAPABILITY_OVERREACH.default_severity.value,
                "detail": f"知识库标明: {gap_descriptions[0] if gap_descriptions else '系统不具备对应能力'}，但回复声称已执行对应操作",
                "confidence": 0.95,
            }

        return None

    # ---- 事实矛盾检测 ----

    def _check_factual_contradictions(self, case_id: str, kb: str, reply: str, question: str) -> list[dict]:
        """
        从 KB 和回复中提取关键数值/事实，逐一对比。
        使用预定义的特征提取规则。
        """
        findings = []
        kb_lower = kb.lower()
        reply_lower = reply.lower()

        # === 退货/保修政策对比 ===
        findings.extend(self._check_policy_terms(kb, reply, kb_lower, reply_lower))

        # === 产品参数对比 ===
        findings.extend(self._check_product_params(kb, reply, kb_lower, reply_lower))

        # === 优惠/促销对比 ===
        findings.extend(self._check_promotions(kb, reply, kb_lower, reply_lower, question))

        # === 品牌/门店/地址等信息对比 ===
        findings.extend(self._check_entity_info(kb, reply, kb_lower, reply_lower))

        return findings

    def _extract_number(self, text: str, keyword: str) -> Optional[str]:
        """从文本中提取与关键词关联的数字"""
        # 匹配 "关键词 + 数字" 或 "数字 + 关键词" 的模式
        patterns = [
            rf'{keyword}[^\d]*(\d+[\.\d]*\s*[天年月个日小周次]?)',
            rf'(\d+[\.\d]*\s*[天年月个日小周次]?)[^\d]*{keyword}',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _check_policy_terms(self, kb: str, reply: str, kb_lower: str, reply_lower: str) -> list[dict]:
        """检测退货/保修/发货政策中的矛盾"""
        findings = []

        policy_checks = [
            # (关键词, KB中的值, Reply中的值, 类型)
            ("退货天数", [r"(\d+)\s*天[^内]*无理由"], "政策编造"),
            ("发货时间", [r"(\d+)\s*小时[内]*发[货出]", r"(\d+)\s*天[内]*发[货出]"], "政策偏差"),
            ("保修期", [r"(\d+)\s*[个]*月[^，。]*保[修期]", r"(\d+)\s*年[^，。]*保[修期]"], "参数编造"),
        ]

        # 通用数值对比法：提取 KB 和 reply 中的关键数字
        # 退货天数
        kb_days = re.findall(r'(\d+)\s*天[^内]*无理由', kb)
        reply_days = re.findall(r'(\d+)\s*天[^内]*无理由', reply)
        if kb_days and reply_days and kb_days != reply_days:
            findings.append({
                "type": HallucinationType.POLICY_FABRICATION.label,
                "severity": HallucinationType.POLICY_FABRICATION.default_severity.value,
                "detail": f"退货政策: KB={kb_days[0]}天无理由 vs 回复={reply_days[0]}天无理由",
                "confidence": 0.9,
            })

        # 退货运费承担
        if "运费.*买家" in kb_lower and ("运费.*我们" in reply_lower or "运费.*商家" in reply_lower or "运费.*承担" in reply_lower and "全品类" in reply_lower):
            # 需要更细致的判断
            if "质量问题" not in reply_lower and "非质量" not in reply_lower:
                findings.append({
                    "type": HallucinationType.POLICY_FABRICATION.label,
                    "severity": HallucinationType.POLICY_FABRICATION.default_severity.value,
                    "detail": "退货运费规则与知识库矛盾",
                    "confidence": 0.85,
                })

        # 退货地址
        kb_has_address_rule = re.search(r'退货地址.*需.*系统.*匹配|不可.*口头.*告知', kb)
        reply_has_address = re.search(r'(?:退货|寄到|地址.*\d+号)', reply)
        if kb_has_address_rule and reply_has_address:
            findings.append({
                "type": HallucinationType.INFORMATION_FABRICATION.label,
                "severity": HallucinationType.INFORMATION_FABRICATION.default_severity.value,
                "detail": "知识库规定退货地址不能口头告知需系统匹配，回复却给出了具体地址",
                "confidence": 0.95,
            })

        # 快递公司对比
        kb_express = re.findall(r'(?:中通|韵达|圆通|顺丰|申通|百世|极兔)', kb)
        reply_express = re.findall(r'(?:中通|韵达|圆通|顺丰|申通|百世|极兔)', reply)
        if kb_express and reply_express and set(kb_express) != set(reply_express):
            findings.append({
                "type": HallucinationType.POLICY_DEVIATION.label,
                "severity": HallucinationType.POLICY_DEVIATION.default_severity.value,
                "detail": f"合作快递: KB={'/'.join(kb_express)} vs 回复={'/'.join(reply_express)}",
                "confidence": 0.85,
            })

        # 支付方式对比
        if "不支持货到付款" in kb_lower and "支持.*货到付款" in reply_lower:
            findings.append({
                "type": HallucinationType.POLICY_DEVIATION.label,
                "severity": HallucinationType.POLICY_DEVIATION.default_severity.value,
                "detail": "货到付款政策与知识库矛盾",
                "confidence": 0.9,
            })

        return findings

    def _check_product_params(self, kb: str, reply: str, kb_lower: str, reply_lower: str) -> list[dict]:
        """检测产品参数中的矛盾"""
        findings = []

        # 蓝牙版本
        kb_bt = re.findall(r'蓝牙\s*(\d+\.?\d*)', kb)
        reply_bt = re.findall(r'蓝牙\s*(\d+\.?\d*)', reply)
        if kb_bt and reply_bt and kb_bt != reply_bt:
            findings.append({
                "type": HallucinationType.PARAMETER_FABRICATION.label,
                "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                "detail": f"蓝牙版本: KB={kb_bt[0]} vs 回复={reply_bt[0]}",
                "confidence": 0.95,
            })

        # 材质对比
        kb_material = re.findall(r'(?:材质|采用|制作)[：:]*\s*([A-Za-z一-鿿]+(?:合成革|皮革|牛皮|羊皮|棉|麻|丝|绒|胶|塑料|硅胶))', kb)
        reply_material = re.findall(r'(?:材质|采用|制作|是)[：:]*\s*([A-Za-z一-鿿]+(?:合成革|皮革|牛皮|羊皮|棉|麻|丝|绒|胶|塑料|硅胶))', reply)
        if kb_material and reply_material and kb_material != reply_material:
            findings.append({
                "type": HallucinationType.PARAMETER_FABRICATION.label,
                "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                "detail": f"材质: KB={kb_material[0]} vs 回复={reply_material[0]}",
                "confidence": 0.9,
            })

        # 延迟
        kb_latency = re.findall(r'(\d+)\s*ms', kb)
        reply_latency = re.findall(r'(\d+)\s*ms', reply)
        if kb_latency and reply_latency and kb_latency != reply_latency:
            findings.append({
                "type": HallucinationType.PARAMETER_FABRICATION.label,
                "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                "detail": f"延迟参数: KB={kb_latency[0]}ms vs 回复={reply_latency[0]}ms",
                "confidence": 0.9,
            })

        # 接口类型
        # 优先提取 KB 中明确声明的"接口类型"
        kb_iface_declared = re.findall(r'接口类型[：:]\s*([A-Za-z\-]+)', kb, re.IGNORECASE)
        reply_iface_declared = re.findall(r'(?:接口|充电头)[^。，]*?(USB-[AC]|Type-C|Lightning|Micro.?USB)', reply, re.IGNORECASE)
        if kb_iface_declared and reply_iface_declared:
            kb_iface_norm = kb_iface_declared[0].upper().replace("-", "")
            reply_iface_norm = reply_iface_declared[0].upper().replace("-", "")
            if kb_iface_norm != reply_iface_norm:
                findings.append({
                    "type": HallucinationType.PARAMETER_FABRICATION.label,
                    "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                    "detail": f"接口类型: KB={kb_iface_declared[0]} vs 回复={reply_iface_declared[0]}",
                    "confidence": 0.95,
                })
        else:
            kb_interface = re.findall(r'(USB-[AC]|Type-C|Lightning|Micro.?USB)', kb, re.IGNORECASE)
            reply_interface = re.findall(r'(USB-[AC]|Type-C|Lightning|Micro.?USB)', reply, re.IGNORECASE)
            for ri in reply_interface:
                if ri.upper() not in [ki.upper() for ki in kb_interface] and kb_interface:
                    findings.append({
                        "type": HallucinationType.PARAMETER_FABRICATION.label,
                        "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                        "detail": f"接口类型: KB={'/'.join(kb_interface)} vs 回复={ri}",
                        "confidence": 0.95,
                    })
                    break

        # NFC / 特殊功能
        if ("nfc" in reply_lower and "支持" in reply_lower
                and ("未标注" in kb or "未提及" in kb or "无.*nfc" in kb_lower)):
            findings.append({
                "type": HallucinationType.PARAMETER_FABRICATION.label,
                "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                "detail": "NFC功能: 知识库未标注但回复声称支持",
                "confidence": 0.9,
            })

        # 多设备连接
        if "多设备" in reply_lower and "单设备" in kb_lower:
            findings.append({
                "type": HallucinationType.PARAMETER_FABRICATION.label,
                "severity": HallucinationType.PARAMETER_FABRICATION.default_severity.value,
                "detail": f"多设备连接: KB=单设备 vs 回复=多设备",
                "confidence": 0.9,
            })

        return findings

    def _check_promotions(self, kb: str, reply: str, kb_lower: str, reply_lower: str, question: str) -> list[dict]:
        """检测优惠/促销信息中的矛盾"""
        findings = []

        # 优惠券金额模式
        kb_coupons = re.findall(r'满\s*(\d+)\s*减\s*(\d+)', kb)
        reply_coupons = re.findall(r'满\s*(\d+)\s*减\s*(\d+)', reply)
        for rc in reply_coupons:
            if rc not in kb_coupons and kb_coupons:
                findings.append({
                    "type": HallucinationType.PROMOTION_FABRICATION.label,
                    "severity": HallucinationType.PROMOTION_FABRICATION.default_severity.value,
                    "detail": f"优惠券活动: 回复声称存在满{rc[0]}减{rc[1]}，但知识库中无此活动",
                    "confidence": 0.9,
                })

        # 学生优惠
        if "学生" in reply_lower and ("优惠" in reply_lower or "折扣" in reply_lower or "认证" in reply_lower):
            if "无.*学生" in kb_lower or "无学生" in kb_lower:
                findings.append({
                    "type": HallucinationType.PROMOTION_FABRICATION.label,
                    "severity": HallucinationType.PROMOTION_FABRICATION.default_severity.value,
                    "detail": "学生优惠: 知识库显示无此政策但回复声称存在",
                    "confidence": 0.95,
                })

        # "发到账户" 能力越界检测
        if re.search(r"发到.*账户|直接发", reply_lower) and "优惠" in question:
            findings.append({
                "type": HallucinationType.CAPABILITY_OVERREACH.label,
                "severity": HallucinationType.CAPABILITY_OVERREACH.default_severity.value,
                "detail": "回复声称已将优惠券发到用户账户，属于能力越界",
                "confidence": 0.85,
            })

        return findings

    def _check_entity_info(self, kb: str, reply: str, kb_lower: str, reply_lower: str) -> list[dict]:
        """检测门店/品牌/地址等实体信息的编造"""
        findings = []

        # 线下门店
        if "无线下" in kb_lower or "纯线上" in kb_lower:
            if re.search(r'(?:北京|上海|广州|深圳|成都|杭州|武汉).*体验店|线下.*门店|门店.*查询', reply):
                findings.append({
                    "type": HallucinationType.INFORMATION_FABRICATION.label,
                    "severity": HallucinationType.INFORMATION_FABRICATION.default_severity.value,
                    "detail": "品牌为纯线上电商，回复杜撰了线下门店信息",
                    "confidence": 0.95,
                })

        # 品牌关联
        if "旗下" in reply_lower or "子品牌" in reply_lower or "母公司" in reply_lower:
            if "未提及" in kb_lower or "无关" in kb_lower:
                findings.append({
                    "type": HallucinationType.INFORMATION_FABRICATION.label,
                    "severity": HallucinationType.INFORMATION_FABRICATION.default_severity.value,
                    "detail": "品牌关联关系: 知识库未提及，回复编造了与其他品牌的关联",
                    "confidence": 0.9,
                })

        # 发货时间
        kb_ship_hours = re.findall(r'(\d+)\s*小时[内]*发[货出]', kb)
        reply_ship_hours = re.findall(r'(\d+)\s*小时[内]*发[货出]', reply)
        if kb_ship_hours and reply_ship_hours and kb_ship_hours != reply_ship_hours:
            findings.append({
                "type": HallucinationType.POLICY_DEVIATION.label,
                "severity": HallucinationType.POLICY_DEVIATION.default_severity.value,
                "detail": f"发货时间: KB={kb_ship_hours[0]}小时 vs 回复={reply_ship_hours[0]}小时",
                "confidence": 0.9,
            })

        # 发票类型
        if "不支持纸质" in kb_lower or "暂不支持纸质" in kb_lower:
            if "纸质发票" in reply_lower and ("支持" in reply_lower or "可以" in reply_lower):
                findings.append({
                    "type": HallucinationType.POLICY_DEVIATION.label,
                    "severity": HallucinationType.POLICY_DEVIATION.default_severity.value,
                    "detail": "发票政策: 回复声称支持纸质发票，知识库只支持电子发票",
                    "confidence": 0.9,
                })
            # 发票申请方式
            if "备注" in reply_lower and "订单详情" in kb_lower:
                findings.append({
                    "type": HallucinationType.POLICY_DEVIATION.label,
                    "severity": HallucinationType.POLICY_DEVIATION.default_severity.value,
                    "detail": "发票申请方式: 回复说'备注里填写'，知识库说'订单详情页申请'",
                    "confidence": 0.85,
                })

        return findings

    # ---- 安全误导检测 ----

    def _check_safety_misleading(self, case_id: str, kb: str, reply: str) -> Optional[dict]:
        """检测回复是否弱化了知识库中的安全警告"""
        # KB 中有孕妇/安全警告
        has_safety_warning = False
        for pattern in self.SAFETY_KEYWORDS:
            if re.search(pattern, kb):
                has_safety_warning = True
                break

        if not has_safety_warning:
            return None

        # 回复中弱化了警告
        for pattern in self.SAFETY_DISMISS_PATTERNS:
            if re.search(pattern, reply):
                return {
                    "type": HallucinationType.SAFETY_MISLEADING.label,
                    "severity": HallucinationType.SAFETY_MISLEADING.default_severity.value,
                    "detail": "知识库中存在安全注意事项，但回复弱化/忽略了该警告，可能造成健康风险",
                    "confidence": 0.9,
                }

        return None

    # ---- 信息遗漏检测 ----

    def _check_information_omission(self, case_id: str, kb: str, reply: str) -> Optional[dict]:
        """检测回复是否遗漏了知识库中对用户决策重要的信息"""
        # 鞋码偏大/偏小
        if re.search(r'偏[大半小]', kb) and "标准" in reply and "不偏" in reply:
            return {
                "type": HallucinationType.INFORMATION_OMISSION.label,
                "severity": HallucinationType.INFORMATION_OMISSION.default_severity.value,
                "detail": "知识库中有用户反馈尺码偏大的信息，回复遗漏了这一关键信息，简单说'尺码标准'",
                "confidence": 0.75,
            }

        # 成分中含有特殊成分但回复未提及注意事项
        kb_warnings = re.findall(r'(?:注意事项|建议|需.*注意)[：:]\s*(.+)', kb)
        if kb_warnings and not any(w in reply for w in kb_warnings):
            # 仅在没有其他发现时才标记
            pass

        return None


# ============================================================
# LLM 检测器 — 支持 DeepSeek / OpenAI / Anthropic
# ============================================================

# 尝试导入 SDK（可选依赖）
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

try:
    import anthropic
    _anthropic_available = True
except ImportError:
    _anthropic_available = False


class LLMDetector(HallucinationDetector):
    """
    基于 LLM API 的检测器。

    支持三种后端：
    - deepseek:  DeepSeek V3/R1（推荐，性价比高）
    - openai:    OpenAI / 任何兼容接口
    - anthropic: Anthropic Claude

    用法:
        # DeepSeek
        detector = LLMDetector(backend="deepseek", api_key="sk-xxx")

        # 自定义 OpenAI 兼容接口
        detector = LLMDetector(
            backend="openai",
            api_key="sk-xxx",
            base_url="https://your-api.com/v1",
            model="your-model",
        )
    """

    # 各后端的默认配置
    BACKENDS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "env_key": "OPENAI_API_KEY",
        },
        "anthropic": {
            "base_url": None,
            "model": "claude-sonnet-4-20250514",
            "env_key": "ANTHROPIC_API_KEY",
        },
    }

    def __init__(
        self,
        backend: str = "deepseek",
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        cfg = self.BACKENDS.get(backend, self.BACKENDS["deepseek"])

        self.backend = backend
        self.api_key = api_key or os.environ.get(cfg["env_key"])
        self.model = model or cfg["model"]
        self.base_url = base_url or cfg["base_url"]
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                f"使用 {backend} 需要 API key。\n"
                f"方式1: 设置环境变量 {cfg['env_key']}\n"
                f"方式2: 传入参数 --api-key YOUR_KEY\n"
                f"方式3: 使用 --mode mock 切换到规则模式"
            )

        # 初始化客户端
        if backend == "anthropic":
            if not _anthropic_available:
                raise ImportError("使用 Anthropic 需要: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            if not _openai_available:
                raise ImportError("使用 OpenAI 兼容接口需要: pip install openai")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def detect(self, case: dict) -> HallucinationResult:
        system_prompt, user_prompt = self._build_prompts(case)

        if self.backend == "anthropic":
            response_text = self._call_anthropic(system_prompt, user_prompt)
        else:
            response_text = self._call_openai_compatible(system_prompt, user_prompt)

        return self._parse_response(case, response_text)

    def detect_all(self, cases: list[dict]) -> list[HallucinationResult]:
        results = []
        for i, case in enumerate(cases):
            print(f"  [{i+1}/{len(cases)}] 检测 {case['id']}...", end=" ")
            try:
                result = self.detect(case)
                status = f"幻觉({result.hallucination_type})" if result.is_hallucination else "正常"
                print(status)
                results.append(result)
            except Exception as e:
                print(f"出错: {e}")
                # 降级：标记为检测失败
                results.append(HallucinationResult(
                    id=case.get("id", "unknown"),
                    user_question=case.get("user_question", ""),
                    system_reply=case.get("system_reply", ""),
                    knowledge_base=case.get("knowledge_base", ""),
                    is_hallucination=False,
                    detail=f"LLM 调用失败: {str(e)[:200]}",
                    confidence=0.0,
                ))
        return results

    # ---- 内部方法 ----

    def _build_prompts(self, case: dict) -> tuple:
        """构建 system + user prompt"""

        types_desc = "\n".join([
            f"- {t.label}: {t.description}（严重程度: {t.default_severity.value}）"
            for t in HallucinationType
        ])

        severity_guide = """
严重程度标准:
- Critical: 可能造成安全/健康风险（如孕妇用药误导）
- High: 严重误导消费决策（编造材质、参数、退货政策、声称不具备的能力）
- Medium: 事实性错误但不直接涉及安全/金钱（部分政策偏差、编造优惠）
- Low: 轻微偏差、信息遗漏"""

        system_prompt = f"""你是一个严格的客服回复幻觉检测系统。你的任务是对比「系统回复」与「知识库」，找出所有不一致。

## 幻觉类型
{types_desc}
{severity_guide}

## 输出格式（严格 JSON，不要其他文字）
{{
  "is_hallucination": true/false,
  "hallucination_type": "参数编造/政策编造/信息编造/优惠编造/能力越界/政策偏差/安全误导/信息遗漏",
  "severity": "Critical/High/Medium/Low",
  "detail": "具体说明哪里不一致，引述 KB 原文和回复原文对比",
  "confidence": 0.0-1.0
}}"""

        user_prompt = f"""## 用户问题
{case.get('user_question', '')}

## 系统回复（待检测）
{case.get('system_reply', '')}

## 知识库（事实依据）
{case.get('knowledge_base', '')}

请逐句对比系统回复与知识库，输出 JSON 结果。"""

        return system_prompt, user_prompt

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI 兼容 API（DeepSeek / OpenAI / 自定义）"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Anthropic Claude API"""
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text.strip()

    def _parse_response(self, case: dict, text: str) -> HallucinationResult:
        """解析 LLM 返回的 JSON"""
        # 清理 markdown 代码块包裹
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 片段
            m = re.search(r'\{[\s\S]*"is_hallucination"[\s\S]*\}', text)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return HallucinationResult(
                        id=case.get("id", "unknown"),
                        user_question=case.get("user_question", ""),
                        system_reply=case.get("system_reply", ""),
                        knowledge_base=case.get("knowledge_base", ""),
                        is_hallucination=False,
                        detail=f"JSON 解析失败，原始响应: {text[:300]}",
                        confidence=0.0,
                    )
            else:
                return HallucinationResult(
                    id=case.get("id", "unknown"),
                    user_question=case.get("user_question", ""),
                    system_reply=case.get("system_reply", ""),
                    knowledge_base=case.get("knowledge_base", ""),
                    is_hallucination=False,
                    detail=f"JSON 解析失败，原始响应: {text[:300]}",
                    confidence=0.0,
                )

        return HallucinationResult(
            id=case.get("id", "unknown"),
            user_question=case.get("user_question", ""),
            system_reply=case.get("system_reply", ""),
            knowledge_base=case.get("knowledge_base", ""),
            is_hallucination=data.get("is_hallucination", False),
            hallucination_type=data.get("hallucination_type"),
            severity=data.get("severity"),
            detail=data.get("detail", ""),
            confidence=data.get("confidence", 0.8),
        )


# ============================================================
# 工厂函数
# ============================================================

def create_detector(mode: str = "mock", **kwargs) -> HallucinationDetector:
    """创建检测器实例

    Args:
        mode: "mock" | "llm"
        **kwargs: 传递给 LLMDetector 的参数
            - backend: "deepseek" (默认) | "openai" | "anthropic"
            - api_key: API key
            - model: 模型名称（可选，使用后端默认值）
            - base_url: 自定义 API 地址（可选）
    """
    if mode == "mock":
        return MockDetector()
    elif mode == "llm":
        return LLMDetector(**kwargs)
    else:
        raise ValueError(f"不支持的检测模式: {mode}，可选: mock, llm")
