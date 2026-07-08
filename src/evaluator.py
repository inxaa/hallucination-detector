"""
评估模块：将检测结果与 ground_truth 对比，计算检出率指标。

指标说明：
- TP (True Positive):  检测为幻觉，ground_truth 也为幻觉
- FP (False Positive): 检测为幻觉，ground_truth 为非幻觉（误报）
- TN (True Negative):  检测为非幻觉，ground_truth 也为非幻觉
- FN (False Negative): 检测为非幻觉，ground_truth 为幻觉（漏检）

精确率 (Precision) = TP / (TP + FP)  —— 检测为幻觉中有多少是真的
召回率 (Recall)    = TP / (TP + FN)  —— 真实幻觉中有多少被检出
F1 分数            = 2 * P * R / (P + R)
"""

import json
from typing import Optional
from dataclasses import dataclass, field

from .classifier import HallucinationResult


@dataclass
class EvalMetrics:
    """评估指标"""
    total: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0

    false_positives: list[dict] = field(default_factory=list)
    false_negatives: list[dict] = field(default_factory=list)
    type_matches: int = 0        # 类型也匹配正确的数量
    type_mismatches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "type_matches": self.type_matches,
            "type_mismatches": self.type_mismatches,
        }


class Evaluator:
    """对比检测结果与 ground_truth"""

    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            self.ground_truth = json.load(f)
        self.gt_map = {item["id"]: item for item in self.ground_truth}

    def evaluate(self, results: list[HallucinationResult]) -> EvalMetrics:
        """评估检测结果"""
        metrics = EvalMetrics()
        metrics.total = len(results)

        for r in results:
            gt = self.gt_map.get(r.id)
            if gt is None:
                print(f"警告: {r.id} 在 ground_truth 中找不到")
                continue

            gt_is_hallucination = gt.get("is_hallucination", False)
            gt_type = gt.get("hallucination_type")

            if r.is_hallucination and gt_is_hallucination:
                # TP: 正确检出幻觉
                metrics.tp += 1
                # 检查类型是否匹配
                if gt_type and r.hallucination_type:
                    if self._type_matches(r.hallucination_type, gt_type):
                        metrics.type_matches += 1
                    else:
                        metrics.type_mismatches.append({
                            "id": r.id,
                            "detected_type": r.hallucination_type,
                            "ground_truth_type": gt_type,
                            "detail": r.detail,
                        })
            elif r.is_hallucination and not gt_is_hallucination:
                # FP: 误报
                metrics.fp += 1
                metrics.false_positives.append({
                    "id": r.id,
                    "detected_type": r.hallucination_type,
                    "ground_truth_type": gt_type,
                    "detail": r.detail,
                    "gt_detail": gt.get("detail", ""),
                })
            elif not r.is_hallucination and gt_is_hallucination:
                # FN: 漏检
                metrics.fn += 1
                metrics.false_negatives.append({
                    "id": r.id,
                    "ground_truth_type": gt_type,
                    "gt_detail": gt.get("detail", ""),
                    "detector_detail": r.detail,
                })
            else:
                # TN: 正确识别非幻觉
                metrics.tn += 1

        # 计算指标
        if metrics.tp + metrics.fp > 0:
            metrics.precision = metrics.tp / (metrics.tp + metrics.fp)
        if metrics.tp + metrics.fn > 0:
            metrics.recall = metrics.tp / (metrics.tp + metrics.fn)
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        if metrics.total > 0:
            metrics.accuracy = (metrics.tp + metrics.tn) / metrics.total

        return metrics

    def _type_matches(self, detected: str, ground_truth: str) -> bool:
        """检查类型是否匹配（允许相近类型的模糊匹配）"""
        # 精确匹配
        if detected == ground_truth:
            return True

        # 相近类型映射
        similar_types = {
            "政策编造": ["政策编造", "政策偏差"],
            "政策偏差": ["政策编造", "政策偏差"],
            "参数编造": ["参数编造"],
            "优惠编造": ["优惠编造", "信息编造"],
            "信息编造": ["信息编造", "优惠编造"],
            "信息遗漏": ["信息遗漏", "政策偏差"],
            "能力越界": ["能力越界"],
            "安全误导": ["安全误导"],
        }

        if detected in similar_types:
            return ground_truth in similar_types[detected]
        return False

    def get_gt_hallucination_ids(self) -> set:
        """获取 ground_truth 中标记为幻觉的 ID 集合"""
        return {item["id"] for item in self.ground_truth if item.get("is_hallucination")}

    def get_gt_non_hallucination_ids(self) -> set:
        """获取 ground_truth 中标记为非幻觉的 ID 集合"""
        return {item["id"] for item in self.ground_truth if not item.get("is_hallucination")}
