"""
报告生成模块

输出：
1. 终端输出：完整评估报告
2. JSON 文件：结构化结果
3. 文本文件：可存档的摘要报告
"""

import json
import os
from datetime import datetime
from collections import Counter

from .classifier import HallucinationResult, DetectionReport
from .evaluator import EvalMetrics


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        results: list[HallucinationResult],
        metrics: EvalMetrics,
    ) -> str:
        """生成完整报告，返回报告文本"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 统计
        hallucination_results = [r for r in results if r.is_hallucination]
        non_hallucination_results = [r for r in results if not r.is_hallucination]

        type_dist = Counter(r.hallucination_type for r in hallucination_results)
        severity_dist = Counter(r.severity for r in hallucination_results)

        # 构建报告
        lines = []
        lines.append("=" * 70)
        lines.append("          客服回复幻觉检测 — 评估报告")
        lines.append(f"          生成时间: {now}")
        lines.append("=" * 70)

        # ---- 一、整体得分 ----
        lines.append("")
        lines.append("一、整体检测结果")
        lines.append("-" * 40)
        lines.append(f"  总样本数:           {len(results)}")
        lines.append(f"  检测为幻觉:         {len(hallucination_results)} 条")
        lines.append(f"  检测为非幻觉:       {len(non_hallucination_results)} 条")
        lines.append(f"  幻觉比例:           {len(hallucination_results)/len(results)*100:.1f}%")
        lines.append("")

        # ---- 二、幻觉类型分布 ----
        lines.append("二、幻觉类型分布")
        lines.append("-" * 40)
        for htype, count in type_dist.most_common():
            bar = "█" * count
            lines.append(f"  {htype:12s}  {count:2d} 条  {bar}")
        lines.append("")

        # ---- 三、严重程度分布 ----
        lines.append("三、严重程度分布")
        lines.append("-" * 40)
        sev_order = ["Critical", "High", "Medium", "Low"]
        for sev in sev_order:
            count = severity_dist.get(sev, 0)
            if count > 0:
                bar = "█" * count
                lines.append(f"  {sev:12s}  {count:2d} 条  {bar}")
        lines.append("")

        # ---- 四、检出率对比 ----
        lines.append("四、与 Ground Truth 对比")
        lines.append("-" * 40)
        lines.append(f"  真正例 (TP):  {metrics.tp:2d}   假正例 (FP):  {metrics.fp:2d}")
        lines.append(f"  假负例 (FN):  {metrics.fn:2d}   真负例 (TN):  {metrics.tn:2d}")
        lines.append("")
        lines.append(f"  精确率 (Precision):  {metrics.precision:.2%}")
        lines.append(f"  召回率 (Recall):     {metrics.recall:.2%}")
        lines.append(f"  F1 分数:             {metrics.f1:.2%}")
        lines.append(f"  准确率 (Accuracy):   {metrics.accuracy:.2%}")
        lines.append(f"  类型匹配率:          {metrics.type_matches}/{metrics.tp}")
        lines.append("")

        # ---- 五、误报分析 (FP) ----
        if metrics.false_positives:
            lines.append("五、误报分析 (False Positives)")
            lines.append("-" * 40)
            lines.append(f"  共 {len(metrics.false_positives)} 条误报：")
            for i, fp in enumerate(metrics.false_positives, 1):
                lines.append(f"  {i}. [{fp['id']}] 检测为「{fp['detected_type']}」")
                lines.append(f"     Ground Truth: 非幻觉")
                lines.append(f"     检测详情: {fp['detail'][:120]}")
                lines.append("")
        else:
            lines.append("五、误报分析: 无误报")
            lines.append("")

        # ---- 六、漏检分析 (FN) ----
        if metrics.false_negatives:
            lines.append("六、漏检分析 (False Negatives)")
            lines.append("-" * 40)
            lines.append(f"  共 {len(metrics.false_negatives)} 条漏检：")
            for i, fn in enumerate(metrics.false_negatives, 1):
                lines.append(f"  {i}. [{fn['id']}] Ground Truth: 「{fn['ground_truth_type']}」")
                lines.append(f"     GT 详情: {fn['gt_detail'][:120]}")
                lines.append("")
        else:
            lines.append("六、漏检分析: 无漏检")
            lines.append("")

        # ---- 七、类型误判 ----
        if metrics.type_mismatches:
            lines.append("七、类型匹配偏差")
            lines.append("-" * 40)
            for i, tm in enumerate(metrics.type_mismatches, 1):
                lines.append(f"  {i}. [{tm['id']}] 检测=「{tm['detected_type']}」 GT=「{tm['ground_truth_type']}」")
            lines.append("")

        # ---- 八、逐条检测详情 ----
        lines.append("八、逐条检测详情")
        lines.append("-" * 40)
        for r in results:
            gt = self._find_gt(r.id, metrics)
            status = "OK" if self._is_correct(r, gt) else "XX"
            lines.append(f"  [{status}] {r.id}")
            lines.append(f"       用户问题: {r.user_question[:60]}")
            lines.append(f"       检测结果: {'幻觉' if r.is_hallucination else '非幻觉'}"
                         f"{' (' + r.hallucination_type + ')' if r.hallucination_type else ''}")
            lines.append(f"       严重程度: {r.severity or 'N/A'}")
            lines.append(f"       置信度:   {r.confidence:.0%}")
            lines.append(f"       详情:     {r.detail[:150]}")
            lines.append("")

        # ---- 九、局限性讨论 ----
        lines.append("九、检测方法局限性")
        lines.append("-" * 40)
        lines.append(self._limitations_text())

        report_text = "\n".join(lines)

        # 保存到文件
        self._save_json(results, metrics, type_dist, severity_dist, now)
        self._save_text(report_text, now)

        return report_text

    def _find_gt(self, result_id: str, metrics: EvalMetrics) -> dict:
        """查找 ground truth 标注"""
        for fp in metrics.false_positives:
            if fp["id"] == result_id:
                return {"is_hallucination": False}
        for fn in metrics.false_negatives:
            if fn["id"] == result_id:
                return {"is_hallucination": True}
        # 默认：与检测结果一致即为正确
        return None

    def _is_correct(self, result: HallucinationResult, gt: dict) -> bool:
        """判断检测是否正确"""
        if gt is None:
            return True
        gt_h = gt.get("is_hallucination")
        return result.is_hallucination == gt_h

    def _save_json(self, results, metrics, type_dist, severity_dist, now):
        """保存 JSON 格式的详细结果"""
        output = {
            "report_time": now,
            "summary": {
                "total": len(results),
                "hallucination_count": sum(1 for r in results if r.is_hallucination),
                "non_hallucination_count": sum(1 for r in results if not r.is_hallucination),
            },
            "metrics": metrics.to_dict(),
            "type_distribution": dict(type_dist),
            "severity_distribution": dict(severity_dist),
            "results": [
                {
                    "id": r.id,
                    "user_question": r.user_question,
                    "system_reply": r.system_reply[:200],
                    "knowledge_base": r.knowledge_base[:200],
                    "is_hallucination": r.is_hallucination,
                    "hallucination_type": r.hallucination_type,
                    "severity": r.severity,
                    "confidence": r.confidence,
                    "detail": r.detail,
                }
                for r in results
            ],
        }
        path = os.path.join(self.output_dir, "detection_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[JSON 结果已保存] {path}")

    def _save_text(self, text: str, now: str):
        """保存纯文本报告"""
        path = os.path.join(self.output_dir, "report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[文本报告已保存] {path}")

    def _limitations_text(self) -> str:
        return (
            "1. Mock 模式覆盖度有限：基于关键词和正则匹配，对于隐含的、\n"
            "   需要语义理解的幻觉（如讽刺、暗示）可能漏检。\n"
            "2. LLM 模式的不确定性：LLM 判断可能受 prompt 表述影响，\n"
            "   存在类型误判的倾向（如倾向于把模糊 case 往更严重类型靠）。\n"
            "3. 参数精度依赖：提取数字/参数的规则难以覆盖所有格式，可能遗漏\n"
            "   非标准表述中的事实错误。\n"
            "4. 「信息遗漏」边界模糊：遗漏多少信息算幻觉需要主观判断，\n"
            "   不同标注者可能给出不同结论（如 h20 的鞋码信息）。\n"
            "5. 知识库覆盖范围：检测效果依赖知识库的完整性，如果知识库不完整，\n"
            "   无法有效判断。\n"
            "改进方向：\n"
            "- Mock + LLM 混合模式：规则初筛 + LLM 二次确认，兼顾速度与准确率\n"
            "- 建立细粒度的主张-证据对齐标注，减少边界争议\n"
            "- 对「部分正确」case 做分层判断而非二分类"
        )
