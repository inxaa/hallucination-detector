"""
客服回复幻觉检测 — 主入口

用法:
    # Mock 规则模式（默认，零依赖）
    python main.py

    # DeepSeek LLM 模式（先在 .env 里配好 key，或设环境变量）
    python main.py --mode llm

    # 也可以直接传 key
    python main.py --mode llm --api-key sk-xxx

    # 自定义后端/模型
    python main.py --mode llm --backend openai --model gpt-4o
"""

import argparse
import json
import os
import sys
import io


def load_dotenv(dotenv_path: str = None):
    """加载 .env 文件到环境变量（零依赖实现）"""
    if dotenv_path is None:
        # 从 main.py 所在目录找 .env
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if value and key not in os.environ:
                    os.environ[key] = value

# 修复 Windows 终端编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.detector import create_detector
from src.evaluator import Evaluator
from src.report import ReportGenerator


def main():
    # 自动加载 .env 文件（如果存在）
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="客服回复幻觉检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py                                        # Mock 规则模式（零依赖）
    python main.py --mode llm --api-key sk-xxx            # DeepSeek（默认后端）
    python main.py --mode llm --backend openai            # OpenAI GPT-4o
    python main.py --mode llm --model deepseek-chat       # 指定模型
    python main.py --mode llm --temperature 0.3           # 调高温度
        """,
    )
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock",
                        help="检测模式: mock (规则引擎) / llm (LLM API)")
    parser.add_argument("--backend", choices=["deepseek", "openai", "anthropic"],
                        default="deepseek",
                        help="LLM 后端 (默认: deepseek)")
    parser.add_argument("--api-key", default=None,
                        help="API key（也可设环境变量 DEEPSEEK_API_KEY / OPENAI_API_KEY）")
    parser.add_argument("--model", default=None,
                        help="模型名称（默认: deepseek-chat）")
    parser.add_argument("--base-url", default=None,
                        help="自定义 API 地址（覆盖后端默认值）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM 温度 (默认: 0)")
    parser.add_argument("--data-dir", default="./data",
                        help="数据目录路径")
    parser.add_argument("--output-dir", default="./output",
                        help="输出目录路径")
    args = parser.parse_args()

    # 路径
    replies_path = os.path.join(args.data_dir, "replies.json")
    ground_truth_path = os.path.join(args.data_dir, "ground_truth.json")

    # 检查文件
    for path in [replies_path, ground_truth_path]:
        if not os.path.exists(path):
            print(f"错误: 找不到文件 {path}")
            sys.exit(1)

    # 加载数据
    print("=" * 70)
    print("  客服回复幻觉检测工具")
    if args.mode == "mock":
        print("  检测模式: Mock (规则引擎)")
    else:
        print(f"  检测模式: LLM ({args.backend} / {args.model or 'default'})")
    print("=" * 70)

    with open(replies_path, "r", encoding="utf-8") as f:
        replies = json.load(f)
    print(f"\n[加载] 回复数据: {len(replies)} 条 ({replies_path})")
    print(f"[加载] Ground Truth ({ground_truth_path})")

    # 创建检测器并执行检测
    print(f"\n[检测] 开始...")

    detector_kwargs = {"mode": args.mode}
    if args.mode == "llm":
        detector_kwargs.update({
            "backend": args.backend,
            "api_key": args.api_key,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
        })
        # 过滤掉 None 值
        detector_kwargs = {k: v for k, v in detector_kwargs.items()
                           if v is not None or k in ("mode", "backend", "temperature")}

    detector = create_detector(**detector_kwargs)
    results = detector.detect_all(replies)

    hallucination_count = sum(1 for r in results if r.is_hallucination)
    print(f"\n  检测完成: {hallucination_count}/{len(results)} 条被标记为幻觉")

    # 评估（与 ground truth 对比）
    print("\n[评估] 对比 Ground Truth...")
    evaluator = Evaluator(ground_truth_path)
    metrics = evaluator.evaluate(results)

    print(f"  精确率: {metrics.precision:.2%}")
    print(f"  召回率: {metrics.recall:.2%}")
    print(f"  F1 分数: {metrics.f1:.2%}")
    print(f"  误报: {len(metrics.false_positives)} 条, 漏检: {len(metrics.false_negatives)} 条")

    # 生成报告
    print("\n[报告] 生成评估报告...")
    reporter = ReportGenerator(output_dir=args.output_dir)
    report_text = reporter.generate(results, metrics)

    # 打印报告到终端
    print("\n" + report_text)


if __name__ == "__main__":
    main()
