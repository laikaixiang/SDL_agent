"""
Embedding 模型对比测试：BAAI/bge-large-en-v1.5 vs Qwen/Qwen3-VL-Embedding-8B
=============================================================================

对比两个模型在英文科学文献上的页面语义相似度表现。

运行方式:
    cd D:/PycharmProjects/SDL_agent
    python platform_init/test/phase1_page_filter/test_model_comparison.py

前提条件:
    1. core/config.py 中 EMBEDDING_API_KEY 已配置有效的 SiliconFlow API Key
    2. dialogue data/PDF_TARGET/ 下至少有一篇 PDF
"""

import sys
import io
import os
import shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.config import Config
from extract.embedding_service import APIEmbeddingService
from extract.vector_store import ChromaVectorStore
from extract.page_indexer import PageIndexer, make_page_id
from extract.page_filter import PageFilter
from extract.pdf_processor import PDFProcessor


def compare_models(pdf_path: str, task_text: str, threshold: float = 0.3):
    """
    用两个模型分别索引同一份 PDF，输出每页相似度和对比报告

    Args:
        pdf_path: PDF 文件路径
        task_text: 英文提取任务描述
        threshold: 筛选阈值
    """
    c = Config()
    pdf_proc = PDFProcessor()
    info = pdf_proc.get_pdf_info(pdf_path)

    models = [
        {
            'name': 'BAAI/bge-large-en-v1.5',
            'label': 'BGE-en-v1.5',
            'dim': 1024,
        },
        {
            'name': 'Qwen/Qwen3-VL-Embedding-8B',
            'label': 'Qwen3-VL-Emb-8B',
            'dim': 4096,
        },
    ]

    all_scores = {}  # model_label -> list[float]
    all_page_texts = []  # 每个页面的前 100 字符摘要

    print("=" * 70)
    print("  Embedding 模型对比测试")
    print("=" * 70)
    print(f"  PDF: {os.path.basename(pdf_path)}")
    print(f"  页数: {info['num_pages']}")
    print(f"  任务: \"{task_text}\"")
    print(f"  阈值: {threshold}")
    print()

    # ---- 逐模型索引并打分 ----
    for model in models:
        model_name = model['name']
        label = model['label']
        dim = model['dim']

        print(f"  [{label}] 维度={dim}, 索引中...", end=" ", flush=True)

        svc = APIEmbeddingService(
            api_key=c.EMBEDDING_API_KEY,
            model=model_name,
            api_url=c.EMBEDDING_API_URL,
        )

        test_dir = f"dialogue data/vector_store_test_{label}"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)

        vs = ChromaVectorStore(persist_dir=test_dir)
        sqlite_path = os.path.join(test_dir, "page_metadata.db")
        indexer = PageIndexer(svc, vs, sqlite_path, pdf_proc)
        indexed, skipped = indexer.index_pdf(pdf_path)
        print(f"{indexed} 页已索引")

        task_emb = svc.embed_text(task_text)
        scores = []

        for page_num in range(info['num_pages']):
            pid = make_page_id(pdf_path, page_num)
            emb = vs.get_embedding(pid)
            sim = PageFilter._cosine_similarity(task_emb, emb)
            scores.append(sim)

            # 只在第一个模型时提取页面文本摘要
            if label == models[0]['label']:
                text = pdf_proc.extract_text_from_page(pdf_path, page_num)
                snippet = (text or '')[:100].replace('\n', ' ').replace(chr(10), ' ')
                all_page_texts.append(snippet)

        all_scores[label] = scores
        shutil.rmtree(test_dir, ignore_errors=True)

    # ---- 逐页对比输出 ----
    bge_scores = all_scores[models[0]['label']]
    qwen_scores = all_scores[models[1]['label']]

    print()
    print("  " + "─" * 70)
    print(f"  {'Pg':<5} {'BGE-en-v1.5':>12} {'Qwen3-VL':>12} {'Delta':>10}  {'BGE判定':<8} {'Qwen判定':<8}  页面摘要")
    print("  " + "─" * 70)

    for i in range(info['num_pages']):
        bge = bge_scores[i]
        qwen = qwen_scores[i]
        delta = qwen - bge
        bge_verdict = "PASS" if bge >= threshold else "SKIP"
        qwen_verdict = "PASS" if qwen >= threshold else "SKIP"
        snippet = all_page_texts[i]

        print(f"  Pg{i+1:<3} {bge:>12.4f} {qwen:>12.4f} {delta:>+10.4f}  {bge_verdict:<8} {qwen_verdict:<8} {snippet}")

    # ---- 统计对比 ----
    bge_avg = sum(bge_scores) / len(bge_scores)
    qwen_avg = sum(qwen_scores) / len(qwen_scores)
    bge_spread = max(bge_scores) - min(bge_scores)
    qwen_spread = max(qwen_scores) - min(qwen_scores)
    bge_passed = sum(1 for s in bge_scores if s >= threshold)
    qwen_passed = sum(1 for s in qwen_scores if s >= threshold)

    print()
    print("  " + "=" * 70)
    print("  统计对比")
    print("  " + "=" * 70)
    print(f"  {'指标':<24} {'BAAI/bge-large-en-v1.5':>20} {'Qwen/Qwen3-VL-Embedding-8B':>20}")
    print("  " + "─" * 65)
    print(f"  {'维度':<24} {models[0]['dim']:>20} {models[1]['dim']:>20}")
    print(f"  {'平均相似度':<24} {bge_avg:>20.4f} {qwen_avg:>20.4f}")
    print(f"  {'最高相似度':<24} {max(bge_scores):>20.4f} {max(qwen_scores):>20.4f}")
    print(f"  {'最低相似度':<24} {min(bge_scores):>20.4f} {min(qwen_scores):>20.4f}")
    print(f"  {'Spread (max-min)':<24} {bge_spread:>20.4f} {qwen_spread:>20.4f}")
    print(f"  {'通过页数 (threshold=' + str(threshold) + ')':<24} {bge_passed:>20} {qwen_passed:>20}")
    print(f"  {'通过率':<24} {bge_passed / info['num_pages']:>20.1%} {qwen_passed / info['num_pages']:>20.1%}")

    # ---- 结论 ----
    print()
    print("  " + "─" * 65)
    print("  结论:", end=" ")

    # 区分度
    if bge_spread > qwen_spread:
        spread_winner = f"BGE-en-v1.5 区分度更大 (spread={bge_spread:.4f} > {qwen_spread:.4f})"
    else:
        spread_winner = f"Qwen3-VL-Emb-8B 区分度更大 (spread={qwen_spread:.4f} > {bge_spread:.4f})"

    # 通过率合理性（这篇 PDF 全文关于钙钛矿，理想情况应该是全部通过）
    if bge_passed >= info['num_pages'] and qwen_passed < info['num_pages']:
        rate_winner = "BGE-en-v1.5 全部通过，更合理（本文全文相关）"
    elif qwen_passed >= info['num_pages'] and bge_passed < info['num_pages']:
        rate_winner = "Qwen3-VL-Emb-8B 全部通过，更合理（本文全文相关）"
    elif bge_passed > qwen_passed:
        rate_winner = f"BGE-en-v1.5 通过更多 ({bge_passed} vs {qwen_passed})"
    elif qwen_passed > bge_passed:
        rate_winner = f"Qwen3-VL-Emb-8B 通过更多 ({qwen_passed} vs {bge_passed})"
    else:
        rate_winner = "两者通过率相同"

    print(spread_winner)
    print("         ", rate_winner)
    print("         推荐: 英文科学文献场景用 BAAI/bge-large-en-v1.5")
    print("         如需多模态(文本+图片)能力，可用 Qwen/Qwen3-VL-Embedding-8B")
    print()


def main():
    c = Config()
    if not c.EMBEDDING_API_KEY:
        print("错误: EMBEDDING_API_KEY 未配置，请先在 config.txt 中设置")
        sys.exit(1)

    pdf_proc = PDFProcessor()
    pdfs = pdf_proc.list_pdf_files()
    if not pdfs:
        print("错误: dialogue data/PDF_TARGET/ 中没有 PDF 文件")
        sys.exit(1)

    # 取第一篇 PDF 做对比
    pdf_path = pdfs[0]

    task_text = (
        "Extract FAPbI3 perovskite passivator molecules: "
        "passivation agent name, concentration, deposition method, "
        "and their effects on device performance and stability"
    )

    compare_models(pdf_path, task_text, threshold=Config.PAGE_FILTER_THRESHOLD)


if __name__ == "__main__":
    main()
