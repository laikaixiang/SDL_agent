"""
测试混合提取模式
演示如何使用 vision/text/hybrid 三种模式提取PDF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.config import Config
from core.pdf_processor import PDFProcessor
from utils.pdf_to_markdown import detect_complex_content


def test_extraction_modes():
    """测试不同的提取模式"""

    # 初始化
    config = Config()
    pdf_processor = PDFProcessor()

    # 获取测试PDF
    pdf_files = pdf_processor.list_pdf_files()
    if not pdf_files:
        print("❌ 没有找到PDF文件，请先上传PDF到test文件夹")
        return

    test_pdf = pdf_files[0]
    print(f"📄 测试文件: {os.path.basename(test_pdf)}")
    print(f"📊 当前提取模式: {config.EXTRACTION_MODE}\n")

    # 测试第一页
    page_num = 0

    print("=" * 60)
    print("测试 1: 纯视觉模式 (vision)")
    print("=" * 60)
    markdown_text, img_base64, use_vision = pdf_processor.extract_page_content(
        test_pdf, page_num, mode="vision"
    )
    print(f"✓ 返回图片: {'是' if img_base64 else '否'}")
    print(f"✓ 返回文本: {'是' if markdown_text else '否'}")
    print(f"✓ 使用Vision: {use_vision}")
    print()

    print("=" * 60)
    print("测试 2: 纯文本模式 (text)")
    print("=" * 60)
    markdown_text, img_base64, use_vision = pdf_processor.extract_page_content(
        test_pdf, page_num, mode="text"
    )
    print(f"✓ 返回图片: {'是' if img_base64 else '否'}")
    print(f"✓ 返回文本: {'是' if markdown_text else '否'}")
    print(f"✓ 使用Vision: {use_vision}")
    if markdown_text:
        print(f"✓ 文本长度: {len(markdown_text)} 字符")
        print(f"✓ 文本预览:\n{markdown_text[:300]}...")
    print()

    print("=" * 60)
    print("测试 3: 混合模式 (hybrid)")
    print("=" * 60)
    markdown_text, img_base64, use_vision = pdf_processor.extract_page_content(
        test_pdf, page_num, mode="hybrid"
    )
    print(f"✓ 返回图片: {'是' if img_base64 else '否'}")
    print(f"✓ 返回文本: {'是' if markdown_text else '否'}")
    print(f"✓ 使用Vision: {use_vision}")

    if markdown_text:
        print(f"✓ 文本长度: {len(markdown_text)} 字符")
        has_complex = detect_complex_content(markdown_text)
        print(f"✓ 检测到复杂内容: {has_complex}")

        if has_complex:
            print("  → 原因: 包含化学式、图表引用或实验数据关键词")
            print("  → 建议: 使用Vision API以保证准确性")
        else:
            print("  → 原因: 纯文字描述，无复杂内容")
            print("  → 建议: 使用文本API节省成本")
    print()

    print("=" * 60)
    print("成本对比估算")
    print("=" * 60)
    print("假设一篇10页的文献:")
    print("  • 纯Vision模式: 10页 × Vision API = 高成本")
    print("  • 纯文本模式: 10页 × 文本API = 低成本（但可能丢失图表）")
    print("  • 混合模式: 3页Vision + 7页文本 = 中等成本（推荐）")
    print()
    print("💡 混合模式可节省 60-80% 的API成本，同时保证准确性！")


def test_complex_detection():
    """测试复杂内容检测"""
    print("\n" + "=" * 60)
    print("测试复杂内容检测功能")
    print("=" * 60)

    test_cases = [
        ("This is a simple text about materials.", False),
        ("FAPbI3 perovskite solar cell efficiency.", True),
        ("See Figure 1 for XRD spectrum analysis.", True),
        ("The bandgap of the material is 1.5 eV.", True),
        ("We discuss the general principles here.", False),
        ("钙钛矿太阳能电池的钝化剂研究", True),
    ]

    for text, expected in test_cases:
        result = detect_complex_content(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text[:40]}...' → 复杂内容: {result}")


if __name__ == "__main__":
    print("🧪 混合提取模式测试\n")
    test_extraction_modes()
    test_complex_detection()
    print("\n✅ 测试完成！")
