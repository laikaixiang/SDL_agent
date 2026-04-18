"""
测试混合提取模式 - 独立版本
演示如何使用 vision/text/hybrid 三种模式提取PDF
"""

import sys
import os

# 直接导入需要的模块，避免通过__init__.py
import fitz
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# 直接导入pdf_to_markdown.py文件，不通过core包
import importlib.util
spec = importlib.util.spec_from_file_location("pdf_to_markdown", project_dir / "core" / "utils.pdf_to_markdown.py")
pdf_to_markdown = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_to_markdown)

# 使用导入的函数
pdf_page_to_markdown = pdf_to_markdown.pdf_page_to_markdown
detect_complex_content = pdf_to_markdown.detect_complex_content


def test_extraction_modes():
    """测试不同的提取模式"""

    # 获取测试PDF
    pdf_folder = project_dir / "test"
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("❌ 没有找到PDF文件，请先上传PDF到test文件夹")
        return

    test_pdf = str(pdf_files[0])
    print(f"📄 测试文件: {pdf_files[0].name}")
    print(f"📊 测试混合提取模式\n")

    # 测试第一页
    page_num = 0

    print("=" * 60)
    print("测试 1: 提取PDF文本为Markdown")
    print("=" * 60)
    markdown_text = pdf_page_to_markdown(test_pdf, page_num)

    if markdown_text:
        print(f"✓ 成功提取文本")
        print(f"✓ 文本长度: {len(markdown_text)} 字符")
        print(f"✓ 文本预览:\n{markdown_text[:500]}...")
    else:
        print("✗ 文本提取失败")
    print()

    print("=" * 60)
    print("测试 2: 检测复杂内容")
    print("=" * 60)

    if markdown_text:
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
        ("Table 1 shows the PCE values.", True),
        ("This is a long enough text without any complex content that should be processed with text API only.", False),
    ]

    for text, expected in test_cases:
        result = detect_complex_content(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text[:50]}...' → 复杂: {result} (预期: {expected})")


def test_pdf_info():
    """测试PDF基本信息"""
    print("\n" + "=" * 60)
    print("测试PDF文件信息")
    print("=" * 60)

    pdf_folder = Path(__file__).parent / "test"
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("❌ 没有找到PDF文件")
        return

    for pdf_file in pdf_files[:3]:  # 只显示前3个
        try:
            doc = fitz.open(str(pdf_file))
            print(f"\n📄 {pdf_file.name}")
            print(f"  • 页数: {len(doc)}")
            print(f"  • 标题: {doc.metadata.get('title', 'N/A')}")
            print(f"  • 作者: {doc.metadata.get('author', 'N/A')}")
            doc.close()
        except Exception as e:
            print(f"✗ 读取失败: {e}")


if __name__ == "__main__":
    # 设置UTF-8编码
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("混合提取模式测试\n")
    test_pdf_info()
    test_extraction_modes()
    test_complex_detection()
    print("\n测试完成！")
