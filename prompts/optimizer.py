"""
PromptOptimizer — 基于 LLM 的 Prompt 优化辅助

职责:
- 调用 LLM 分析 prompt 质量并产出优化建议
- 批量测试 prompt 在多个用例上的表现
- 优化前后对比
- 坏 case 诊断

原则: 只读 + 建议，不直接修改 prompt。
      真正写入走 PromptManager.update()。
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class OptimizeResult:
    """优化结果"""
    original: str
    optimized: str
    changes_summary: str


@dataclass
class TestResult:
    """单次测试结果"""
    rendered_prompt: str
    llm_response: str = ""
    error: str = ""
    passed: bool = True


@dataclass
class CompareResult:
    """优化前后对比结果"""
    original_results: List[TestResult] = field(default_factory=list)
    optimized_results: List[TestResult] = field(default_factory=list)
    summary: str = ""


class PromptOptimizer:
    """Prompt 优化器

    Args:
        manager: PromptManager 实例
        llm_client: LLMClient 或兼容对象（需有 call_api(model, messages, temperature, max_tokens) 方法）
    """

    def __init__(self, manager, llm_client):
        self.manager = manager
        self.llm = llm_client

    def optimize(
        self, name: str, requirements: str, test_inputs: list[dict] = None
    ) -> OptimizeResult:
        """调用 LLM 优化指定 prompt

        使用 meta_optimize 元 prompt 驱动 LLM 产出优化版本。

        Args:
            name: prompt 名称
            requirements: 优化需求描述（如"提高钙钛矿钝化场景的准确率"）
            test_inputs: 可选测试输入 [{"var": "val"}, ...]，帮助 LLM 理解使用场景

        Returns:
            OptimizeResult(original, optimized, changes_summary)
        """
        current = self.manager.get_meta(name)
        original_template = current["current_template"]

        # 渲染测试输入为可读文本
        test_text = self._format_test_inputs(test_inputs or [])

        # 用 meta_optimize 模板生成优化请求
        try:
            optimize_prompt = self.manager.get(
                "meta_optimize",
                current_prompt=original_template,
                prompt_name=name,
                prompt_description=current["description"],
                requirements=requirements,
                test_inputs=test_text,
            )
        except Exception:
            # meta_optimize 不存在时的 fallback
            optimize_prompt = self._fallback_optimize_prompt(
                name, original_template, requirements, test_text
            )

        messages = [{"role": "user", "content": optimize_prompt}]
        response = self._call_llm(messages, temperature=0.3, max_tokens=4096)

        optimized = self._strip_markdown(response)

        return OptimizeResult(
            original=original_template,
            optimized=optimized,
            changes_summary=self._diff_summary(original_template, optimized),
        )

    def batch_test(self, name: str, test_cases: list[dict]) -> list[TestResult]:
        """对 prompt 跑多个测试用例

        Args:
            name: prompt 名称
            test_cases: [{"variables": {...}, "user_content": "..."}, ...]
                        user_content 可选，如果提供则额外调用 LLM 验证

        Returns:
            [TestResult, ...]
        """
        results = []
        for case in test_cases:
            try:
                rendered = self.manager.get(name, **case.get("variables", {}))
                result = TestResult(rendered_prompt=rendered)

                user_content = case.get("user_content")
                if user_content:
                    messages = [
                        {"role": "system", "content": rendered},
                        {"role": "user", "content": user_content},
                    ]
                    result.llm_response = self._call_llm(
                        messages, temperature=0.1, max_tokens=2048
                    )
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    rendered_prompt="",
                    error=str(e),
                    passed=False,
                ))
        return results

    def compare(
        self, name: str, original: str, candidate: str, test_cases: list[dict]
    ) -> CompareResult:
        """优化前后对比测试

        先用 original 模板跑全量测试，临时切到 candidate 再跑一遍，逐条对比。

        Args:
            name: prompt 名称
            original: 原始模板
            candidate: 候选优化模板
            test_cases: 测试用例

        Returns:
            CompareResult 含两组结果 + 差异摘要
        """
        # 备份当前模板
        meta = self.manager.get_meta(name)

        # 测试 original
        orig_results = self._run_with_template(name, original, test_cases)

        # 测试 candidate
        cand_results = self._run_with_template(name, candidate, test_cases)

        # 生成摘要
        orig_fail = sum(1 for r in orig_results if not r.passed)
        cand_fail = sum(1 for r in cand_results if not r.passed)
        summary = (
            f"Original: {len(orig_results)} 测试, {orig_fail} 失败. "
            f"Optimized: {len(cand_results)} 测试, {cand_fail} 失败."
        )

        return CompareResult(
            original_results=orig_results,
            optimized_results=cand_results,
            summary=summary,
        )

    def suggest_improvements(self, name: str, bad_outputs: list[dict]) -> str:
        """对坏 case 给出诊断建议

        Args:
            name: prompt 名称
            bad_outputs: [{"input": "...", "actual_output": "...", "expected_output": "..."}, ...]

        Returns:
            LLM 诊断文本
        """
        current = self.manager.get_meta(name)
        cases_text = self._format_bad_cases(bad_outputs)

        diagnose_prompt = (
            f"以下 prompt 在特定场景产生了不符合预期的输出。\n\n"
            f"## Prompt\n{current['current_template']}\n\n"
            f"## 坏 Case\n{cases_text}\n\n"
            f"请分析 prompt 哪里导致了这些问题，给出具体的修改建议（不需要输出完整 prompt，只给诊断和建议）。"
        )

        messages = [{"role": "user", "content": diagnose_prompt}]
        return self._call_llm(messages, temperature=0.2, max_tokens=2048)

    # ═══════════════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════════════

    def _call_llm(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """调用 LLM，兼容 LLMClient 和直接 requests 风格"""
        try:
            # 尝试 LLMClient 风格
            result = self.llm.call_api(
                model=None,  # 用 LLMClient 默认模型
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if isinstance(result, tuple):
                _, text = result
                return text if isinstance(text, str) else str(text)
            return result
        except TypeError:
            # fallback: 直接 requests 调用
            return self._call_llm_raw(messages, temperature, max_tokens)

    def _call_llm_raw(self, messages: list, temperature: float, max_tokens: int) -> str:
        """直接 HTTP 调用 LLM API（兼容 auto_analyze / prompt_template 的 _call_llm 模式）"""
        import requests
        import json as _json

        api_url = getattr(self.llm, 'api_url', None)
        api_key = getattr(self.llm, 'api_key', None)

        if not api_url or not api_key:
            return ""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": getattr(self.llm, 'model_name', 'default'),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        return ""

    def _format_test_inputs(self, test_inputs: list[dict]) -> str:
        if not test_inputs:
            return "（无测试输入）"
        lines = []
        for i, case in enumerate(test_inputs, 1):
            lines.append(f"用例 {i}: {case}")
        return "\n".join(lines)

    def _format_bad_cases(self, bad_outputs: list[dict]) -> str:
        lines = []
        for i, case in enumerate(bad_outputs, 1):
            lines.append(
                f"### Case {i}\n"
                f"- 输入: {case.get('input', '')}\n"
                f"- 实际输出: {case.get('actual_output', '')}\n"
                f"- 期望输出: {case.get('expected_output', '')}\n"
            )
        return "\n".join(lines)

    def _fallback_optimize_prompt(
        self, name: str, template: str, requirements: str, test_text: str
    ) -> str:
        return (
            f"你是一个 prompt 工程专家。请优化以下 prompt。\n\n"
            f"## 当前 Prompt ({name})\n{template}\n\n"
            f"## 优化需求\n{requirements}\n\n"
            f"## 测试输入\n{test_text}\n\n"
            f"## 要求\n"
            f"1. 保持输出格式不变、变量名不变\n"
            f"2. 直接输出优化后的完整模板"
        )

    def _strip_markdown(self, text: str) -> str:
        """去除可能的 markdown 代码块包裹"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            end = 0
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            lines = lines[1:end]
            text = "\n".join(lines)
        return text.strip()

    def _diff_summary(self, original: str, optimized: str) -> str:
        """简单的差异摘要"""
        if original == optimized:
            return "无变化"
        if len(optimized) > len(original) * 1.3:
            return f"大幅扩展（{len(original)} → {len(optimized)} 字符）"
        if len(optimized) < len(original) * 0.7:
            return f"大幅精简（{len(original)} → {len(optimized)} 字符）"
        return f"微调（{len(original)} → {len(optimized)} 字符）"

    def _run_with_template(
        self, name: str, template: str, test_cases: list[dict]
    ) -> list[TestResult]:
        """用指定模板临时跑测试"""
        results = []
        for case in test_cases:
            try:
                result = TestResult(rendered_prompt=template)
                user_content = case.get("user_content")
                if user_content:
                    messages = [
                        {"role": "system", "content": template},
                        {"role": "user", "content": user_content},
                    ]
                    result.llm_response = self._call_llm(
                        messages, temperature=0.1, max_tokens=2048
                    )
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    rendered_prompt=template,
                    error=str(e),
                    passed=False,
                ))
        return results
