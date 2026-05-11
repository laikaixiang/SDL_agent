"""
并发批处理器
使用ThreadPoolExecutor实现多PDF并发提取，支持进度回调和失败重试
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
import traceback


class BatchProcessor:
    """
    并发批处理器

    职责：
    - 管理并发线程池
    - 对每篇PDF执行提取流程
    - 失败自动重试
    - 进度回调通知
    - 汇总结果统计
    """

    def __init__(self, max_workers: int = 3, retry_attempts: int = 2):
        """
        初始化批处理器

        Args:
            max_workers: 最大并发数
            retry_attempts: 单个PDF失败后重试次数
        """
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts

    def process_all(
        self,
        pdf_paths: list[str],
        process_one: Callable[[str], dict],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        并发处理所有PDF

        Args:
            pdf_paths: PDF文件路径列表
            process_one: 单篇处理函数，签名为 fn(pdf_path: str) -> dict
                         返回 {"status": "done"|"skipped"|"failed", ...}
            on_progress: 进度回调 (current, total, status_message)

        Returns:
            汇总结果: {
                "total": 总数,
                "skipped": 跳过数（mtime未变）,
                "extracted": 成功提取数,
                "failed": 失败数,
                "errors": [{"file": "xxx.pdf", "error": "错误信息"}, ...]
            }
        """
        total = len(pdf_paths)
        stats = {"total": total, "skipped": 0, "extracted": 0, "failed": 0, "errors": []}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._process_with_retry, process_one, path): path
                for path in pdf_paths
            }

            for future in as_completed(future_map):
                pdf_path = future_map[future]
                try:
                    result = future.result()
                    if result["status"] == "done":
                        stats["extracted"] += 1
                    elif result["status"] == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append({
                            "file": pdf_path,
                            "error": result.get("error", "未知错误")
                        })
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({
                        "file": pdf_path,
                        "error": str(e)
                    })

                if on_progress:
                    completed = stats["extracted"] + stats["skipped"] + stats["failed"]
                    on_progress(completed, total, f"已完成 {completed}/{total}")

        return stats

    def _process_with_retry(self, process_one: Callable, pdf_path: str) -> dict:
        """带重试的单篇处理"""
        last_error = ""
        for attempt in range(self.retry_attempts + 1):
            try:
                result = process_one(pdf_path)
                return result
            except Exception as e:
                last_error = f"{e}\n{traceback.format_exc()}"
                if attempt < self.retry_attempts:
                    print(f"处理失败 [{pdf_path}]，第{attempt+1}次重试中...")
                else:
                    print(f"处理失败 [{pdf_path}]，已达最大重试次数")
        return {"status": "failed", "error": last_error}
