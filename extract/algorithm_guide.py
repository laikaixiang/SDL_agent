"""
逐步引导式算法生成模块

通过 4 轮问答引导用户描述算法需求，收集答案后拼接为结构化 prompt，
调用软件管理器生成算法代码。

会话持久化到 session_path/algorithm_guide.json，服务重启不丢失。
"""

import json
import os
import uuid
from typing import Optional

_GUIDE_QUESTIONS = [
    # Q1
    """好的，让我一步步了解您的需求。

（1/4）首先，请告诉我：**这个算法要解决什么问题？它的核心功能是什么？**

比如：
- 对光谱数据做平滑去噪处理
- 对实验数据进行正态分布拟合并计算置信区间
- 对多维数组做主成分分析（PCA）降维

请用一两句话描述算法的主要功能。""",

    # Q2
    """明白了。

（2/4）接下来：**输入数据是什么样的？**

请描述数据的结构，包括：
- **数据格式**：dict、list、numpy 数组、CSV 文件路径？
- **关键字段**：各字段的名称和含义（如 wavelength 波长列表、intensity 强度列表）
- **数据规模**：通常有多少个数据点？

比如：「输入是一个 dict，包含 'wavelength'（波长列表，长度约 2000）和 'intensity'（对应的强度值列表）两个键。」""",

    # Q3
    """清楚了。

（3/4）接下来：**您希望算法输出什么结果？**

请描述期望的输出内容，包括：
- **输出类型**：数值、列表、统计指标、图表？
- **各字段含义**：如平滑后的强度序列、平滑度评分、残差序列
- **是否需要可视化**：是否需要生成图表（折线图、散点图等）？

比如：「输出包括平滑后的强度序列（list）、每个点的残差值（list）、平滑程度的综合评分（float）。」""",

    # Q4
    """好的。

（4/4）最后一个问题：**算法有哪些可配置的参数？**

请列出用户可调整的参数，包括：
- **参数名称和含义**：如 window_size 窗口大小、poly_order 多项式阶数
- **默认值**：建议的默认值
- **参数类型和范围**：整数/浮点数/布尔值/字符串选项，有效范围

比如：「window_size：窗口大小，整数，默认值 5，范围 3-21；poly_order：多项式阶数，整数，默认值 2，可选 1-5。」

---

以上 4 个方面填写完成后，我将自动为您生成算法代码。""",
]


class AlgorithmGuide:
    """逐步引导式算法生成器，管理 4 轮问答会话，持久化到文件。"""

    def __init__(self, session_path: str = ""):
        self._sessions: dict[str, dict] = {}
        self._session_path = session_path
        self._load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _file_path(self) -> str:
        if not self._session_path:
            return ""
        return os.path.join(self._session_path, "algorithm_guide.json")

    def _save(self):
        path = self._file_path()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._sessions, f, ensure_ascii=False, indent=2)

    def _load(self):
        path = self._file_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._sessions = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._sessions = {}

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def handle(self, session_id: Optional[str] = None, answer: Optional[str] = None,
               action: str = "answer") -> dict:
        """
        处理引导流程的一次交互。

        Args:
            session_id: 会话 ID，首次调用时为 None。
            answer: 用户回答文本，action="answer" 时使用。
            action: "answer" | "back" | "cancel"

        Returns:
            dict: 提问阶段 {"stage":"question","reply":str,"progress":str,"session_id":str}
                  回退阶段 同上，额外含 "previous_answer":str
                  就绪阶段 {"stage":"ready","combined_prompt":str,"session_id":str}
                  取消阶段 {"stage":"cancelled"}
                  完成阶段 {"stage":"done","reply":str,"progress":"complete","success":bool,...}
        """
        action = action or "answer"

        # --- 取消 ---
        if action == "cancel":
            if session_id:
                self._sessions.pop(session_id, None)
                self._save()
            return {"stage": "cancelled"}

        # --- 首次调用 ---
        if not session_id:
            sid = str(uuid.uuid4())
            self._sessions[sid] = {"answers": ["", "", "", ""], "current_q": 0}
            self._save()
            return {
                "stage": "question",
                "reply": _GUIDE_QUESTIONS[0],
                "progress": "1/4",
                "session_id": sid,
            }

        # --- 验证会话 ---
        session = self._sessions.get(session_id)
        if not session:
            return {
                "stage": "done",
                "reply": "会话已过期，请重新点击「生成新算法」。",
                "progress": "complete",
                "success": False,
            }

        # --- 返回上一步 ---
        if action == "back":
            q_idx = session["current_q"]
            if q_idx > 0:
                session["current_q"] = q_idx - 1
                self._save()
                new_q = session["current_q"]
                return {
                    "stage": "question",
                    "reply": _GUIDE_QUESTIONS[new_q],
                    "progress": f"{new_q + 1}/4",
                    "session_id": session_id,
                    "previous_answer": session["answers"][new_q],
                }
            # 已在 Q1，无法回退
            return {
                "stage": "question",
                "reply": _GUIDE_QUESTIONS[0],
                "progress": "1/4",
                "session_id": session_id,
                "previous_answer": session["answers"][0],
            }

        # --- 回答当前问题 ---
        answer = (answer or "").strip()
        q_idx = session["current_q"]
        if answer:
            session["answers"][q_idx] = answer

        next_q = q_idx + 1

        # 4 题全部答完：返回拼接后的 prompt
        if next_q >= 4:
            answers = session["answers"]
            if not any(a.strip() for a in answers):
                self._sessions.pop(session_id, None)
                self._save()
                return {
                    "stage": "done",
                    "reply": "请至少填写一个方面的内容。请重新点击「生成新算法」开始。",
                    "progress": "complete",
                    "success": False,
                }

            combined = (
                f"算法功能：{answers[0]}\n"
                f"输入数据：{answers[1]}\n"
                f"期望输出：{answers[2]}\n"
                f"可调参数：{answers[3]}"
            )
            session["combined_prompt"] = combined
            session["current_q"] = next_q
            self._save()
            return {
                "stage": "ready",
                "combined_prompt": combined,
                "session_id": session_id,
            }

        # 还有下一题
        session["current_q"] = next_q
        self._save()
        return {
            "stage": "question",
            "reply": _GUIDE_QUESTIONS[next_q],
            "progress": f"{next_q + 1}/4",
            "session_id": session_id,
        }

    def finish(self, session_id: str, result: dict) -> dict:
        """
        收到算法生成结果后，构建最终响应并清理会话。

        Args:
            session_id: 会话 ID。
            result: software_manager.generate_algorithm() 的返回值。

        Returns:
            dict: 最终响应，stage="done"。
        """
        self._sessions.pop(session_id, None)
        self._save()

        if result.get("success"):
            reply = (
                f"✅ 算法生成成功！\n\n"
                f"算法名称: {result['name']}\n"
                f"文件路径: {result['filepath']}\n\n"
                f"{result.get('message', '')}\n\n"
                f"你现在可以在算法面板中使用这个算法了。"
            )
        else:
            reply = f"❌ 算法生成失败\n\n{result.get('message', '未知错误')}"

        return {
            "stage": "done",
            "reply": reply,
            "progress": "complete",
            "success": result.get("success", False),
            "name": result.get("name", ""),
            "filepath": result.get("filepath", ""),
        }
