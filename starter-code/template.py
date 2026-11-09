"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

# TODO: Viết System Prompt cho VinAssistant
# Gợi ý các phần cần có:
# 1. PERSONA: Tên, vai trò, giọng nói
# 2. AVAILABLE TOOLS: Liệt kê {tools}
# 3. CORE RULES: Không bịa dữ liệu, bắt buộc gọi tool khi cần
# 4. OPERATIONAL BOUNDARIES: Chỉ trả lời về Vingroup
# 5. OUTPUT CONTRACT: Format trả lời (Thought/Action/Observation/Final Answer)

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm và dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả), trả lời trực tiếp KHÔNG cần gọi tool.
4. Nếu yêu cầu cần NHIỀU tool, hãy gọi TUẦN TỰ từng tool rồi tổng hợp kết quả cuối cùng.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về sản phẩm/dịch vụ thuộc hệ sinh thái Vingroup.
- Từ chối lịch sự nếu yêu cầu nằm ngoài phạm vi (tiền điện tử, chính trị, v.v.).

## OUTPUT CONTRACT
- Luôn trả lời bằng tiếng Việt.
- Khi liệt kê sản phẩm, dùng bảng hoặc danh sách có thứ tự.
- Khi tạo ticket, xác nhận lại ticket_id và mức ưu tiên cho khách hàng.

## RESPONSE FORMAT (cho mỗi bước suy luận)
Thought: <suy nghĩ bước này>
Action: {"name": "<tên tool>", "args": {<các tham số>}}
Observation: <kết quả trả về từ tool>
(Lặp lại nếu cần thêm tool)
Final Answer: <câu trả lời hoàn chỉnh cho khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không dùng Tool Calling hay ReAct Loop.

    Dùng để so sánh chất lượng trả lời khi LLM không có quyền tra cứu dữ liệu.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        if self.api_key:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                "Bạn là chatbot tư vấn sản phẩm Vingroup. "
                "Trả lời câu hỏi sau mà KHÔNG có quyền tra cứu cơ sở dữ liệu hay gọi tool: "
                f"{user_input}"
            )
            return {
                "answer": response.text,
                "tool_calls": [],
                "status": "success",
                "mode": "live_api"
            }

        return {
            "answer": (
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích yêu cầu bằng các quy tắc từ khóa đơn giản."""
        text = user_input.lower()
        faq_keywords = ("bảo hành", "đổi trả", "chính sách")
        action_keywords = (
            "khiếu nại", "bị lỗi", "hỏng hóc", "không hoạt động",
            "ticket", "phản ánh", "sự cố", "hỗ trợ", "sửa chữa"
        )
        has_action = any(keyword in text for keyword in action_keywords)
        has_faq = any(keyword in text for keyword in faq_keywords)
        catalog_request_keywords = (
            "xe điện", "xe vinfast", "du lịch", "vinpearl", "resort",
            "giá", "tìm", "xem", "sản phẩm", "bao nhiêu"
        )
        has_catalog = any(keyword in text for keyword in catalog_request_keywords)
        if has_action and not any(
            keyword in text for keyword in ("giá", "tìm", "xem", "bao nhiêu", "sản phẩm")
        ):
            has_catalog = False
        if has_faq and not has_action:
            has_catalog = False

        category = "du_lich" if any(
            keyword in text for keyword in ("du lịch", "vinpearl", "resort")
        ) else "xe_dien"
        max_price = 999999999999
        price_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|trieu|tr|m)\b",
            text
        )
        if price_match:
            amount = float(price_match.group(1).replace(",", "."))
            unit = price_match.group(2)
            multiplier = 1_000_000_000 if unit in ("tỷ", "ty") else 1_000_000
            max_price = int(amount * multiplier)
        else:
            price_match = re.search(
                r"(?:dưới|tối đa|giá)\s*(\d[\d.,]*)",
                text
            )
            if price_match:
                digits = re.sub(r"\D", "", price_match.group(1))
                if digits:
                    max_price = int(digits)

        name_match = re.search(
            r"(?:tôi\s+tên\s+là|tôi\s+tên|tên\s+tôi\s+là|tôi\s+là)\s+"
            r"([^,.;]+)",
            user_input,
            re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng ẩn danh"
        issue_description = user_input
        if name_match:
            issue_description = re.sub(
                r"(?:tôi\s+tên\s+là|tôi\s+tên|tên\s+tôi\s+là|tôi\s+là)\s+"
                r"[^,.;]+[,.;]?\s*",
                "",
                user_input,
                count=1,
                flags=re.IGNORECASE
            ).strip()
        if not issue_description:
            issue_description = user_input

        priority = "medium"
        if any(keyword in text for keyword in ("không gấp", "ít nghiêm trọng")):
            priority = "low"
        elif any(keyword in text for keyword in ("khẩn cấp", "nghiêm trọng", "gấp", "urgent")):
            priority = "high"

        return {
            "needs_catalog": has_catalog,
            "needs_ticket": has_action,
            "is_faq": has_faq and not has_action,
            "catalog_args": {"category": category, "max_price": max_price},
            "ticket_args": {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }
        }

    def _execute_step(
        self,
        user_input: str,
        intents: Dict[str, Any],
        iteration: int
    ) -> Dict[str, Any]:
        """Thực thi một bước của Agent."""
        if intents["is_faq"]:
            text = user_input.lower()
            if "bảo hành pin" in text:
                observation = (
                    "Chính sách bảo hành pin xe điện VinFast: pin được bảo hành "
                    "10 năm hoặc 200.000 km (tùy điều kiện nào đến trước). "
                    "Để biết thêm chi tiết, vui lòng liên hệ hotline 1900-23-23-89."
                )
            elif "đổi trả" in text:
                observation = (
                    "Chính sách đổi trả phụ thuộc vào sản phẩm và điều kiện mua hàng. "
                    "Vui lòng liên hệ hotline 1900-23-23-89 để được xác nhận."
                )
            elif "bảo hành" in text:
                observation = (
                    "Thời hạn bảo hành phụ thuộc vào sản phẩm và hạng mục cần bảo hành. "
                    "Vui lòng liên hệ hotline 1900-23-23-89 để được xác nhận."
                )
            else:
                observation = (
                    "Vui lòng cung cấp thêm chi tiết về chính sách bạn muốn hỏi hoặc "
                    "liên hệ hotline 1900-23-23-89 để được hỗ trợ."
                )
            self.trace.append({
                "iteration": iteration,
                "thought": "Đây là câu hỏi FAQ, có thể trả lời trực tiếp.",
                "action": None,
                "observation": observation
            })
            return {"faq_answer": observation}

        tool_name = None
        args: Dict[str, Any] = {}
        if intents["needs_catalog"] and not any(
            step.get("action", {}).get("name") == "search_product_catalog"
            for step in self.trace
            if isinstance(step.get("action"), dict)
        ):
            tool_name = "search_product_catalog"
            args = intents["catalog_args"]
        elif intents["needs_ticket"] and not any(
            step.get("action", {}).get("name") == "submit_support_ticket"
            for step in self.trace
            if isinstance(step.get("action"), dict)
        ):
            tool_name = "submit_support_ticket"
            args = intents["ticket_args"]

        if tool_name is None:
            observation = "Không cần gọi tool cho yêu cầu này."
            self.trace.append({
                "iteration": iteration,
                "thought": "Yêu cầu không cần tra cứu hoặc tạo ticket.",
                "action": None,
                "observation": observation
            })
            return {"faq_answer": observation}

        result = TOOL_MAP[tool_name](**args)
        action = {"name": tool_name, "args": args}
        self.trace.append({
            "iteration": iteration,
            "thought": f"Cần gọi {tool_name} để xử lý yêu cầu.",
            "action": action,
            "observation": result
        })
        return {"tool_name": tool_name, "result": result}

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO 3: Phân tích intent từ user_input
        #   - Xác định cần gọi tool nào (catalog? ticket? cả hai? FAQ?)
        #   - Gợi ý: Dùng keyword matching hoặc regex

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        #   - Iteration 1: Gọi tool #1 nếu cần (search_product_catalog)
        #   - Iteration 2: Gọi tool #2 nếu cần (submit_support_ticket)
        #   - Iteration 3+: Tổng hợp Final Answer từ trace
        #   - Lưu mỗi bước vào self.trace

        intents = self._detect_intents(user_input)
        self.trace.append({"step": "init", "user_input": user_input, "intents": intents})
        results: Dict[str, Any] = {}
        tool_steps = 0

        if intents["is_faq"] or not (intents["needs_catalog"] or intents["needs_ticket"]):
            step_result = self._execute_step(user_input, intents, 1)
            results.update(step_result)
            iterations = 1
        else:
            for iteration in range(1, self.max_iterations + 1):
                step_result = self._execute_step(user_input, intents, iteration)
                if step_result.get("tool_name") is None:
                    break
                tool_steps += 1
                results[step_result["tool_name"]] = step_result["result"]
                if tool_steps >= int(intents["needs_catalog"]) + int(intents["needs_ticket"]):
                    break
            iterations = max(tool_steps, 1)

        if "faq_answer" in results:
            answer = results["faq_answer"]
        else:
            answer_parts = []
            catalog_result = results.get("search_product_catalog")
            if catalog_result is not None:
                if catalog_result and "error" in catalog_result[0]:
                    answer_parts.append("Rất tiếc, hiện không thể truy cập danh mục sản phẩm.")
                elif not catalog_result:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với điều kiện.")
                else:
                    products = []
                    for product in catalog_result:
                        price = f"{product['price_vnd']:,}".replace(",", ".")
                        products.append(
                            f"- {product['name']}: {price} VNĐ — {product['description']}"
                        )
                    answer_parts.append(
                        "Các sản phẩm tìm thấy:\n" + "\n".join(products)
                    )

            ticket_result = results.get("submit_support_ticket")
            if ticket_result is not None:
                answer_parts.append(
                    f"Đã tạo ticket {ticket_result['ticket_id']} cho "
                    f"{ticket_result['customer_name']} với mức ưu tiên "
                    f"{ticket_result['priority']}."
                )
            answer = "\n".join(answer_parts) if answer_parts else (
                "Tôi chưa thể xử lý yêu cầu này. Vui lòng cung cấp thêm thông tin."
            )

        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
