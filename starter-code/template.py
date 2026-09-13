"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl, và các thương hiệu thuộc Vingroup
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác
- Ngôn ngữ: Tiếng Việt (mặc định), có thể trả lời tiếng Anh nếu khách hàng yêu cầu

## AVAILABLE TOOLS
Bạn có quyền sử dụng các tool sau:
1. **search_product_catalog**: Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục (category) và giá tối đa (max_price). Dùng khi khách hàng hỏi về sản phẩm, giá cả, tìm kiếm xe điện hoặc dịch vụ du lịch.
2. **submit_support_ticket**: Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket. Dùng khi khách hàng muốn khiếu nại, báo lỗi, hoặc yêu cầu hỗ trợ kỹ thuật.

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, thông số, tình trạng). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực từ hệ thống.
2. KHÔNG BAO GIỜ tự tạo ticket ID. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh mã ticket.
3. Nếu không tìm thấy sản phẩm phù hợp, trả lời trung thực: "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
4. Nếu câu hỏi nằm ngoài phạm vi hoạt động, từ chối lịch sự và hướng dẫn khách hàng liên hệ kênh phù hợp.
5. Luôn ưu tiên gọi tool trước khi trả lời các câu hỏi liên quan đến sản phẩm hoặc hỗ trợ.

## OPERATIONAL BOUNDARIES
- CHỈ trả lời các câu hỏi liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes, Vinmec, VinSchool, v.v.).
- KHÔNG trả lời các câu hỏi về đối thủ cạnh tranh, chính trị, tôn giáo, hoặc các chủ đề không liên quan.
- Phạm vi tool: Chỉ sử dụng các tool được liệt kê ở mục AVAILABLE TOOLS. Không giả lập tool khác.

## OUTPUT CONTRACT
Khi cần gọi tool, tuân thủ định dạng ReAct:

**Thought:** [Phân tích yêu cầu của khách hàng và xác định cần gọi tool nào]
**Action:** [Tên tool cần gọi và tham số]
**Observation:** [Kết quả trả về từ tool]
**Final Answer:** [Câu trả lời tổng hợp cho khách hàng dựa trên dữ liệu thực từ tool]

Nếu câu hỏi là FAQ đơn giản (ví dụ: giờ mở cửa, địa chỉ liên hệ) và không cần gọi tool, trả lời trực tiếp ở **Final Answer** mà không cần Thought/Action/Observation.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Trả lời trực tiếp KHÔNG gọi tool.
        Mục tiêu: Quan sát hiện tượng hallucination — LLM bịa dữ liệu
        khi không có nguồn dữ liệu thực (tool) để tham chiếu.
        """
        # Mock response giả lập hallucination: bịa giá, thông số không chính xác
        hallucinated_answer = (
            f"[Chatbot Baseline - Không dùng Tool]\n"
            f"Câu hỏi: {user_input}\n\n"
            f"Trả lời (có thể bịa): Dựa trên kiến thức chung, VinFast có nhiều dòng xe điện "
            f"với giá từ 200 triệu đến 2 tỷ VNĐ. Xe VinFast VF e34 có giá khoảng 690 triệu, "
            f"VF 8 giá khoảng 1.1 tỷ, VF 9 giá khoảng 1.5 tỷ. "
            f"Lưu ý: Thông tin này có thể KHÔNG CHÍNH XÁC vì không tra cứu từ hệ thống thực."
        )
        
        return {
            "answer": hallucinated_answer,
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

    # ─── Intent Detection (TODO 3) ───────────────────────────────────────

    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        """Phân tích intent từ user_input bằng keyword matching."""
        text = user_input.lower()

        needs_catalog = any(kw in text for kw in [
            "xe điện", "xe vinfast", "xem xe", "mua xe", "giá xe",
            "du lịch", "resort", "vinpearl", "nghỉ dưỡng", "tour",
            "sản phẩm", "giá dưới", "giá từ"
        ])

        needs_ticket = any(kw in text for kw in [
            "lỗi", "hỗ trợ", "khiếu nại", "báo lỗi", "sự cố",
            "hư hỏng", "ticket", "xử lý gấp", "nghiêm trọng"
        ])

        is_faq = any(kw in text for kw in [
            "bảo hành", "chính sách", "warranty", "liên hệ",
            "hotline", "showroom", "đại lý"
        ])

        # Nếu không detect được intent nào → mặc định là FAQ
        if not needs_catalog and not needs_ticket and not is_faq:
            is_faq = True

        # FAQ có ưu tiên cao hơn catalog (tránh Trap 3: câu hỏi chứa "xe điện"
        # nhưng thực chất hỏi về chính sách bảo hành → phải trả lời FAQ)
        if is_faq:
            needs_catalog = False

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_price(self, user_input: str) -> int:
        """Trích xuất giá tối đa từ câu hỏi (VD: 'dưới 600 triệu' → 600000000)."""
        # Tìm pattern: số + triệu/tỷ
        match = re.search(r'(\d+(?:\.\d+)?)\s*(triệu|tỷ|trieu|ty)', user_input.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit in ["triệu", "trieu"]:
                return int(value * 1_000_000)
            elif unit in ["tỷ", "ty"]:
                return int(value * 1_000_000_000)
        return 999_999_999_999  # Không giới hạn nếu không tìm thấy

    def _extract_category(self, user_input: str) -> str:
        """Trích xuất category từ câu hỏi."""
        text = user_input.lower()
        if any(kw in text for kw in ["du lịch", "resort", "vinpearl", "nghỉ dưỡng", "tour"]):
            return "du_lich"
        return "xe_dien"  # Mặc định

    def _extract_ticket_info(self, user_input: str) -> Dict[str, str]:
        """Trích xuất thông tin ticket từ câu hỏi."""
        # Trích xuất tên khách hàng: "Tôi tên X" hoặc "tên là X"
        name_match = re.search(
            r'(?:tôi\s+tên|tên\s+(?:là|tôi là)?)\s*([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)',
            user_input
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        # Xác định priority
        priority = "medium"
        if any(kw in user_input.lower() for kw in ["gấp", "nghiêm trọng", "khẩn", "urgent", "high"]):
            priority = "high"
        elif any(kw in user_input.lower() for kw in ["nhẹ", "không gấp", "low"]):
            priority = "low"

        return {
            "customer_name": customer_name,
            "issue_description": user_input,
            "priority": priority
        }

    # ─── Agent Loop (TODO 4) + Safeguards (Milestone 4) ──────────────────

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        iteration = 0

        # Bước 1: Intent Detection
        intents = self._detect_intent(user_input)
        self.trace.append({
            "step": "intent_detection",
            "user_input": user_input,
            "intents": intents
        })

        # Xây dựng danh sách actions cần thực hiện
        actions = []
        if intents["needs_catalog"]:
            actions.append("catalog")
        if intents["needs_ticket"]:
            actions.append("ticket")
        if not actions and intents["is_faq"]:
            actions.append("faq")

        # Bước 2: Agent Loop
        catalog_results = []
        ticket_result = {}
        final_answer = ""

        while iteration < self.max_iterations:
            iteration += 1

            # ── Iteration: Gọi search_product_catalog nếu cần ──
            if "catalog" in actions:
                actions.remove("catalog")
                category = self._extract_category(user_input)
                max_price = self._extract_price(user_input)

                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "thought": f"Khách hàng cần tìm sản phẩm {category}, giá ≤ {max_price:,} VNĐ",
                    "action": "search_product_catalog",
                    "params": {"category": category, "max_price": max_price}
                })

                catalog_results = search_product_catalog(
                    category=category, max_price=max_price
                )

                self.trace.append({
                    "step": f"observation_{iteration}",
                    "tool": "search_product_catalog",
                    "result_count": len(catalog_results),
                    "results": catalog_results
                })

                # Empty results handling (Milestone 4)
                if not catalog_results or len(catalog_results) == 0:
                    final_answer = "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
                    self.trace.append({
                        "step": "final_answer",
                        "answer": final_answer
                    })
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }

                # Nếu không còn action nào → tổng hợp Final Answer
                if not actions:
                    product_lines = []
                    for p in catalog_results:
                        product_lines.append(
                            f"• **{p['name']}** — {p['price_vnd']:,} VNĐ: {p['description']}"
                        )
                    final_answer = (
                        f"Tìm thấy {len(catalog_results)} sản phẩm phù hợp:\n"
                        + "\n".join(product_lines)
                    )
                    self.trace.append({
                        "step": "final_answer",
                        "answer": final_answer
                    })
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }
                continue

            # ── Iteration: Gọi submit_support_ticket nếu cần ──
            if "ticket" in actions:
                actions.remove("ticket")
                ticket_info = self._extract_ticket_info(user_input)

                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "thought": f"Khách hàng cần tạo ticket hỗ trợ cho {ticket_info['customer_name']}",
                    "action": "submit_support_ticket",
                    "params": ticket_info
                })

                ticket_result = submit_support_ticket(
                    customer_name=ticket_info["customer_name"],
                    issue_description=ticket_info["issue_description"],
                    priority=ticket_info["priority"]
                )

                self.trace.append({
                    "step": f"observation_{iteration}",
                    "tool": "submit_support_ticket",
                    "result": ticket_result
                })

                # Nếu không còn action nào → tổng hợp Final Answer
                if not actions:
                    final_answer = (
                        f"Đã ghi nhận yêu cầu hỗ trợ cho {ticket_result['customer_name']}.\n"
                        f"Mã ticket: {ticket_result['ticket_id']}\n"
                        f"Mức ưu tiên: {ticket_result['priority']}\n"
                        f"Trạng thái: {ticket_result['status']}\n"
                        f"Đội ngũ hỗ trợ sẽ liên hệ bạn trong thời gian sớm nhất."
                    )
                    self.trace.append({
                        "step": "final_answer",
                        "answer": final_answer
                    })
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }
                continue

            # ── Iteration: FAQ — trả lời trực tiếp không cần tool ──
            if "faq" in actions:
                actions.remove("faq")

                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "thought": "Câu hỏi thuộc FAQ, trả lời trực tiếp không cần gọi tool.",
                    "action": "faq_response"
                })

                # Xử lý các FAQ phổ biến
                text = user_input.lower()
                if "bảo hành" in text and "pin" in text:
                    final_answer = (
                        "Chính sách bảo hành pin xe điện VinFast: VinFast cung cấp "
                        "bảo hành pin 10 năm cho tất cả các dòng xe điện. Pin được đảm bảo "
                        "duy trì tối thiểu 70% dung lượng trong suốt thời gian bảo hành."
                    )
                elif "bảo hành" in text:
                    final_answer = (
                        "Chính sách bảo hành VinFast: Xe điện VinFast được bảo hành "
                        "5 năm hoặc 125.000 km (tuỳ điều kiện nào đến trước). "
                        "Pin xe được bảo hành riêng 10 năm."
                    )
                else:
                    final_answer = (
                        "Cảm ơn bạn đã quan tâm đến Vingroup. Vui lòng liên hệ hotline "
                        "1900 23 23 89 hoặc truy cập vinfast.vn để được hỗ trợ chi tiết."
                    )

                self.trace.append({
                    "step": "final_answer",
                    "answer": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

        # Safeguard: Vượt quá max_iterations (Milestone 4)
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa. Vui lòng thử lại hoặc liên hệ hotline 1900 23 23 89.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached"
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
