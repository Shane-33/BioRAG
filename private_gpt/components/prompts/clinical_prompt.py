"""Clinical trial prompt template for BioRAG."""

from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole


class ClinicalPromptTemplate:
    """Prompt template for clinical trial queries."""

    DEFAULT_SYSTEM_PROMPT = """You are a clinical research assistant specialized in analyzing clinical trial documents, 
eligibility criteria, dosing schemas, and safety information. When answering questions:

1. Focus on clinical trial data, patient eligibility, dosing information, and safety profiles
2. Extract and present eligibility criteria clearly
3. Provide dosing schemas with precise dosages and schedules
4. Highlight safety notes, adverse events, and contraindications
5. Reference specific clinical trial phases, endpoints, and outcomes when available
6. If information is not in the provided context, clearly state that

Use the retrieved context to provide accurate, evidence-based answers about clinical trials."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for clinical mode."""
        return ClinicalPromptTemplate.DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def format_messages(
        messages: list[ChatMessage], context: str | None = None
    ) -> list[ChatMessage]:
        """Format messages with clinical context."""
        formatted_messages: list[ChatMessage] = []

        # Add system prompt
        formatted_messages.append(
            ChatMessage(
                content=ClinicalPromptTemplate.get_system_prompt(),
                role=MessageRole.SYSTEM,
            )
        )

        # Add context if provided
        if context:
            context_message = f"""Context from clinical documents:

{context}

Based on the above context, answer the following question:"""
            # Find the last user message and prepend context
            for msg in messages:
                if msg.role == MessageRole.USER and msg == messages[-1]:
                    formatted_messages.append(
                        ChatMessage(
                            content=f"{context_message}\n\n{msg.content}",
                            role=MessageRole.USER,
                        )
                    )
                else:
                    formatted_messages.append(msg)
        else:
            formatted_messages.extend(messages)

        return formatted_messages

