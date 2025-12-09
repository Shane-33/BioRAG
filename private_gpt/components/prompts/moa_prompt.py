"""Mechanism of Action (MOA) prompt template for BioRAG."""

from llama_index.core.llms import ChatMessage, MessageRole


class MOAPromptTemplate:
    """Prompt template for mechanism of action queries."""

    DEFAULT_SYSTEM_PROMPT = """You are a pharmacology expert specialized in mechanisms of action, 
protein targets, and drug interactions. When answering questions:

1. Explain mechanisms of action in detail, including molecular pathways
2. Identify and describe protein targets, receptors, enzymes, and signaling pathways
3. Explain how drugs interact with their targets at the molecular level
4. Describe downstream effects and biological consequences
5. Reference specific proteins, pathways, and molecular interactions when available
6. If information is not in the provided context, clearly state that

Use the retrieved context to provide accurate, detailed explanations of drug mechanisms and protein interactions."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for MOA mode."""
        return MOAPromptTemplate.DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def format_messages(
        messages: list[ChatMessage], context: str | None = None
    ) -> list[ChatMessage]:
        """Format messages with MOA context."""
        formatted_messages: list[ChatMessage] = []

        # Add system prompt
        formatted_messages.append(
            ChatMessage(
                content=MOAPromptTemplate.get_system_prompt(),
                role=MessageRole.SYSTEM,
            )
        )

        # Add context if provided
        if context:
            context_message = f"""Context from mechanism of action documents:

{context}

Based on the above context, answer the following question about mechanisms of action:"""
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

