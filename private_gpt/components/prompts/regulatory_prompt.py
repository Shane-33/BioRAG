"""Regulatory prompt template for BioRAG."""

from llama_index.core.llms import ChatMessage, MessageRole


class RegulatoryPromptTemplate:
    """Prompt template for regulatory queries."""

    DEFAULT_SYSTEM_PROMPT = """You are a regulatory affairs expert specialized in FDA/EMA approvals, 
indications, and regulatory documentation. When answering questions:

1. Provide information about FDA/EMA approvals and regulatory status
2. List approved indications and therapeutic uses
3. Reference specific regulatory documents, guidelines, and requirements
4. Explain regulatory pathways and approval processes
5. Include information about clinical trial phases and regulatory milestones
6. If information is not in the provided context, clearly state that

Use the retrieved context to provide accurate, up-to-date regulatory information."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for regulatory mode."""
        return RegulatoryPromptTemplate.DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def format_messages(
        messages: list[ChatMessage], context: str | None = None
    ) -> list[ChatMessage]:
        """Format messages with regulatory context."""
        formatted_messages: list[ChatMessage] = []

        # Add system prompt
        formatted_messages.append(
            ChatMessage(
                content=RegulatoryPromptTemplate.get_system_prompt(),
                role=MessageRole.SYSTEM,
            )
        )

        # Add context if provided
        if context:
            context_message = f"""Context from regulatory documents:

{context}

Based on the above context, answer the following question about regulatory information:"""
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

