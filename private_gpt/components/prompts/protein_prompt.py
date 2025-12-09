"""Protein target prompt template for BioRAG."""

from llama_index.core.llms import ChatMessage, MessageRole


class ProteinPromptTemplate:
    """Prompt template for protein target queries."""

    DEFAULT_SYSTEM_PROMPT = """You are a protein biology expert specialized in protein targets, 
protein-protein interactions, and therapeutic targeting. When answering questions:

1. Identify and describe protein targets in detail
2. Explain protein structure, function, and biological roles
3. Describe how drugs interact with protein targets
4. Explain protein-protein interactions and signaling networks
5. Reference specific protein names, domains, and interaction sites when available
6. If information is not in the provided context, clearly state that

Use the retrieved context to provide accurate, detailed information about protein targets and their therapeutic relevance."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for protein mode."""
        return ProteinPromptTemplate.DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def format_messages(
        messages: list[ChatMessage], context: str | None = None
    ) -> list[ChatMessage]:
        """Format messages with protein context."""
        formatted_messages: list[ChatMessage] = []

        # Add system prompt
        formatted_messages.append(
            ChatMessage(
                content=ProteinPromptTemplate.get_system_prompt(),
                role=MessageRole.SYSTEM,
            )
        )

        # Add context if provided
        if context:
            context_message = f"""Context from protein target documents:

{context}

Based on the above context, answer the following question about protein targets:"""
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

