from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from flipkart.config import Config
from typing import List
import logging

logging.basicConfig()


class _InstrumentedMultiQueryRetriever(MultiQueryRetriever):
    """MultiQueryRetriever that writes generated queries into a capture list for UI display."""

    capture: list = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        queries = self.generate_queries(query, run_manager)
        self.capture.clear()
        self.capture.extend(queries)
        return self.retrieve_documents(queries, run_manager)


def build_flipkart_retriever_tool(retriever):

    @tool
    def flipkart_retriever_tool(query: str) -> str:
        """
        Retrieve top product reviews and metadata related to the user query.
        Returns product name, rating, summary, and full review for each result.
        """
        docs = retriever.invoke(query)
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            entry = (
                f"[Source {i}]\n"
                f"Product: {meta.get('product_name', 'Unknown')}\n"
                f"Rating: {meta.get('rating', 'N/A')}/5\n"
                f"Summary: {meta.get('summary', 'N/A')}\n"
                f"Review: {doc.page_content}"
            )
            results.append(entry)
        return "\n\n---\n\n".join(results)

    return flipkart_retriever_tool


class RAGAgentBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = init_chat_model(Config.RAG_MODEL)
        # Separate lightweight model for query expansion — avoids burning RAG model tokens
        self.retriever_model = init_chat_model(Config.RETRIEVER_MODEL)

    def build_agent(self):

        # Base retriever — k=3 to keep token payload manageable on free-tier Groq limits
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Instrumented MultiQueryRetriever — exposes .capture list for UI query expansion display
        retriever = _InstrumentedMultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.retriever_model
        )
        self.query_capture = retriever.capture  # reference shared by the UI

        flipkart_tool = build_flipkart_retriever_tool(retriever)

        agent = create_agent(
            model=self.model,
            tools=[flipkart_tool],
            system_prompt="""You are a world-class AI shopping assistant embedded in an e-commerce platform. Your role is to help customers make confident, informed purchase decisions using ONLY the product data retrieved from the store's database.

RESPONSE FORMAT — always follow this structure:

1. **Response**: Address the customer's question clearly and concisely in 1-2 sentences.
2. **Product Highlights**: Use bullet points to present key features, pros/cons, or comparisons drawn from the retrieved reviews.
3. **Customer Verdict**: Summarize what real customers are saying (ratings, sentiments). Quote brief snippets where impactful.


STRICT RULES:
- ALWAYS call `flipkart_retriever_tool` first. Never answer from your own knowledge.
- Only use information present in the retrieved results. Do NOT invent specs, prices, or reviews.
- If the retrieved data does not answer the question, respond: "I don't have enough product data to answer that confidently. Please contact our support team for assistance."
- Decline all off-topic questions politely: "I'm specialized for product queries only. How can I help you find the right product today?"
- Use markdown formatting (bold, bullets, headers) to make responses scannable and professional.
- Keep responses concise but complete — aim for quality over length.""",
            checkpointer=InMemorySaver(),
            middleware=[
                SummarizationMiddleware(
                    model=self.model,
                    trigger=("messages", 10),
                    keep=("messages", 4),
                )
            ],
        )

        return agent
