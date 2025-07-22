import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from sklearn.metrics.pairwise import cosine_similarity

from retrieval import Retrieval


class RAGAgent:
    def __init__(self, retriever: Retrieval):
        if retriever is None:
            raise ValueError("A valid retriever instance must be provided.")
        self.retriever = retriever

        # Define the tool as a method of this class
        @tool
        def search_diabetes_info(query: str) -> str:
            """
            Searches a knowledge base for information about diabetes.
            Use this tool ONLY when you need to answer specific questions about diabetes.
            """
            print(f"\n--- TOOL CALLED: search_diabetes_info ---")
            print(f"Query: {query}")
            # This uses the retriever instance stored in the class
            result = self.retriever.get_similar_questions(query)
            return (
                f"Found relevant information with confidence score {result['score']:.2f}:\n"
                f"Answer: {result['answer']}\n"
                f"Reference: {result['reference']}"
            )

        load_dotenv(override=True)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        tools = [search_diabetes_info]

        system_prompt = """
        You are DiaBot, a specialized AI assistant for diabetes and its related inquiries such as food, diet, symptoms, lifestyle and any information that can be related to diabetes indirectly or directly. Your tone must be helpful, empathetic, and clear.

        Your process for EVERY user query is non-negotiable and follows these steps:

        1.  **ALWAYS use the `search_diabetes_info` tool first.** This is your only way to access information. Do not try to answer from memory or general knowledge.
        2.  After the tool runs, you will get a confidence score. This score is critical for your next step.
        3.  **If the confidence score is high (0.60 or above):**
            - Synthesize the retrieved "Answer" into a clear and concise response for the user.
            - Your final answer MUST be based on the information provided by the tool.
        4.  **If the confidence score is low ( below 0.60):**
            - This indicates the user's question is either not in your knowledge base or not related to diabetes.
            - You MUST respond by politely stating that you specialize in diabetes-related questions and cannot provide information on that topic, then recommend they consult a healthcare professional."""

        self.agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
        print("✅ RAG Agent initialized successfully.")

    async def get_agentic_response(self, user_query: str) -> dict:
        """Runs the agentic RAG process asynchronously."""
        print(f"--- Running Agent for query: '{user_query}' ---")
        inputs = {"messages": [("user", user_query)]}
        final_answer = ""
        retrieved_info = None

        async for s in self.agent_executor.astream(inputs, stream_mode="values"):
            last_message = s["messages"][-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_answer = last_message.content
            if isinstance(last_message, ToolMessage):
                previous_message = s["messages"][-2]
                if previous_message.tool_calls:
                    tool_call_query = previous_message.tool_calls[0]["args"]["query"]
                    retrieved_info = self.retriever.get_similar_questions(tool_call_query)

        if retrieved_info:
            return {
                "llm_answer": final_answer,
                "retrieved_answer": retrieved_info.get("answer"),
                "reference": retrieved_info.get("reference"),
                "score": retrieved_info.get("score"),
            }
        else:
            return {"llm_answer": final_answer, "retrieved_answer": None, "reference": None, "score": None}


# 5. Run the Agent
if __name__ == "__main__":
    print("--- Running Agentic RAG with LangGraph and Google Gemini ---")
