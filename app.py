from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from rag import RAGAgent
from retrieval import Retrieval

retriever: Retrieval | None = None
rag_agent: RAGAgent | None = None


retriever = Retrieval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, rag_agent
    print("🚀 Server starting up, loading model...")

    # 1. Create and load the retriever instance
    retriever = Retrieval()
    retriever.load("data.xlsx")

    # 2. Create the agent instance, passing the loaded retriever to it
    rag_agent = RAGAgent(retriever=retriever)

    yield

    # Cleanup logic can go here if needed
    print("✅ Server shutdown complete.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# 3. Create the API endpoint
@app.get("/ask/")
async def ask_question(query: str):
    return await run_in_threadpool(retriever.get_similar_questions, query)


@app.get("/ask-agent/")
async def ask_agent_question(query: str):
    """
    Receives a user query and returns a detailed response from the RAG agent.
    """
    if rag_agent is None:
        return {"error": "Agent not initialized"}, 503
        
    return await rag_agent.get_agentic_response(query)
