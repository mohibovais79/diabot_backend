from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from retrieval import Retrieval

retriever = Retrieval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting up, loading model...")
    retriever.load()
    yield
    print(" shutting down.")


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
