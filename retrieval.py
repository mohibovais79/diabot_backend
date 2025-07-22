import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent
from sklearn.metrics.pairwise import cosine_similarity


class Retrieval:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.df = None
        self.embedding_model = TextEmbedding(model_name=embedding_model_name)
        self.question_embeddings = None

    def load(self, db_path: str = "data.xlsx"):
        self.df = pd.read_excel(db_path, skiprows=7, header=0, engine="calamine")
        self.question_embeddings = np.array(list(self.embedding_model.embed(list(self.df["Questions"]))))
        print("✅ Model and data loaded successfully.")

    def get_similar_questions(self, query: str):
        query_embedding = list(self.embedding_model.embed([query]))[0]
        similarities = cosine_similarity([query_embedding], self.question_embeddings)[0]

        best_match_index = np.argmax(similarities)

        best_match_question = self.df.loc[best_match_index, "Questions"]
        answer = self.df.loc[best_match_index, "answers"]
        reference = self.df.loc[best_match_index, "refrences"]

        return {
            "matched_question": best_match_question,
            "answer": answer,
            "reference": reference,
            "score": float(similarities[best_match_index]),
        }


async def main():
    retriever = Retrieval("data.xlsx")
    answer = await retriever.get_similar_questions("can my sister get diabetes?")
    print(answer)




if __name__ == "__main__":
    asyncio.run(main())
