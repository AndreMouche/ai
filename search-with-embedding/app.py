import os

import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        question = request.form["animal"]
        model="text-davinci-003"
        amodel = request.args.get("model")#request.form["model"]
        if amodel:
            #if amodel in 
            model = amodel
        response = {"question":question}
        docs = search_redis(redis_client,question)
        answers =  answer_question(docs,model=model,question=question)
        response["dataset"] = layout_ans(docs)
        response["answer"] = answers
        return render_template("index.html", result=response)
    result = request.args.get("result")
    return render_template("index.html", result=result)

def layout_ans(docs):
    data = []
    for i, article in enumerate(docs):
            score = 1 - float(article.vector_score)
            article.vector_score = round(score ,3)
            num = article.num
            article.title = article.title+f"({num})"
            if len(article.text)>500:
                article.text = article.text[0:500]
            
    return docs 

import tiktoken

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")  

def answer_question(
    docs,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=2500,
    size="ada",
    debug=True,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context="" 
    cur_tokens = 0
    for doc in docs:
        cur_tokens += len(tokenizer.encode(" " + doc.text))
        if cur_tokens > max_len: 
            break
        context+=doc.text
         
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("cur_tokens",cur_tokens)
        print("\n\n")
    try:
       
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \" 抱歉呀，I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )

        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return str(e)


import redis
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField
)

REDIS_HOST =  "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
redis_client.ping()

# Constants
# VECTOR_DIM = len(dataset[0]['content_vec']) # length of the vectors
# VECTOR_NUMBER = len(dataset[0])                 # initial number of vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)

import numpy as np
import pandas as pd
from typing import List
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index",
    vector_field: str = "content_vec",
    return_fields: list = ["title", "url", "text", "vector_score","num"],
    hybrid_fields = "*",
    k: int = 20,
    print_results: bool = False,
) -> List[dict]:

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(input=user_query,
                                            model="text-embedding-ada-002",
                                            )["data"][0]['embedding']

    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

    # perform vector search
    results = redis_client.ft(index_name).search(query, params_dict)
    if print_results:
        print("相关文档如下：")
        for i, article in enumerate(results.docs):
            score = 1 - float(article.vector_score)
            print(f"{i}. {article.title} (Score: {round(score ,3) }),{article.url},{article.text}")
    return results.docs