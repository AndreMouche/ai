

import tiktoken

root_path="/home/tidb/shirly/docs-cn-master"
url_prefix="https://github.com/pingcap/docs-cn/blob/master/"

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
max_tokens = 1000


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    #sentences = text.split('. ')
    sentences = text.split('。')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
       # print("tokens so far",tokens_so_far)

    return chunks
    

shortened = []
rows = 0

# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.
import os
import openai

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    #print(openai.api_key)
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")

# connect to redis
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
#REDIS_HOST = "120.92.92.145"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
redis_client.ping()
print("connect to redis ok")


# Constants
VECTOR_DIM = 1536 # length of the vectors
VECTOR_NUMBER = 5                # initial number of vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
#print(VECTOR_DIM,VECTOR_NUMBER)
# Define RediSearch fields for each of the columns in the dataset
title = TextField(name="title")
url = TextField(name="url")
text = TextField(name="text")
text_embedding = VectorField("content_vec",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
fields = [title, url, text, text_embedding]
# Check if index exists
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
)

import sys
import numpy as np
import pandas as pd

def get_kb_summary(text):
    #print("start parse text")
    summary = ""
    for content in text.split("#"):
      #print("meet ",content)
      lines = content.splitlines()
      title = ""
      data = ""
      for l in lines:
         if len(l.strip()) == 0:
            continue
         if len(title) == 0:
            title = l.strip() 
            continue
         data += l.strip()
      if len(title) == 0:
          continue
      if len(summary) == 0:
          summary = title.strip() 
          continue 
      if title.find("修复版本")!=-1 :
          summary = summary + " "+title + data
      if title.find("Affected Versions") != -1 :
          summary = summary +" "+ title + data
      #print("cur title",title,"cur content",data)
    return summary

def embedding_and_save_dir(client:redis.Redis,dir_path,base_url):
    print("start import ",dir_path)
    for file in os.listdir(dir_path):
        cur_path = os.path.join(dir_path,file)
        if os.path.isdir(cur_path):
            embedding_and_save_dir(client,cur_path,os.path.join(base_url,file))
            #break
            continue
        if file.endswith(".md") == False:
            print("ignore",file)
            continue
        # Open the file and read the text
        with open(os.path.join(dir_path,file), "r") as f:
            text = f.read()
            txt = text.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ').replace('  ', ' ')
            summary = get_kb_summary(txt)
            print("get summary for ",file,summary)
            number = 0
            cur_url = os.path.join(base_url,file)
            key= f"{PREFIX}:" + cur_url+ f"{number}"
            old = client.hget(key,"title")
            print(old)
            if old:
                print("already have")
                break
            content_embeddings = openai.Embedding.create(input=summary, engine='text-embedding-ada-002')['data'][0]['embedding']
            # # # replace list of floats with byte vectors
            rcontent_embedding = np.array(content_embeddings, dtype=np.float32).tobytes()
            cur_doc = {
                "num": number,
                "title":file,
                "text":txt,
                "url":cur_url,
                "content_vec":rcontent_embedding,
            }
            client.hset(key, mapping = cur_doc) 

    print("finished import docs summary from dir",dir_path)

def embedding_and_save_dir_content(client:redis.Redis,dir_path,base_url):
    print("start import ",dir_path)
    for file in os.listdir(dir_path):
        cur_path = os.path.join(dir_path,file)
        if os.path.isdir(cur_path):
            embedding_and_save_dir_content(client,cur_path,os.path.join(base_url,file))
            #break
            continue
        if file.endswith(".md") == False:
            print("ignore",file)
            continue
        # Open the file and read the text
        with open(os.path.join(dir_path,file), "r") as f:
            text = f.read()
            txt = text.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ').replace('  ', ' ')
            chunks = split_into_many(txt,max_tokens)
            number = 0
            cur_url = os.path.join(base_url,file)
            print("met doc",file,cur_url)
            for c in chunks:
                number += 1
                key= f"{PREFIX}:" + cur_url+ f"{number}"
                old = client.hget(key,"title")
                print(old)
                if old:
                    print("already have")
                    break
                c=c+file
                content_embeddings = openai.Embedding.create(input=c, engine='text-embedding-ada-002')['data'][0]['embedding']
                # # # replace list of floats with byte vectors
                rcontent_embedding = np.array(content_embeddings, dtype=np.float32).tobytes()
                cur_doc = {
                 "num": number,
                 "title":file,
                 "text":c,
                 "url":cur_url,
                 "content_vec":rcontent_embedding,
                }
                client.hset(key, mapping = cur_doc) 

    print("finished import docs from dir",dir_path)
    

embedding_and_save_dir(redis_client,root_path,url_prefix)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")
