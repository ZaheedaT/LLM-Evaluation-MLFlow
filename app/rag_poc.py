import ast
import os

import chromadb
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.databricks import DatabricksEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import Databricks
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import mlflow
import mlflow.deployments
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance
