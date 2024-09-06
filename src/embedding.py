import re, os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

groq_api_key = 'YOUR_GROQ_API_KEY'
headers = {
    'Authorization': f'Bearer {groq_api_key}',
    'Content-Type': 'application/json'
}