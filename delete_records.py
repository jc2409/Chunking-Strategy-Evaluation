# pip install "pinecone[grpc]"
import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()   

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/manage-data/target-an-index
index = pc.Index(host=os.getenv("PINECONE_INDEX_HOST"))

index.delete(delete_all=True, namespace='late')
index.delete(delete_all=True, namespace='semantic')
index.delete(delete_all=True, namespace='recursive')
index.delete(delete_all=True, namespace='fixed')