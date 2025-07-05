def get_embedding(prompt, model="nomic-embed-text"):

  import requests
  url="http://localhost:11434/api/embeddings"
  data = {
    "model": model,
    "prompt": prompt,
  }
  response =  requests.post(url, json=data)
  response.raise_for_status()
  
  return response.json().get("embedding", None)


def get_opensearch_client(host, port):
  from opensearchpy import OpenSearch


  client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    retry_on_timeout = True,
    max_retries=3,
    timeout=30
  )

  if client.ping():
    print("connected to Opensearch")

  return client

if __name__ == "__main__":
  # prompt = "what is celebrated on july 4"
  # embedding = get_embedding(prompt)
  # print(f"Embadding for prompt=> {prompt} : {embedding}")
  get_opensearch_client("localhost", 9200)
  sample_embedding = get_embedding("Sample text for dimension detection")
  dimension = len(sample_embedding)
  print(f"Using embedding dimension: {dimension}")

