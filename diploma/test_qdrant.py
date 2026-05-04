from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(
    url="https://dd00f16b-733b-40e7-a5b5-4e0665cf32d6.eu-west-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.gjHRKcRdiLOaURX8w1aQ_1pSNEA2aNo_ghH7xWR2GvY"
)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ✅ Коллекция: 11765 документов! 🔥
info = client.get_collection("kr_production_cloud")
print(f"📊 Коллекция: {info.points_count} документов")

# ✅ НОВЫЙ API: query вместо query_vector
query = "инфаркт миокарда"
vector = model.encode(query).tolist()
hits = client.query_points(
    collection_name="kr_production_cloud",  # ✅ Имя правильное!
    query=vector,  # ✅ query вместо query_vector
    limit=3,
    score_threshold=0.0
)
print(f"🔍 Найдено: {len(hits.points)}")
for i, h in enumerate(hits.points, 1):
    print(f"  {i}. Score: {h.score:.3f} | {h.payload.get('title', 'No title')}")