# verify_day1.py
# Author: Rajesh (Junior AI Engineer)
# Ticket: SAGE-001
# Purpose: Verify ChromaDB and embeddings working
#          Run before raising PR

import sys
sys.path.append(".")

print("=" * 55)
print("SAGE-001 — Day 1 Verification")
print("ChromaDB + Embeddings")
print("=" * 55)

# TEST 1 — Embedding model loads
print("\n[TEST 1] Loading embedding model...")
try:
    from rag_pipeline.embeddings import (
        embed_text,
        embed_documents,
        EMBEDDING_DIMENSIONS
    )
    vector = embed_text("test sentence for verification")
    assert len(vector) == EMBEDDING_DIMENSIONS, \
        f"Expected {EMBEDDING_DIMENSIONS} dims, got {len(vector)}"
    print(f"  Dimensions: {len(vector)}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# TEST 2 — ChromaDB connects
print("\n[TEST 2] Connecting to ChromaDB...")
try:
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# TEST 3 — Create collection
print("\n[TEST 3] Creating test collection...")
try:
    try:
        client.delete_collection("sage_verify_test")
    except Exception:
        pass
    collection = client.create_collection(
        name="sage_verify_test",
        metadata={"hnsw:space": "cosine"}
    )
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# TEST 4 — Store vectors
print("\n[TEST 4] Storing vectors in ChromaDB...")
try:
    sample_texts = [
        "The penalty clause states 2 percent per week for late delivery.",
        "Lead time for ECU parts is 45 days from purchase order.",
        "Safety stock for A-class parts is 30 days minimum.",
        "WMAPE target is less than 15 percent at model level.",
        "Forecast variance above 40 percent escalate to Supply Chain Director."
    ]
    vectors = embed_documents(sample_texts)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(sample_texts))],
        embeddings=vectors,
        documents=sample_texts,
        metadatas=[
            {"source": "supplier_contract.txt", "chunk_index": 0},
            {"source": "supplier_contract.txt", "chunk_index": 1},
            {"source": "sop_demand_planning.txt", "chunk_index": 0},
            {"source": "sop_demand_planning.txt", "chunk_index": 1},
            {"source": "sop_demand_planning.txt", "chunk_index": 2},
        ]
    )
    print(f"  Stored: {len(sample_texts)} vectors")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# TEST 5 — Semantic search
print("\n[TEST 5] Semantic search on ChromaDB...")
try:
    query = "what is the penalty for late delivery"
    query_vector = embed_text(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )
    top_doc = results["documents"][0][0]
    top_source = results["metadatas"][0][0]["source"]
    top_similarity = 1 - results["distances"][0][0]
    print(f"  Query: '{query}'")
    print(f"  Top result similarity: {top_similarity:.3f}")
    print(f"  Top result: {top_doc[:60]}...")
    print(f"  Source: {top_source}")
    assert top_similarity > 0.5, \
        f"Similarity too low: {top_similarity}"
    assert "penalty" in top_doc.lower(), \
        "Wrong result returned"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)
finally:
    try:
        client.delete_collection("sage_verify_test")
    except Exception:
        pass

print("\n" + "=" * 55)
print("ALL 5 TESTS PASSED")
print("SAGE-001 acceptance criteria met.")
print("Ready to raise PR.")
print("=" * 55)