import asyncio
import time
from app.services.retrieval_service import RetrievalService

TEST_CASES = [
    {"query": "what are the features of spring boot?", "expected": None},  
    {"query": "what is dependency injection in fastapi?", "expected": "dependencies"},
    {"query": "explain digital certificates and certificate authorities", "expected": "comptia"},
    {"query": "what is a python generator?", "expected": "generator"},
    {"query": "what is sqlalchemy alembic used for?", "expected": "alembic"},
    {"query": "what's the difference between symmetric and asymmetric encryption?", "expected": "comptia"},
    {"query": "how does fastapi middleware work?", "expected": "middleware"},
    {"query": "what is the capital of France?", "expected": None},
    {"query": "how do I bake a chocolate cake?", "expected": None},
    {"query": "whut is dpendency injecshun?", "expected": "dependencies"}, 
    {"query": "how to manage migrations in django", "expected": None},     
]

SCORE_THRESHOLD = 0.25 

SEMAPHORE = asyncio.Semaphore(2)

async def run_query(service: RetrievalService, case: dict):
    query = case["query"]
    expected = case["expected"]
    
    async with SEMAPHORE:  
        start = time.perf_counter()
        nodes = await service.retrieve_parent_context(query)
        latency = (time.perf_counter() - start) * 1000

    top_score = nodes[0].score if nodes else 0.0
    top_files = [n.node.metadata.get("file_name", "") for n in nodes]

    if expected is None:
        ok = top_score < SCORE_THRESHOLD
    else:
        has_file = any(expected.lower() in f.lower() for f in top_files)
        ok = has_file and (top_score >= SCORE_THRESHOLD)

    return {
        "ok": ok,
        "query": query,
        "expected": expected,
        "score": top_score,
        "files": top_files[:3],
        "latency_ms": latency
    }

async def main():
    service = RetrievalService()
    tasks = [run_query(service, case) for case in TEST_CASES]
    results = await asyncio.gather(*tasks)

    passed = sum(1 for r in results if r["ok"])
    
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] ({r['latency_ms']:.1f}ms) \"{r['query']}\"")
        print(f"       score={r['score']:.3f} files={r['files']}")
        if not r["ok"]:
            print(f"       expected match: {r['expected']}")
        print()

    print(f"Results: {passed}/{len(results)} passed.")

if __name__ == "__main__":
    asyncio.run(main())