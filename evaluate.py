from ask import answer, retrieve
import json

# Test questions with the bank we KNOW the answer should come from.
# This lets us measure whether retrieval pulls from the right source.
TEST_CASES = [
    {"question": "What does JPMorgan say about operational risk?", "expected_source": "JPMorgan"},
    {"question": "How does Bank of America manage credit concentrations?", "expected_source": "Bank of America"},
    {"question": "What are JPMorgan's main liquidity risks?", "expected_source": "JPMorgan"},
    {"question": "How does Bank of America measure derivative credit exposure?", "expected_source": "Bank of America"},
    {"question": "What capital requirements does JPMorgan discuss?", "expected_source": "JPMorgan"},
]

def evaluate():
    results = []
    retrieval_hits = 0

    for case in TEST_CASES:
        q = case["question"]
        expected = case["expected_source"]

        # 1. RETRIEVAL QUALITY: did we pull chunks from the expected bank?
        hits = retrieve(q, top_k=5)
        retrieved_sources = [src for _, src, _ in hits]
        top_source_correct = retrieved_sources[0] == expected
        expected_in_top5 = expected in retrieved_sources

        # 2. GENERATION: get the actual answer
        ans, _ = answer(q)

        # 3. GROUNDING CHECK: does the answer cite the expected source?
        cites_source = expected.lower() in ans.lower()

        if expected_in_top5:
            retrieval_hits += 1

        results.append({
            "question": q,
            "expected_source": expected,
            "top_retrieved": retrieved_sources[0],
            "top_source_correct": top_source_correct,
            "expected_in_top5": expected_in_top5,
            "answer_cites_source": cites_source,
        })
        print(f"Q: {q}")
        print(f"   Expected: {expected} | Top retrieved: {retrieved_sources[0]} | "
              f"In top 5: {expected_in_top5} | Cites source: {cites_source}")
        print()

    # Summary metrics
    n = len(TEST_CASES)
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print(f"  Retrieval accuracy (expected source in top 5): {retrieval_hits}/{n} = {retrieval_hits/n:.0%}")
    top1 = sum(r["top_source_correct"] for r in results)
    print(f"  Top-1 retrieval accuracy: {top1}/{n} = {top1/n:.0%}")
    grounded = sum(r["answer_cites_source"] for r in results)
    print(f"  Answers citing expected source: {grounded}/{n} = {grounded/n:.0%}")

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed results to evaluation_results.json")

if __name__ == "__main__":
    evaluate()