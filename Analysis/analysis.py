import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import random

import spacy
import random
import pickle
import pandas as pd

from Trie.trie import Trie, TrieNode
from Trie_with_LDA.trie_with_lda import Trie_with_LDA, Trie_with_LDA_Node, load_models, suggest_words

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def load_test_data():
    """Load raw test articles separated by @delimiter."""

    with open("Dataset/raw_test_set.txt", 'r', encoding='utf8') as f:
        articles = f.read().split('@delimiter')
    return articles


def evaluate_hit_at_k_trie_only(
    articles,
    k=10,
    max_docs=50,
    max_words_per_doc=20,
    check_matrix=None,
    max_prefix_len=6,
    verbose=False,
):
    """Evaluate Hit@K using ONLY Trie (no LDA).

    For each word token, we query the Trie with the prefix and count hits
    without considering LDA context.

    Returns a dict with hit counts per prefix length.
    """

    with open("Trie/Trie.pkl", 'rb') as f:
        trie = pickle.load(f)

    hit_count = 0
    total_queries = 0

    # Track hit counts per prefix length
    hit_counts_by_prefix = {l: 0 for l in range(1, max_prefix_len + 1)}
    total_queries_by_prefix = {l: 0 for l in range(1, max_prefix_len + 1)}

    for doc_i, doc in enumerate(articles):
        if max_docs is not None and doc_i >= max_docs:
            break

        doc = nlp(doc.lower())
        tokens = [t.text for t in doc if t.is_alpha and not t.is_punct and not t.is_space]

        for wi, word in enumerate(tokens):
            if max_words_per_doc is not None and not wi in check_matrix[doc_i]:
                continue

            if not word:
                continue

            for prefix_len in range(1, min(len(word), max_prefix_len) + 1):
                prefix = word[:prefix_len]

                # Get top K suggestions from Trie only
                candidates = trie.topK(prefix, k)
                suggested_words = [w for w, _ in candidates]

                if word in suggested_words[:k]:
                    hit_count += 1
                    hit_counts_by_prefix[prefix_len] += 1

                total_queries += 1
                total_queries_by_prefix[prefix_len] += 1

        if verbose and (doc_i + 1) % 10 == 0:
            print(f"Processed {doc_i + 1} docs, queries so far: {total_queries}")

    hit_rate = hit_count / total_queries if total_queries > 0 else 0.0
    hit_rates_by_prefix = {
        prefix_len: (
            hit_counts_by_prefix[prefix_len] / total_queries_by_prefix[prefix_len]
            if total_queries_by_prefix[prefix_len] > 0
            else 0.0
        )
        for prefix_len in range(1, max_prefix_len + 1)
    }

    results = {
        "k": k,
        "method": "Trie Only",
        "hit_count": hit_count,
        "total_queries": total_queries,
        "hit_rate": hit_rate,
        "hit_counts_by_prefix": hit_counts_by_prefix,
        "total_queries_by_prefix": total_queries_by_prefix,
        "hit_rates_by_prefix": hit_rates_by_prefix,
    }

    return results


def evaluate_hit_at_k_with_lda(
    articles,
    k=10,
    max_docs=50,
    max_words_per_doc=20,
    check_matrix=None,
    max_prefix_len=6,
    verbose=False,
):
    """Evaluate Hit@K using Trie + LDA context.

    For each word token in each article, we generate prefixes and call
    `suggest_words(...)` using the preceding words as context (LDA-aware).
    A hit is counted when the target word appears in the top-K suggestions.

    Returns a dict with hit counts per prefix length.
    """

    lda_model, word_to_id, topic_word_matrix, nlp = load_models()

    with open("Trie_with_LDA/Trie_with_LDA.pkl", 'rb') as f:
        trie_with_lda = pickle.load(f)

    hit_count = 0
    total_queries = 0

    # Track hit counts per prefix length (for plotting / analysis)
    hit_counts_by_prefix = {l: 0 for l in range(1, max_prefix_len + 1)}
    total_queries_by_prefix = {l: 0 for l in range(1, max_prefix_len + 1)}

    for doc_i, doc in enumerate(articles):
        if max_docs is not None and doc_i >= max_docs:
            break

        doc = nlp(doc.lower())
        tokens = [t.text for t in doc if t.is_alpha and not t.is_punct and not t.is_space]

        for wi, word in enumerate(tokens):
            if max_words_per_doc is not None and not wi in check_matrix[doc_i]:
                continue

            if not word:
                continue

            context = " ".join(tokens[:wi])

            for prefix_len in range(1, min(len(word), max_prefix_len) + 1):
                prefix = word[:prefix_len]
                user_input = f"{context} {prefix}".strip()

                suggestions = suggest_words(
                    trie_with_lda,
                    lda_model,
                    word_to_id,
                    topic_word_matrix,
                    nlp,
                    user_input,
                    K=k,
                    alpha=[None, 0.25, 1.75, 2.5, 2.0, 1.75, 0.0],
                )

                suggested_words = [w for w, _ in suggestions]

                if word in suggested_words[:k]:
                    hit_count += 1
                    hit_counts_by_prefix[prefix_len] += 1

                total_queries += 1
                total_queries_by_prefix[prefix_len] += 1

        if verbose and (doc_i + 1) % 10 == 0:
            print(f"Processed {doc_i + 1} docs, queries so far: {total_queries}")

    hit_rate = hit_count / total_queries if total_queries > 0 else 0.0
    hit_rates_by_prefix = {
        prefix_len: (
            hit_counts_by_prefix[prefix_len] / total_queries_by_prefix[prefix_len]
            if total_queries_by_prefix[prefix_len] > 0
            else 0.0
        )
        for prefix_len in range(1, max_prefix_len + 1)
    }

    results = {
        "k": k,
        "method": "Trie + LDA",
        "hit_count": hit_count,
        "total_queries": total_queries,
        "hit_rate": hit_rate,
        "hit_counts_by_prefix": hit_counts_by_prefix,
        "total_queries_by_prefix": total_queries_by_prefix,
        "hit_rates_by_prefix": hit_rates_by_prefix,
    }

    return results


def plot_hit_rate_by_prefix_length(
    hit_rates_by_prefix,
    k=10,
    save_path=None,
    show=True,
):
    """Plot Hit@K vs prefix length.

    Args:
        hit_rates_by_prefix: dict from prefix length -> hit rate.
        k: the K value being plotted.
        save_path: if provided, saves the chart to this path as a PNG.
        show: whether to call plt.show().
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to plot results. Install it with: pip install matplotlib")
        return

    prefix_lengths = sorted(hit_rates_by_prefix.keys())
    rates = [hit_rates_by_prefix[p] for p in prefix_lengths]

    plt.figure(figsize=(9, 4))
    plt.bar(prefix_lengths, rates, width=0.6, color='steelblue', label=f"Hit@{k}")

    plt.xlabel("Prefix length")
    plt.ylabel("Hit rate")
    plt.title(f"Hit@{k} vs Prefix Length")
    plt.xticks(prefix_lengths)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            # In headless environments, show() may fail; ignore.
            pass

    plt.close()


if __name__ == "__main__":
    articles = load_test_data()

    max_words_per_doc = 10
    check_matrix = []
    for doc_i, doc in enumerate(articles):
        doc = nlp(doc.lower())
        tokens = [t.text for t in doc if t.is_alpha and not t.is_punct and not t.is_space]
        check_list = random.sample(range(0, len(tokens)), min(len(tokens), max_words_per_doc))
        check_matrix.append(check_list)
        print(f"Doc {doc_i+1}: {len(tokens)} tokens, checking {len(check_list)} tokens")

    print("="*70)
    print("EVALUATION: Trie Only vs Trie + LDA")
    print("="*70)

    # ===== EVALUATE TRIE ONLY =====
    print("\n[1/2] Evaluating Trie Only...")
    results_trie_only = evaluate_hit_at_k_trie_only(
        articles,
        k=10,
        max_docs=None,
        max_words_per_doc=max_words_per_doc,
        check_matrix=check_matrix,
        max_prefix_len=6,
        verbose=True,
    )

    # ===== EVALUATE TRIE + LDA =====
    print("\n[2/2] Evaluating Trie + LDA...")
    results_with_lda = evaluate_hit_at_k_with_lda(
        articles,
        k=10,
        max_docs=None,
        max_words_per_doc=max_words_per_doc,
        check_matrix=check_matrix,
        max_prefix_len=6,
        verbose=True,
    )

    # ===== TABLE 1: TRIE ONLY =====
    print("\n" + "="*70)
    print("TABLE 1: TRIE ONLY")
    print("="*70)
    print(f"Hit@{results_trie_only['k']}: {results_trie_only['hit_rate']:.4f} ({results_trie_only['hit_count']}/{results_trie_only['total_queries']})\n")

    trie_only_data = []
    for prefix_len in sorted(results_trie_only['hit_rates_by_prefix'].keys()):
        hit_rate = results_trie_only['hit_rates_by_prefix'][prefix_len]
        hit_count = results_trie_only['hit_counts_by_prefix'][prefix_len]
        total = results_trie_only['total_queries_by_prefix'][prefix_len]
        trie_only_data.append({
            'Prefix Length': prefix_len,
            'Hit Count': hit_count,
            'Total Queries': total,
            'Hit Rate': f"{hit_rate:.4f}",
        })

    df_trie_only = pd.DataFrame(trie_only_data)
    print(df_trie_only.to_string(index=False))

    # ===== TABLE 2: TRIE + LDA =====
    print("\n" + "="*70)
    print("TABLE 2: TRIE + LDA")
    print("="*70)
    print(f"Hit@{results_with_lda['k']}: {results_with_lda['hit_rate']:.4f} ({results_with_lda['hit_count']}/{results_with_lda['total_queries']})\n")

    lda_data = []
    for prefix_len in sorted(results_with_lda['hit_rates_by_prefix'].keys()):
        hit_rate = results_with_lda['hit_rates_by_prefix'][prefix_len]
        hit_count = results_with_lda['hit_counts_by_prefix'][prefix_len]
        total = results_with_lda['total_queries_by_prefix'][prefix_len]
        lda_data.append({
            'Prefix Length': prefix_len,
            'Hit Count': hit_count,
            'Total Queries': total,
            'Hit Rate': f"{hit_rate:.4f}",
        })

    df_lda = pd.DataFrame(lda_data)
    print(df_lda.to_string(index=False))

    # ===== TABLE 3: COMPARISON =====
    print("\n" + "="*70)
    print("TABLE 3: COMPARISON (Trie + LDA vs Trie Only)")
    print("="*70)

    comparison_data = []
    for prefix_len in sorted(results_trie_only['hit_rates_by_prefix'].keys()):
        rate_trie_only = results_trie_only['hit_rates_by_prefix'][prefix_len]
        rate_lda = results_with_lda['hit_rates_by_prefix'][prefix_len]
        improvement = rate_lda - rate_trie_only
        improvement_pct = (improvement / rate_trie_only * 100) if rate_trie_only > 0 else 0.0

        comparison_data.append({
            'Prefix Length': prefix_len,
            'Trie Only': f"{rate_trie_only:.4f}",
            'Trie + LDA': f"{rate_lda:.4f}",
            'Improvement': f"{improvement:+.4f}",
            'Improvement %': f"{improvement_pct:+.2f}%",
        })

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    print("\n" + "="*70)
    print("Overall Improvement: Trie + LDA vs Trie Only")
    print("="*70)
    overall_improvement = results_with_lda['hit_rate'] - results_trie_only['hit_rate']
    overall_improvement_pct = (overall_improvement / results_trie_only['hit_rate'] * 100) if results_trie_only['hit_rate'] > 0 else 0.0
    print(f"Trie Only:   {results_trie_only['hit_rate']:.4f}")
    print(f"Trie + LDA:  {results_with_lda['hit_rate']:.4f}")
    print(f"Improvement: {overall_improvement:+.4f} ({overall_improvement_pct:+.2f}%)")

    # ===== PLOT BOTH METHODS =====
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)

    try:
        import matplotlib.pyplot as plt

        prefix_lengths = sorted(results_trie_only['hit_rates_by_prefix'].keys())
        rates_trie_only = [results_trie_only['hit_rates_by_prefix'][p] for p in prefix_lengths]
        rates_lda = [results_with_lda['hit_rates_by_prefix'][p] for p in prefix_lengths]

        plt.figure(figsize=(10, 5))
        plt.plot(prefix_lengths, rates_trie_only, 'o-', linewidth=2, label='Trie Only', color='#FF6B6B')
        plt.plot(prefix_lengths, rates_lda, 's-', linewidth=2, label='Trie + LDA', color='#4ECDC4')

        plt.xlabel("Prefix Length", fontsize=12)
        plt.ylabel("Hit Rate", fontsize=12)
        plt.title("Hit@10 vs Prefix Length: Trie Only vs Trie + LDA", fontsize=14, fontweight='bold')
        plt.xticks(prefix_lengths)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("Analysis/Hit@k_comparison.png", dpi=150)
        print("✓ Saved plot to: hit_at_k_comparison.png")
        plt.show()

    except ImportError:
        print("⚠ matplotlib is required for plotting. Install it with: pip install matplotlib")
