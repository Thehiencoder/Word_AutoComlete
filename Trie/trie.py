import heapq
import pickle


class TrieNode:
    def __init__(self):
        self.child = {}
        self.is_end = False
        self.freq = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.child:
                cur.child[c] = TrieNode()
            cur = cur.child[c]

        cur.is_end = True
        cur.freq += 1

    def _dfs(self, node, cur_word, K, heap):
        if node.is_end:
            if len(heap) < K:
                heapq.heappush(heap, (node.freq, cur_word))
            elif node.freq > heap[0][0]:
                heapq.heapreplace(heap, (node.freq, cur_word))

        for c, nxt in node.child.items():
            self._dfs(nxt, cur_word + c, K, heap)

    def topK(self, prefix, K):
        cur = self.root
        for c in prefix:
            if c not in cur.child:
                return []
            cur = cur.child[c]

        heap = []
        self._dfs(cur, prefix, K, heap)

        # trả về danh sách (word, freq) giảm dần theo freq
        return sorted(
            [(word, freq) for freq, word in heap],
            key=lambda x: -x[1]
        )


def build_trie():
    trie = Trie()

    with open("Dataset/training_data_for_Trie.pkl", 'rb') as f:
        tokenized_articles = pickle.load(f)

    words = [word for doc in tokenized_articles for word in doc]

    for w in words:
        trie.insert(w)

    return trie


if __name__ == "__main__":
    # Save
    trie = build_trie()

    # filename = "Trie/Trie.pkl"
    # with open(filename, 'wb') as f:
    #     pickle.dump(trie, f)

    # Load
    # with open("Trie/Trie.pkl", 'rb') as f:
    #     trie = pickle.load(f)

    topK = trie.topK("app", 10)
    [print(i) for i in topK]