import heapq


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


if __name__ == "__main__":
    trie = Trie()

    words = [
        "apple", "app", "application", "app",
        "apply", "apple", "apple", "apt"
    ]

    for w in words:
        trie.insert(w)

    [print(i) for i in trie.topK("app", 3)]
