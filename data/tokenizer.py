from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = list(corpus)
        merges = []
        for i in range(num_merges):
            pairs = defaultdict(int)
            for i in range(len(chars)-1):
                pair = (chars[i], chars[i+1])
                pairs[pair] += 1
            # FIND MOST FREQUENT PAIR TIEBREAK LEXICOGRPHICALLY
            best_pair = None
            best_count = -1
            for pair, count in pairs.items():
                if count > best_count:
                    best_count = count
                    best_pair = pair
                elif pair < best_pair and count == best_count:
                    best_pair = pair

            merges.append([best_pair[0],best_pair[1]])
            pair = best_pair[0]+best_pair[1]

            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars)-1 and chars[i] == best_pair[0] and chars[i+1] == best_pair[1]:
                    new_chars.append(pair)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return merges
            

            
        pass
