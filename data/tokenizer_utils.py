from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for number in numbers:
            num = str(number)
            tokens = self._greedy_tokenize(num, vocab)
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self._greedy_tokenize(text, vocab)
        return len(tokens)
        

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self._greedy_tokenize(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)

    def _greedy_tokenize(self, text, vocab):
        tokens = []
        i = 0
        while i < len(text):
            best = None
            for length in range(len(text) - i, 0, -1):
                substr = text[i:i+length]
                if substr in vocab:
                    best = substr
                    break
            if best is None:
                tokens.append(text[i])
                i += 1
            else:
                tokens.append(best)
                i += len(best)
        return tokens
