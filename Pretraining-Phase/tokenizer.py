from pathlib import Path

class byteTokenizer:
    def __init__(self, file_path: Path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Create sorted list of unique characters
        self.chars = sorted(set(self.text))
        print("Unique characters:", self.chars)
        print(f"Length of unique chars: {len(self.chars)}")
        
        # Mapping: char -> int and int -> char
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}


    @property
    def vocab_size(self)-> int:
        return len(self.chars)
        
        
    def encode(self, s: str) -> list[int]:
        """Convert string to list of integers"""
        return [self.char_to_int[c] for c in s]

    def decode(self, l: list[int]) -> str:
        """Convert list of integers back to string"""
        return ''.join([self.int_to_char[i] for i in l])