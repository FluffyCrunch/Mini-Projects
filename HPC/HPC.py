import heapq
from collections import defaultdict
import cupy as cp

# Step 1: Build frequency dictionary
def calculate_frequencies(data):
    freq = defaultdict(int)
    for ch in data:
        freq[ch] += 1
    return freq

# Step 2: Build Huffman Tree
class Node:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [Node(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

# Step 3: Generate Huffman Codes
def generate_codes(node, current_code="", code_dict={}):
    if node is None:
        return

    if node.char is not None:
        code_dict[node.char] = current_code
        return

    generate_codes(node.left, current_code + "0", code_dict)
    generate_codes(node.right, current_code + "1", code_dict)

    return code_dict

# Step 4: Encode the input using generated codes
def encode_data(data, code_dict):
    return ''.join([code_dict[ch] for ch in data])

# Step 5: Convert to GPU array for processing (for demonstration)
def to_gpu_binary(encoded_str):
    binary_array = cp.array([int(bit) for bit in encoded_str], dtype=cp.int8)
    return binary_array

# Main Function
if __name__ == "__main__":
    input_data = "this is an example for huffman encoding"

    # CPU side Huffman steps
    freq_dict = calculate_frequencies(input_data)
    huffman_tree = build_huffman_tree(freq_dict)
    codes = generate_codes(huffman_tree)
    encoded_str = encode_data(input_data, codes)

    # GPU Conversion
    gpu_encoded = to_gpu_binary(encoded_str)

    print("Original Text:", input_data)
    print("Huffman Codes:", codes)
    print("Encoded Binary String (First 100 bits):", gpu_encoded[:100])
    print("Total Bits Encoded:", gpu_encoded.size)
