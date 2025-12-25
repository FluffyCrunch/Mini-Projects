# High Performance Computing - Huffman Encoding with GPU Acceleration

A High Performance Computing project that implements Huffman encoding algorithm with GPU acceleration using CuPy. The project demonstrates parallel processing capabilities by offloading binary encoding operations to GPU for improved performance.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithm](#algorithm)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements the Huffman encoding algorithm, a lossless data compression technique, with GPU acceleration. The algorithm:
- Analyzes character frequencies in input data
- Builds an optimal binary tree (Huffman tree)
- Generates variable-length codes for each character
- Encodes data using these codes
- Utilizes GPU (via CuPy) for efficient binary array processing

## ✨ Features

- **Huffman Encoding**: Classic lossless compression algorithm
- **GPU Acceleration**: Uses CuPy for GPU-accelerated binary processing
- **Optimal Compression**: Variable-length codes based on frequency analysis
- **Tree-based Structure**: Efficient binary tree implementation
- **Memory Efficient**: Heap-based tree construction

## 🔬 Algorithm

### Huffman Encoding Steps:

1. **Frequency Calculation**: Count occurrences of each character
2. **Tree Construction**: Build Huffman tree using priority queue (heap)
3. **Code Generation**: Traverse tree to generate binary codes
4. **Encoding**: Convert input data to binary string using codes
5. **GPU Processing**: Transfer binary data to GPU for parallel operations

### Time Complexity:
- Frequency calculation: O(n)
- Tree building: O(k log k) where k is unique characters
- Encoding: O(n)
- Overall: O(n + k log k)

## 📦 Requirements

- Python 3.7+
- CuPy (for GPU acceleration)
- NumPy (for array operations)
- CUDA-capable GPU (optional, for GPU acceleration)

**Note**: If you don't have a GPU, you can modify the code to use NumPy instead of CuPy.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/FluffyCrunch/Mini-Projects.git
cd Mini-Projects/HPC
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
# For GPU support (requires CUDA)
pip install cupy-cuda11x  # Replace with your CUDA version

# For CPU-only (fallback)
pip install numpy
```

**CUDA Version Compatibility:**
- CuPy requires CUDA toolkit
- Check your CUDA version: `nvcc --version`
- Install matching CuPy version:
  - CUDA 11.x: `pip install cupy-cuda11x`
  - CUDA 12.x: `pip install cupy-cuda12x`
  - CPU only: Use NumPy instead

## 🚀 Usage

### Basic Usage

```python
python HPC.py
```

This will:
1. Process the example string: "this is an example for huffman encoding"
2. Calculate character frequencies
3. Build Huffman tree
4. Generate encoding codes
5. Encode the data
6. Convert to GPU array
7. Display results

### Using as a Module

```python
from HPC import (
    calculate_frequencies,
    build_huffman_tree,
    generate_codes,
    encode_data,
    to_gpu_binary
)

# Your input data
data = "your text here"

# Encode
freq = calculate_frequencies(data)
tree = build_huffman_tree(freq)
codes = generate_codes(tree)
encoded = encode_data(data, codes)
gpu_array = to_gpu_binary(encoded)

print("Codes:", codes)
print("Encoded size:", gpu_array.size)
```

### CPU-Only Version

If you don't have GPU support, modify the code:

```python
# Replace:
import cupy as cp
# With:
import numpy as np

# Replace:
cp.array([...])
# With:
np.array([...])
```

## 📁 Project Structure

```
HPC/
├── HPC.py              # Main implementation file
├── README.md           # This file
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore rules
```

## 🔬 How It Works

### Step 1: Frequency Calculation
```python
freq_dict = calculate_frequencies("example")
# Result: {'e': 2, 'x': 1, 'a': 1, 'm': 1, 'p': 1, 'l': 1}
```

### Step 2: Huffman Tree Construction
- Create leaf nodes for each character with its frequency
- Use min-heap to always merge two nodes with lowest frequencies
- Build tree bottom-up until single root node remains

### Step 3: Code Generation
- Traverse tree: left = '0', right = '1'
- Generate unique binary codes for each character
- More frequent characters get shorter codes

### Step 4: Encoding
- Replace each character with its binary code
- Concatenate all codes into single binary string

### Step 5: GPU Processing
- Convert binary string to array of integers (0s and 1s)
- Transfer to GPU memory using CuPy
- Enables parallel processing of binary data

## ⚡ Performance

### Benefits of GPU Acceleration:
- **Parallel Processing**: Multiple operations simultaneously
- **Large Data**: Efficient for processing large binary arrays
- **Memory Bandwidth**: Higher memory throughput on GPU

### When to Use GPU:
- Large input data (>1MB)
- Batch processing multiple files
- Real-time compression requirements

### CPU vs GPU:
- **CPU**: Better for small data, simpler setup
- **GPU**: Better for large data, requires CUDA

## 📊 Example Output

```
Original Text: this is an example for huffman encoding
Huffman Codes: {' ': '00', 'a': '0100', 'c': '01010', 'd': '01011', ...}
Encoded Binary String (First 100 bits): [1 0 1 1 0 0 0 0 1 0 ...]
Total Bits Encoded: 156
```

## 🔮 Future Improvements

- [ ] Add decoding functionality
- [ ] Implement file compression/decompression
- [ ] Add performance benchmarking
- [ ] Support for binary file input
- [ ] Multi-threaded CPU version
- [ ] Compression ratio analysis
- [ ] Visualization of Huffman tree
- [ ] Comparison with other compression algorithms
- [ ] Batch processing multiple files
- [ ] Memory usage optimization

## 🐛 Troubleshooting

### Common Issues

1. **CuPy Installation Error**:
   ```bash
   # If GPU not available, use CPU version
   # Replace cupy with numpy in the code
   ```

2. **CUDA Not Found**:
   - Install CUDA toolkit from NVIDIA
   - Or use CPU-only version with NumPy

3. **Import Error**:
   ```bash
   pip install cupy numpy
   ```

## 📝 Notes

- **GPU Requirement**: CuPy requires NVIDIA GPU with CUDA support
- **CPU Fallback**: Can be modified to use NumPy for CPU-only systems
- **Compression Ratio**: Huffman encoding provides optimal prefix codes
- **Lossless**: Original data can be perfectly reconstructed

## 👤 Author

FluffyCrunch

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This is a mini-project for learning HPC concepts. For production use, consider using established compression libraries like zlib or gzip.

