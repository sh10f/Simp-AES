 [中文文档](./中文文档.md)/[English Document](./README.md)

 [User Guide](./Report/用户指南.md)/[Test Results](./Report/测试结果.md)/[API Documentation](./Report/接口文档.md)

---

# S-AES Encryption and Decryption

This project implements a simplified version of the Advanced Encryption Standard (S-AES), which performs encryption and decryption on 16-bit data blocks using 16-bit keys. It extends AES concepts to provide multi-layer encryption, decryption, and various cryptographic modes, including CBC (Cipher Block Chaining). The project is designed with modularity and flexibility in mind, allowing for custom multi-layer encryption without layer limits, making it a useful tool for both educational and research purposes.

## Features

- **Basic S-AES Encryption and Decryption**: Implements standard S-AES encryption and decryption for 16-bit data and 16-bit keys.
- **Multi-layer Encryption Support**: Supports custom multi-layer encryption and decryption, where the number of layers is unlimited. Users can define encryption and decryption strategies using different keys and key orders.
- **CBC Mode**: Supports Cipher Block Chaining (CBC) mode for encrypting longer messages, with user-defined initialization vectors (IVs).
- **Meet-in-the-Middle Attack**: Implements a meet-in-the-middle attack method for breaking double encryption by comparing intermediate states.
- **Customizable and Modular**: The code is modular, with clearly defined functions for key expansion, S-box substitution, and round transformations, making it easy to extend or modify the code for various cryptographic tasks.

## Code Structure

- **`Cipher.py`**: Contains the main implementation of the S-AES algorithm, supporting encryption, decryption, multi-layer encryption, and CBC mode operations.
- **`utils.py`**: Contains utility functions for converting between binary and decimal, handling string-to-byte conversions, and other data transformations necessary for cryptographic operations.

### Main Components

1. **S-AES Class**: 
    - Manages the encryption, decryption, and multi-layer encryption workflows.
    - Provides CBC mode support, making the encryption of longer messages possible.
    - Flexible control for custom encryption and decryption strategies, allowing users to define how many layers they want to apply and in what sequence.
  
2. **SBox Class**: 
    - Handles the substitution step in the encryption process using predefined S-boxes for both forward and reverse transformations.
    - Allows easy replacement or modification of substitution logic by updating the S-boxes, providing flexibility for research or experimentation.

3. **Key Generation**:
    - A flexible key expansion mechanism, which generates sub-keys for multiple encryption rounds.
    - Keys are expanded using the provided 16-bit input key and used across multiple rounds.

4. **Meet-in-the-Middle Attack**:
    - Implements an attack strategy to break double encryption by comparing intermediate encryption states from both sides of the encryption process.

## Installation

This project requires Python 3.6 or higher. The following dependencies must be installed:

```bash
pip install numpy
```

After installing the dependencies, you can run the program directly using Python.

## Usage

### Encryption/Decryption

You can perform encryption and decryption operations by providing a plaintext or ciphertext and a key.

Example usage in `Cipher.py`:

```python
from Cipher import S_AES
from utils import strToBytes, bytesToStr

# Initialize S-AES instance
sAES = S_AES()

# Example plaintext and key
plaintext = strToBytes("efbt", isBinary=False)  # Converts the string to binary format
key = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8).T
key = key.T.reshape(-1,)

# Perform encryption
ciphertext = sAES.control(plaintext, key, mode="en")
print("Ciphertext:", bytesToStr(ciphertext, isBinary=False))

# Perform decryption
decrypted = sAES.control(ciphertext, key, mode="de")
print("Decrypted Text:", bytesToStr(decrypted, isBinary=False))
```

### Multi-layer Encryption

The `multi()` function in the `S_AES` class allows for an arbitrary number of encryption layers. Users can define the encryption order and strategy (encryption or decryption) for each layer.

```python
keyOrder = np.array([0, 1, 2], dtype=np.uint8)
strategy = np.array([0, 1, 0], dtype=np.uint8)  # Define encryption (0) and decryption (1) order

# Perform multi-layer encryption
ciphertext = sAES.multi(plaintext, key, keyOrder, strategy, isForward=True)
```

### CBC Mode

For encrypting longer messages using CBC mode, define an initialization vector (IV) and input plaintext blocks.

```python
IV = decToBin(52, isInt8=False)  # Initialization vector
ciphertext = sAES.cbc(plaintext, IV, key, isForward=True)
```

### Meet-in-the-Middle Attack

The program also provides a method for performing a meet-in-the-middle attack on double encryption:

```python
keys = sAES.mimAttach(plain, cipher)
print("Discovered keys:", keys)
```

## Project Characteristics

### Code Generality

The program is designed to be flexible and easily modifiable:
- **Modular Design**: The code is split into self-contained functions and classes, which makes it easy to adapt and extend. For instance, users can modify the S-box substitution logic by providing different S-boxes or change the key expansion algorithm.
- **Multi-layer Encryption Support**: Unlike traditional fixed-round encryption schemes, this implementation allows users to define an arbitrary number of encryption layers, making the code highly customizable for different cryptographic experiments.

### Extensibility

The design of the code ensures it is adaptable to future requirements:
- **Customizable Layers**: Users can define not only the number of encryption layers but also the encryption and decryption order, giving more control over the process.
- **Support for Multiple Modes**: While the CBC mode is currently implemented, the modular design allows for easy extension to other cryptographic modes, such as CFB, OFB, or CTR.
- **Meet-in-the-Middle Attack**: The code includes advanced cryptanalysis features, like meet-in-the-middle attacks, that can be extended for further study in cryptographic security.

## Testing

To test the functionality, the project provides pre-built test cases in the `__main__` block of `Cipher.py`, where different encryption, decryption, and attack scenarios are demonstrated.

### Running Tests

Run the following command to execute the provided tests:

```bash
python Cipher.py
```

This will execute a series of encryption, decryption, and CBC mode operations, along with testing the meet-in-the-middle attack function.

## Conclusion

This S-AES project is a versatile and extendable implementation of simplified AES encryption, suitable for educational purposes and research. Its modular design, support for custom multi-layer encryption, and flexibility in cryptographic mode extension make it a powerful tool for studying encryption algorithms.

