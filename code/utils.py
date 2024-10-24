import numpy as np


def binToDec(input):
    binary_string = ''.join(np.array(input).astype(str))

    # 使用 int 函数将二进制字符串转换为十进制
    decimal_value = int(binary_string, 2)
    return decimal_value


def decToBin(input, isInt8=True):  # 返回一个字节 --- 8bit
    if isInt8:
        t = np.array(input, dtype=np.uint8)
        binary_array = np.unpackbits(t)
        return binary_array
    else:
        t = np.array(input, dtype=np.uint16)
        # 将 np.uint16 类型的数据拆分为两个 np.uint8 类型的数据
        low_byte = np.array(t & 0xFF, dtype=np.uint8)
        high_byte = np.array((t >> 8) & 0xFF, dtype=np.uint8)

        # 使用 np.unpackbits 分别处理低字节和高字节，得到二进制数组
        low_bits = np.unpackbits(low_byte)
        high_bits = np.unpackbits(high_byte)

        # 将低字节和高字节的二进制数组合并为一个数组
        binary_array = np.concatenate((high_bits, low_bits))
        return binary_array


def strToBytes(strings, isBinary=True):
    if isBinary:
        binary_array = np.array([ord(char) - 48 for char in strings], dtype=np.uint8)
    else:
        t = strings.encode('utf-8')
        t = np.array([i for i in t], dtype=np.uint8)
        binary_array = np.unpackbits(t)
        print(binary_array.shape)

    return binary_array


def bytesToStr(binary_array, isBinary=True):
    if isBinary:
        string_result = str(binary_array).strip("[]").replace(" ", "")

    else:
        # 将二进制数组重塑为字节数组
        byte_array = np.packbits(binary_array)

        # 将字节数组转换为字符串
        string_result = byte_array.tobytes().decode('utf-8', errors='ignore')
    return string_result


def hexToBytes(hex_string):
    result = np.array([], dtype=np.uint8)
    for x in hex_string:
        if ord(x) >= 65:
            result = np.append(result, decToBin(ord(x) - 55)[-4:])
        else:
            result = np.append(result, decToBin(ord(x) - 48)[-4:])
    return result


def bytesToHex(byte_array):
    result = ''
    for i in range(0, len(byte_array), 4):
        t = binToDec(byte_array[i:i + 4])
        if t < 10:
            result += chr(t + 48)
        else:
            result += chr(t + 55)

    return result


if __name__ == '__main__':
    # s = "aa"
    # a = strToBytes(s, isBinary=False)
    # a = a.reshape(16,1)
    # print(a.shape)
    # print(bytesToStr(a, isBinary=False))
    #
    # a = [1,1]
    # print(binToDec(a))

    t = 0
    print(decToBin(t, False))

    # v = bytesToHex(a)
    # print(v)
    # print(binToDec([1, 0, 1, 0]))
