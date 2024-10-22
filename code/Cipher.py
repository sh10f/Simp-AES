import time

import numpy as np

from utils import split, swap, merge, decToBin, binToDec, strToBytes, bytesToStr


class SBox:
    def __init__(self, forward_order=None, reversed_order=None):
        if forward_order is None:
            self.forward_order = np.array([[9, 4, 10, 11],
                                           [13, 1, 8, 5],
                                           [6, 2, 0, 3],
                                           [12, 14, 15, 7]], dtype=np.uint8)
        else:
            self.forward_order = forward_order

        if reversed_order is None:
            self.reverse_order = np.array([[10, 5, 9, 11],
                                           [1, 7, 8, 15],
                                           [6, 0, 2, 3],
                                           [12, 4, 13, 14]], dtype=np.uint8)
        else:
            self.reverse_order = reversed_order

    def forward(self, input):  # input 是 4 bit 的 array
        row_index = binToDec(input[:2])
        col_index = binToDec(input[2:])

        result = self.forward_order[row_index][col_index]
        result = decToBin(result)
        return result[-4:]

    def reverse(self, input):
        row_index = binToDec(input[:2])
        col_index = binToDec(input[2:])

        result = self.reverse_order[row_index][col_index]
        result = decToBin(result)
        return result[-4:]


# class S_AED:
#     def __init__(self):

def addRoundKey(input, key):
    # input 为 8 * 2 矩阵， key为 8 * 2
    input = input.T.reshape(1, -1)
    key = key.T.reshape(1, -1)
    result = input ^ key
    result = result.reshape(2, -1).T

    return result


def subByte(input, isForward=True):
    # input 8 * 2
    sbox = SBox()
    t_input = input.T.reshape(-1, 4)

    result = np.array([], dtype=np.uint8)

    for i in t_input:
        if isForward:
            t = sbox.forward(i)
        else:
            t = sbox.reverse(i)
        result = np.append(result, t)

    result = result.reshape(-1, 8).T
    return result


def shiftRow(input):
    # input 为 8 * 2
    input = input.T

    for i in range(len(input[0]) // 2, len(input[0])):
        t = input[0][i]
        input[0][i] = input[1][i]
        input[1][i] = t

    result = input.T
    return result


def mixColumn(input, isForward=True):
    # input 8 * 2
    if isForward:
        M = np.array([[0, 0, 0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 1]], dtype=np.uint8).T
    else:
        M = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 1, 0, 0, 1]], dtype=np.uint8).T

    input = input.T
    M = M.T

    t_input = np.array([], dtype=np.uint8)
    t_M = np.array([], dtype=np.uint8)
    for i in range(0, len(input[0]), 4):
        for j in [0, 1]:
            t_input = np.append(t_input, binToDec(input[j, i:i + 4]))
            t_M = np.append(t_M, binToDec(M[j, i:i + 4]))

    t_input = t_input.reshape(2, -1)
    t_M = t_M.reshape(2, -1)
    result = []
    result.append(decToBin(GF_ADD(GF_MUL(t_M[0][0], t_input[0][0]),
                                  GF_MUL(t_M[0][1], t_input[1][0])
                                  ))[-4:]
                  )

    result.append(decToBin(GF_ADD(GF_MUL(t_M[1][0], t_input[0][0]),
                                  GF_MUL(t_M[1][1], t_input[1][0])
                                  ))[-4:]
                  )

    result.append(decToBin(GF_ADD(GF_MUL(t_M[0][0], t_input[0][1]),
                                  GF_MUL(t_M[0][1], t_input[1][1])
                                  ))[-4:]
                  )

    result.append(decToBin(GF_ADD(GF_MUL(t_M[1][0], t_input[0][1]),
                                  GF_MUL(t_M[1][1], t_input[1][1])
                                  ))[-4:]
                  )

    result = np.array(result).reshape(2, -1).T
    return result


def GF_ADD(row, column):
    addTable = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                         [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14],
                         [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13],
                         [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
                         [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
                         [5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10],
                         [6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9],
                         [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
                         [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
                         [9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6],
                         [10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5],
                         [11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4],
                         [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3],
                         [13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2],
                         [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1],
                         [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]], dtype=np.uint8)
    result = addTable[row][column]
    return result


def GF_MUL(row, column):
    mulTable = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                         [0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13],
                         [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 14, 7, 4, 1, 2, 13],
                         [0, 4, 8, 12, 3, 7, 11, 15, 6, 2, 14, 10, 5, 1, 13, 9],
                         [0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1, 9, 12, 3, 6],
                         [0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4],
                         [0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11],
                         [0, 8, 3, 11, 6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1],
                         [0, 9, 1, 8, 2, 11, 3, 10, 4, 13, 5, 12, 6, 15, 7, 14],
                         [0, 10, 7, 13, 14, 4, 9, 3, 15, 5, 8, 2, 1, 11, 6, 12],
                         [0, 11, 5, 14, 10, 1, 15, 4, 7, 12, 2, 9, 13, 6, 8, 3],
                         [0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1, 13, 15, 3, 4, 8],
                         [0, 13, 9, 4, 1, 12, 8, 5, 11, 6, 3, 14, 15, 7, 2, 10],
                         [0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 4, 8, 10, 11, 5],
                         [0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10]], dtype=np.uint8)

    result = mulTable[row][column]

    return result


class keyGenerator:
    def __init__(self, key, num_keys):
        # key 8 * 2
        self.key = key
        self.num_keys = num_keys
        self.RCon = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.uint8).T

    def expandKey(self, isForward=True):
        child_keys = np.array([], dtype=np.uint8)
        w0 = self.key[:, 0]
        w1 = self.key[:, 1]
        for i in range(self.num_keys - 1):
            t_w1 = np.concatenate((w1[len(w1) // 2:], w1[:len(w1) // 2]))
            t = subByte(t_w1).reshape(8, )
            w0 = w0 ^ self.RCon[:, i % self.RCon.shape[1]] ^ t
            w1 = w0 ^ w1
            child_keys = np.append(child_keys, w0)
            child_keys = np.append(child_keys, w1)
        child_keys = child_keys.reshape(-1, 8).T
        if isForward:
            return self.key, child_keys[:, 0:2], child_keys[:, 2:4]
        else:
            return child_keys[:, 2:4], child_keys[:, 0:2], self.key


class S_AES:
    def __init__(self):
        self.expandLen = None
        pass

    def control(self, input, key, mode="en", isForward=True, keyOrder=None, strategy=None):
        input = self.dataTrans(input)
        key = self.dataTrans(key)
        result = None
        if mode == "en":
            result = self.encrypt(input, key)
        elif mode == "de":
            result = self.decrypt(input, key)
        elif mode == "multi":
            result = self.multi(input, key, keyOrder, strategy, isForward)

        result = self.dataOut(result)
        return result

    def encrypt(self, input, key):
        keyG = keyGenerator(key, num_keys=3)
        k1, k2, k3 = keyG.expandKey(isForward=True)

        x = addRoundKey(input, k1)

        x = subByte(x, isForward=True)
        x = shiftRow(x)
        x = mixColumn(x, isForward=True)
        x = addRoundKey(x, k2)

        x = subByte(x, isForward=True)
        x = shiftRow(x)
        x = addRoundKey(x, k3)

        return x

    def decrypt(self, input, key):
        keyG = keyGenerator(key, num_keys=3)
        k3, k2, k1 = keyG.expandKey(isForward=False)

        x = addRoundKey(input, k3)

        x = shiftRow(x)
        x = subByte(x, isForward=False)
        x = addRoundKey(x, k2)
        x = mixColumn(x, isForward=False)

        x = shiftRow(x)
        x = subByte(x, isForward=False)
        x = addRoundKey(x, k1)

        return x

    def multi(self, input, key, keyOrder, strategy, isForward=True):
        x = input
        # key 8 * 4
        for i in range(len(strategy)):
            if isForward:
                k = key[:, keyOrder[i] * 2:keyOrder[i] * 2 + 2]
            else:
                k = key[:, keyOrder[i] * 2:keyOrder[i] * 2 + 2]
            if strategy[i] == 0:
                x = self.encrypt(x, k)
            elif strategy[i] == 1:
                x = self.decrypt(x, k)

        return x


    def dataTrans(self, input):
        # input (x,)
        self.expandLen = 0
        while input.shape[0] % 16 != 0:
            input = np.append(input, 0)
            self.expandLen += 1

        # print("Trans: ",input.shape)

        result = input.reshape(-1, 8).T
        return result

    def dataOut(self, input):
        result = input.T.reshape(-1,)
        return result

    def mimAttach(self, plain, cipher):
        # 只针对双重加密
        # strategies= [[0, 1]]  # attack strategy
        strategies= [[0, 1], [1, 0], [1, 1], [0 ,0] ]  # attack strategy

        keys = {}
        for strategy in strategies:
            for i in range(2 ** 16):
                k = decToBin(i, isInt8=False)   # (16, )
                if strategy[0] == 0:
                    mid_left = self.control(plain, k, "en")
                else:
                    mid_left = self.control(plain, k, "de")

                for j in range(2 ** 16):
                    k = decToBin(j, isInt8=False)
                    if strategy[1] == 0:
                        mid_right = self.control(cipher, k, "en")
                    else:
                        mid_right = self.control(cipher, k, "de")

                    mid_left = np.array(mid_left, dtype=np.uint8)
                    mid_right = np.array(mid_right, dtype=np.uint8)
                    if np.sum(mid_left == mid_right) == len(mid_right):
                        keys[binToDec(strategy)] = [i, j]
                        return keys[binToDec(strategy)]




    def cbc(self, inputs, IV, key, isForward=True):
        inputs = inputs.reshape(-1, 16)
        if not isForward:
            inputs = inputs[list(range(inputs.shape[0]))[::-1], :]

        result = np.array([], dtype=np.uint8)
        for i in range(inputs.shape[0]):
            input = inputs[i]
            if isForward:
                x = IV ^ input
                x = self.control(x, key, "en")
                IV = x
                result = np.append(result, x)
            else:
                x = self.control(input, key, "de")
                if i + 1 < inputs.shape[0]:
                    x = inputs[i+1] ^ x
                else:
                    x = IV ^ x
                result = np.append(result, x)

        if isForward:
            return result
        else:
            result = result.reshape(-1, 16)
            result = result[list(range(result.shape[0]))[::-1], :]
            return result.reshape(-1,)









if __name__ == '__main__':
    # # input = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.uint8)
    # # input = np.array([[1, 0, 1, 0, 0, 1, 1, 1],
    # #                   [0, 1, 0, 0, 1, 0, 0, 1]], dtype=np.uint8).T
    #
    # a = "ab"
    # input = strToBytes(a, isBinary=False)
    # print("input: ", input)
    # key = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 0, 0, 0, 0, 1, 1, 1]
    #                 ], dtype=np.uint8).T
    # key = key.T.reshape(-1,)
    # print("key:, ", key, key.shape)
    #
    # keyOrder = np.array([0, 1], dtype=np.uint8)
    # strategy = np.array([0, 0], dtype=np.uint8)
    #
    # # input = np.array([1,0,0,1,1,1,1,0], dtype=np.uint8).T
    # # input = np.array([[0,1,1,0,],
    # #                   [0,0,0,1,1,1,0,0]]).T
    # # key = np.array([[0,0,1,0,1,1,0,1],
    # #                   [0,1,0,1,0,1,0,1]], dtype=np.uint8).T
    #
    # sAES = S_AES()
    # # a = sAES.multi(input, key, keyOrder, strategy)
    # a = sAES.control(input, key, "multi", True, keyOrder, strategy)
    # print(a.shape)
    # keyOrder_de = keyOrder[::-1]
    # strategy_de = (strategy ^ np.ones_like(strategy, dtype=np.uint8))[::-1]
    # # b = sAES.multi(a, key, keyOrder_de, strategy_de, isForward=False)
    # b = sAES.control(a, key, "multi", False, keyOrder_de, strategy_de)
    # print("cipher: ", b, b.shape)
    #
    # keys = sAES.mimAttach(input, a)
    # print("keys: ",keys)



    # input = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.uint8)
    # input = np.array([[1, 0, 1, 0, 0, 1, 1, 1],
    #                   [0, 1, 0, 0, 1, 0, 0, 1]], dtype=np.uint8).T

    a = "abcdef"
    input = strToBytes(a, isBinary=False)
    print("input: \n", input)
    key = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                    ], dtype=np.uint8).T
    key = key.T.reshape(-1,)
    print("key:, ", key, key.shape)

    IV = 500
    IV = decToBin(IV, isInt8=False)
    print("IV: ", IV)



    sAES = S_AES()
    # a = sAES.multi(input, key, keyOrder, strategy)
    a = sAES.cbc(input, IV, key, isForward=True)
    # print("a: \n", a.reshape(-1, 8))

    b = sAES.cbc(a, IV, key, isForward=False)
    print("cipher: \n", b, b.shape)

    print("true? \n", b == input)


