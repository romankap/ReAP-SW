__author__ = 'Roman'

import os,sys

class FixedPointFormat:
    def __init__(self, integer_bits, fraction_bits):
        self.integer_bits = integer_bits
        self.fraction_bits = fraction_bits
        self.total_bits = integer_bits + fraction_bits

        self.rounding = 1 << fraction_bits
        self.max = (1 << (integer_bits - 1)) - 1/self.rounding  # 1 bit saved for sign

    def cycles_per_ADD(self):
        return (self.integer_bits + self.fraction_bits) * 4

    def cycles_per_MUL(self):
        return (self.integer_bits + self.fraction_bits) ** 2

    def convert(self, value):
        res = int(value * self.rounding) / self.rounding
        if res > self.max:
            res = self.max
        elif res < -self.max:
            res = -self.max
        return res


class FixedPointNumber:
    def __init__(self, value, rep_format):
        self.rep_format = rep_format
        self.val = self.convert(value)


    def convert(self, value):
        res = int(value * self.rep_format.rounding) / self.rep_format.rounding
        if res > self.rep_format.max:
            res = self.rep_format.max
        elif res < -self.rep_format.max:
            res = -self.rep_format.max

        return res


    def __add__(self, other):
        return FixedPointNumber(self.val + other.val, self.rep_format)

    def __radd__(self, other):
        return other + self


    def __sub__(self, other):
        return FixedPointNumber(self.val - other.val, self.rep_format)

    def __radd__(self, other):
        return other - self


    def __mul__(self, other):
        tmp_res = self.convert(self.val * other.val)
        return FixedPointNumber(tmp_res, self.rep_format)

    def __rmul__(self, other):
        return other * self


def test():
    num_format = FixedPointFormat(8,4)
    test_num_a = FixedPointNumber(3.12, num_format)
    test_num_b = FixedPointNumber(4.5, num_format)

    print("a.val =", test_num_a.val)
    print("b.val =", test_num_b.val)

    test_num_c = test_num_a - test_num_b;
    print("a-b =", test_num_c.val)
    test_num_c = test_num_b - test_num_a;
    print("b-a =", test_num_c.val)

    test_num_c = test_num_a + test_num_b;
    test_num_c = test_num_c - test_num_a;
    print("a+b-a =", test_num_c.val)

    test_num_c = test_num_a * test_num_b;
    print("a*b =", test_num_c.val)

    print()
    print("overflow test")
    overflow_test_num = FixedPointNumber(999999, num_format)
    print("overflow_test_num.val =", overflow_test_num.val)

    print("negative overflow test")
    overflow_test_num = FixedPointNumber(-9999999, num_format)
    print("overflow_test_num.val =", overflow_test_num.val)


#test()
