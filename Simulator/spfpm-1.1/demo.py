#!/usr/bin/python3
# Demonstration of Simple Python Fixed-Point Module
# (C)Copyright 2006-2014, RW Penney

import os, sys, time
try:
    lib_path = os.path.abspath('//anaconda3/lib/site-packages')
    sys.path.append(lib_path)
    import matplotlib, numpy
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

lib_path = os.path.abspath(os.path.join('spfpm-1.1'))
sys.path.append(lib_path)
import FixedPoint


def print_FXnum_parameters(fx_num):
    print('resolution: ' + str(fx_num.family.resolution))
    print('scale: ' + str(fx_num.family.scale))
    print('round_up: ' + str(fx_num.family._roundup))
    print('integer bits: ' + str(fx_num.family.integer_bits))
    print('fraction bits: ' + str(fx_num.family.fraction_bits))

def basicDemo():
    """Basic demonstration of roots & exponents at various accuracies"""

    #for resolution in [8, 32, 80, 274]:
    for resolution in [8, 16]:
        print('=== {0} bits === NEGATIVE POWER'.format(resolution))

        for bits in range(1, resolution+3):
            family = FixedPoint.FXfamily(4, resolution-4)
            num = FixedPoint.FXnum(2 ** -bits, family)

            print_FXnum_parameters(num)
            print('2^-{0}= '.format(bits) + str(num))
            print('Scaled 2^-{0}= '.format(bits) + str(num.scaledval))
            print()

        print('=== {0} bits === POSITIVE POWER'.format(resolution))
        for bits in range(1,resolution + 3):
            family = FixedPoint.FXfamily(resolution)
            num = FixedPoint.FXnum(2 ** bits, family)

            print_FXnum_parameters(num)
            print('2^{0} = '.format(bits) + str(num))

            print()

        # print('sqrt(' + str(val) + ')~ ' + str(rt))
        # print('sqrt(' + str(val) + ')^2 ~ ' + str(rt * rt))
        # print('exp(1) ~ ' + str(family.exp1))



def overflowDemo():
    """Illustrate how finite range limits calculation of exponents"""

    res = 20
    print('=== {0}-bit fractional part ==='.format(res))
    for intsize in [4, 8, 16, 32]:
        family = FixedPoint.FXfamily(res, intsize)
        x = FixedPoint.FXnum(0.0, family)
        step = 0.1
        while True:
            try:
                ex = x.exp()
            except FixedPoint.FXoverflowError:
                print('{0:2d}-bit integer part: exp(x) overflows near x={1:.3g}'.format(intsize, float(x)))
                break
            x += step
    print()


def speedDemo():
    """calculate indicative speed of floating-point operations"""

    print('=== speed test ===')
    for res, count in [ (16, 10000), (32, 10000), (64, 10000), (128, 10000), (256, 10000), (512, 10000) ]:
        fam = FixedPoint.FXfamily(res)
        x = FixedPoint.FXnum(0.5, fam)
        lmb = FixedPoint.FXnum(3.6, fam)
        one = FixedPoint.FXnum(1.0, fam)
        t0 = time.clock()
        for i in range(0, count):
            # use logistic-map in chaotic region:
            x = lmb * x * (one - x)
        t1 = time.clock()
        ops = count * 3
        Dt = t1 - t0
        print('{0} {1}-bit operations in {2:.2f}s ~ {3:.2g} FLOPS'.format(ops, res, Dt, (ops / Dt)))


def plotDemo():
    """Plot graph of approximations to Pi"""

    pi_true = FixedPoint.FXfamily(200).pi
    b_min, b_max = 8, 25
    pipoints = []
    for res in range(b_min, b_max+1):
        val = 4 * FixedPoint.FXnum(1, FixedPoint.FXfamily(res)).atan()
        pipoints.append([res, val])
    pipoints = numpy.array(pipoints)
    truepoints = numpy.array([[b_min, pi_true], [b_max, pi_true]])

    plt.xlabel('bits')
    plt.ylabel('$4 tan^{-1}1$')
    plt.xlim([b_min, b_max])
    plt.ylim([3.13, 3.16])
    plt.grid(True)
    for arr in (truepoints, pipoints):
        plt.plot(arr[:,0], arr[:,1])
    plt.show()


if __name__ == "__main__":
    basicDemo()
    # overflowDemo()
    # speedDemo()
    # if HAVE_MATPLOTLIB:
    #     plotDemo()

# vim: set ts=4 sw=4 et:
