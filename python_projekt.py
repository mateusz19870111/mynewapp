"""
http://home.agh.edu.pl/~chwiej/mn/generatory_2015.pdf
http://www.if.pw.edu.pl/~agatka/numeryczne/wyklad_12.pdf

1. Wygenerowaæ liczby za pomoc¹ "Generatora liczb z rozk³adu normalnego" dwoma metodami
- Metod¹ Boxa-Mullera (funkcja)
- Metod¹ Centralnego twierdzenia granicznego(funkcja)

2. Przetestowaæ czy generetory, generuj¹ w taki sam sposób jak wbudowany w Pythona generator liczb z rozk³adu normalnego

3. Wykonaæ testy statystyczne:
- Test Shapiro-Wilka
- Test Kolmogorov–Smirnov

4. Check the output with Monte Carlo theory
"""

import logging
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from numpy import random, sqrt, log, sin, cos, pi

logging.basicConfig()


# Part 1

def box_muller_transform(inp_u1, inp_u2):
    """
    Box Muller transformation function
    :param inp_u1: numpy.ndarray  or numeric
    :param inp_u2: numpy.ndarray or numeric
    :return: transformed numpy.ndarray or numeric
    """

    try:
        out_z1 = sqrt(-2 * log(inp_u1)) * cos(2 * pi * inp_u2)
        out_z2 = sqrt(-2 * log(inp_u1)) * sin(2 * pi * inp_u2)
    except (TypeError, AttributeError) as msg:
        logging.warning(msg='Error {0}, \nexiting program'.format(msg))
        exit(0)

    return out_z1, out_z2


def clt_transform(x_size, out_count):
    """
    CLT transformation function
    :param x_size: int (N)
    :param out_count: int (count of generated output numbers)
    :return: transformed list of values
    """

    try:
        output_clt = []
        for k in range(out_count):
            trans_x = np.random.uniform(-1, 1, size=x_size)
            trans_value = np.mean(trans_x - np.mean(trans_x)) / (np.std(trans_x) / np.sqrt(x_size))
            output_clt.append(trans_value)
    except (TypeError, AttributeError, ZeroDivisionError) as msg:
        logging.warning(msg='Error {0}, \nexiting program'.format(msg))
        exit(0)

    return output_clt


# Generate numbers with Box Muller transformation
# uniformly distributed values between 0 and 1
u_1 = random.rand(4000)
u_2 = random.rand(4000)
bm_1, bm_2 = box_muller_transform(u_1, u_2)

# Generate numbers with Central Limit Theorem transformation
clt_1 = clt_transform(100, 5000)

# Part 2
def rand_norm_generator(mu, sigma, cnt_nb):
    """
    Random Normal generator
    :param mu: int, float
    :param sigma: int, float
    :param cnt_nb: int (count numb)
    :return: list, rand norm
    """

    try:
        output_rand_norm = random.normal(mu, sigma, cnt_nb)
    except (TypeError, AttributeError) as msg:
        logging.warning(msg='Error {0}, \nexiting program'.format(msg))
        exit(0)

    return output_rand_norm


g_1 = rand_norm_generator(0, 0.1, 4000)
plt.hist(g_1, 100, facecolor='green', alpha=0.75)
plt.title('Random Normal generator')
plt.grid(linestyle='--', linewidth=0.5)
plt.show()

plt.hist(bm_1, 100, facecolor='blue', alpha=0.75)
plt.hist(bm_2, 100, facecolor='yellow', alpha=0.70)
plt.title('Box - Muller')
plt.grid(linestyle='--', linewidth=0.5)
plt.show()

plt.hist(clt_1, 100, facecolor='red', alpha=0.75)
plt.title('Central Limit Theorem')
plt.grid(linestyle='--', linewidth=0.5)
plt.show()

# Part 3
# 3.1 Tests for Box - Muller
print('========  Tests for Box - Muller  =========')
# Shapiro - Wilk test
shap_w_bm_1 = scipy.stats.shapiro(bm_1)
print("\nShapiro-Wilk Test\n", shap_w_bm_1)

# Kołmogorov - Smirnov test
kol_smir_bm_1 = scipy.stats.kstest(bm_1, 'norm')
print("\nKołmogorov - Smirnov Test\n", kol_smir_bm_1)

# 3.2 Tests for Central Limit Theorem
print('\n========  Tests for Central Limit Theorem  =========')
# Shapiro - Wilk test
shap_w_clt_1 = scipy.stats.shapiro(clt_1)
print("\nShapiro-Wilk Test\n", shap_w_clt_1)

# Kołmogorov - Smirnov test
kol_smir_clt_1 = scipy.stats.kstest(clt_1, 'norm')
print("\nKołmogorov - Smirnov Test\n", kol_smir_clt_1)


# Part 4 Test 100 runs based on Monte Carlo assumption
print('\nTest 100 runs based on Monte Carlo assumption\n')
pv_counter_box_miller_shapiro = []
pv_counter_box_miller_kolmogorow = []
pv_counter_clt_shapiro = []
pv_counter_clt_kolmogorow = []

long_i = 100.0

for i in range(int(long_i)):
    u_1 = random.rand(4000)
    u_2 = random.rand(4000)
    bm_1, bm_2 = box_muller_transform(u_1, u_2)
    clt_1 = clt_transform(30, 1000)

    shap_w_bm_1 = scipy.stats.shapiro(bm_1)
    pv_counter_box_miller_shapiro.append(shap_w_bm_1[1])
    kol_smir_bm_1 = scipy.stats.kstest(bm_1, 'norm')
    pv_counter_box_miller_kolmogorow.append(kol_smir_bm_1.pvalue)

    shap_w_clt_1 = scipy.stats.shapiro(clt_1)
    pv_counter_clt_shapiro.append(shap_w_clt_1[1])
    kol_smir_clt_1 = scipy.stats.kstest(clt_1, 'norm')
    pv_counter_clt_kolmogorow.append(kol_smir_clt_1.pvalue)

# count and results
if len([1 for i in pv_counter_box_miller_shapiro if i < 0.05]) / long_i > 0.5:
    print('Box Miller by Shapiro test reject normality')
else:
    print('Box Miller by Shapiro test does not reject normality')

if len([1 for i in pv_counter_box_miller_kolmogorow if i < 0.05]) / long_i > 0.5:
    print('Box Miller by Kolmogorow test reject normality')
else:
    print('Box Miller by Kolmogorow test does not reject normality')

if len([1 for i in pv_counter_clt_shapiro if i < 0.05]) / long_i > 0.5:
    print('CLT by Shapiro test reject normality')
else:
    print('CLT by Shapiro test does not reject normality')

if len([1 for i in pv_counter_clt_kolmogorow if i < 0.05]) / long_i > 0.5:
    print('CLT by Kolmogorow test reject normality')
else:
    print('CLT by Kolmogorow test does not reject normality')
