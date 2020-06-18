import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
with open('curveball.pkl', 'rb') as f1:
    a = pkl.load(f1)
with open('multi_curveball.pkl', 'rb') as f2:
    b = pkl.load(f2)
plt.figure()
plt.xlabel('epochs')
plt.ylabel('test acc')
plt.plot(a['test_acc'], label='cuveball')
plt.plot(b['test_acc'], label='multi-step 3 order')
plt.legend()
plt.savefig('test_acc.png', dpi=200)
