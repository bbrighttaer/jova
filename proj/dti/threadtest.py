# Author: bbrighttaer
# Project: jova
# Date: 12/4/19
# Time: 5:16 PM
# File: threadtest.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import ceil
from threading import Thread

import numpy as np


def foo(i, x, q):
    q.append((i, x))


a = np.random.randn(5000, 10)
bsize = 100
num = ceil(len(a) / bsize)
q = []
procs = []
for i in range(num):
    thread = Thread(target=foo, args=(i, a[i * bsize:i * bsize + bsize], q))
    procs.append(thread)
    thread.start()
for t in procs:
    t.join()
q.sort(key=lambda x: x[0])
print('done!', num, np.array(q)[:5], a[:5])
