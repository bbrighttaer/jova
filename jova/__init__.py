# Author: bbrighttaer
# Project: jova
# Date: 5/23/19
# Time: 10:30 AM
# File: __init__.py.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.cuda import is_available

allow_cuda = False
cuda = is_available() and allow_cuda
