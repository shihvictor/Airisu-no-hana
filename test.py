# Check the versions of libraries

# # Python version
# import sys
# print('Python: {}'.format(sys.version))
# # scipy
# import scipy
# print('scipy: {}'.format(scipy.__version__))
# # numpy
# import numpy
# print('numpy: {}'.format(numpy.__version__))
# # matplotlib
# import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# # pandas
# import pandas
# print('pandas: {}'.format(pandas.__version__))
# # scikit-learn
# import sklearn
# print('sklearn: {}'.format(sklearn.__version__))
import numpy as np

a = np.zeros((4, 1), dtype=int)
for i in range(0, 4):
    a[i] = 1

aa = 2

bb = 3.1

cc = aa*bb
print(type(aa))
print(type(bb))
print(type(cc))