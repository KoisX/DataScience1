import numpy as np

A = np.array([[16, 25, 17, 25], [17, 16, 25, 16], [17, 25, 16, 17]])
B = np.array([[25, 16], [17, 25], [25, 16], [17, 25]])
C = np.array([[17, 16, 17, 25]])

print('A: ', A)
print('B: ', B)
print('C: ', C)

print('Ravel:')
print('A: ', A.ravel())
print('B: ', B.ravel())
print('C: ', C.ravel())

print('\nShape')
print('A: ', A.shape)
print('B: ', B.shape)
print('C: ', C.shape)

print('\nVStack')
print('A: ', np.vstack(A))
print('B: ', np.vstack(B))
print('C: ', np.vstack(C))

print('\nHStack')
print('A: ', np.hstack(A))
print('B: ', np.hstack(B))
print('C: ', np.hstack(C))

print('\nReshape')
print('A: ', np.reshape(A, 12))
print('B: ', np.reshape(B, (2, 4)))
print('C: ', np.reshape(C, (2, 2)))

print('\nNDim')  # number of dimensions of the matrix
print('A: ', A.ndim)
print('B: ', B.ndim)
print('C: ', C.ndim)

print('\nShape')
print('A: ', A.shape)
print('B: ', B.shape)
print('C: ', C.shape)

print('\nSize')
print('A: ', A.size)
print('B: ', B.size)
print('C: ', C.size)

print('\nDType')
print('A: ', A.dtype)
print('B: ', B.dtype)
print('C: ', C.dtype)

print('\nSqrt')
A_sqrt = np.sqrt(A)
B_sqrt = np.sqrt(B)
C_sqrt = np.sqrt(C)
print('A: ', A_sqrt)
print('B: ', B_sqrt)
print('C: ', C_sqrt)

print('\nFloor')
print('A: ', np.floor(A_sqrt))
print('B: ', np.floor(B_sqrt))
print('C: ', np.floor(C_sqrt))

print('\nMedian')
print('A: ', np.median(A))
print('B: ', np.median(B))
print('C: ', np.median(C))

print('\nMean')
print('A: ', np.mean(A))
print('B: ', np.mean(B))
print('C: ', np.mean(C))

print('\nVar')  # variance
print('A: ', np.var(A))
print('B: ', np.var(B))
print('C: ', np.var(C))

print('\nStd')  # standard deviation
print('A: ', np.std(A))
print('B: ', np.std(B))
print('C: ', np.std(C))

print('\nSum')  # sum of elements by each axis
print('A: ', A.sum(axis=1))
print('B: ', B.sum(axis=1))
print('C: ', C.sum(axis=1))

print('\nMin')
print('A: ', A.min(initial=100))
print('B: ', B.min(initial=100))
print('C: ', C.min(initial=100))

print('\nmax')  # max by each axis
print('A: ', A.max(axis=1, initial=0))
print('B: ', B.max(axis=1, initial=0))
print('C: ', C.max(axis=1, initial=0))

print('\nPercentile 25')
print('A: ', np.percentile(A, 25))
print('B: ', np.percentile(B, 25))
print('C: ', np.percentile(C, 25))

print('\nPercentile 50')
print('A: ', np.percentile(A, 50))
print('B: ', np.percentile(B, 50))
print('C: ', np.percentile(C, 50))

print('\nPercentile 75')
print('A: ', np.percentile(A, 75))
print('B: ', np.percentile(B, 75))
print('C: ', np.percentile(C, 75))

print('\nUnique')
print('A: ', np.unique(A))
print('B: ', np.unique(B))
print('C: ', np.unique(C))
