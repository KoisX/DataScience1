import numpy as np
from numpy.linalg import LinAlgError

A = np.array([[16, 25, 17, 25], [17, 16, 25, 16], [17, 25, 16, 17]])
B = np.array([[25, 16], [17, 25], [25, 16], [17, 25]])
C = np.array([[17, 16, 17, 25]])
D = np.array([[16, 25, 17]])
E = 16
S = np.array([[25, 16], [68, 25]])


def main():
    run_numpy_functions()
    run_operations()
    matrix_file_upload()


#  first part of the assignment
def run_numpy_functions():
    print('In run_numpy_functions function')

    print('A: ', A)
    print('B: ', B)
    print('C: ', C)

    print('Ravel:')  # matrix in one line
    print('A: ', A.ravel())
    print('B: ', B.ravel())
    print('C: ', C.ravel())

    print('\nShape')  # how many rows and columns
    print('A: ', A.shape)
    print('B: ', B.shape)
    print('C: ', C.shape)

    print('\nVStack')  # Stack arrays in sequence vertically (row wise).
    print('A: ', np.vstack(A))
    print('B: ', np.vstack(B))
    print('C: ', np.vstack(C))

    print('\nHStack')  # Stack arrays in sequence horizontally (column wise).
    print('A: ', np.hstack(A))
    print('B: ', np.hstack(B))
    print('C: ', np.hstack(C))

    print('\nReshape')  # Change dimensions of matrix
    print('A: ', np.reshape(A, 12))
    print('B: ', np.reshape(B, (2, 4)))
    print('C: ', np.reshape(C, (2, 2)))

    print('\nNDim')  # number of dimensions of the matrix
    print('A: ', A.ndim)
    print('B: ', B.ndim)
    print('C: ', C.ndim)

    print('\nShape')  # Number of rows and columns
    print('A: ', A.shape)
    print('B: ', B.shape)
    print('C: ', C.shape)

    print('\nSize')  # Number of elements
    print('A: ', A.size)
    print('B: ', B.size)
    print('C: ', C.size)

    print('\nDType')  # data type of matrix elements
    print('A: ', A.dtype)
    print('B: ', B.dtype)
    print('C: ', C.dtype)

    print('\nSqrt')  # sqrt of each element
    a = np.sqrt(A)
    b_sqrt = np.sqrt(B)
    c_sqrt = np.sqrt(C)
    print('A: ', a)
    print('B: ', b_sqrt)
    print('C: ', c_sqrt)

    print('\nFloor')  # floor of matrix elementwise
    print('A: ', np.floor(a))
    print('B: ', np.floor(b_sqrt))
    print('C: ', np.floor(c_sqrt))

    print('\nMedian')  # median of matrix elements
    print('A: ', np.median(A))
    print('B: ', np.median(B))
    print('C: ', np.median(C))

    print('\nMean')  # mean of matrix elements
    print('A: ', np.mean(A))
    print('B: ', np.mean(B))
    print('C: ', np.mean(C))

    print('\nVar')  # variance of matrix elements
    print('A: ', np.var(A))
    print('B: ', np.var(B))
    print('C: ', np.var(C))

    print('\nStd')  # standard deviation of matrix elements
    print('A: ', np.std(A))
    print('B: ', np.std(B))
    print('C: ', np.std(C))

    print('\nSum')  # sum of elements by each axis
    print('A: ', A.sum(axis=1))
    print('B: ', B.sum(axis=1))
    print('C: ', C.sum(axis=1))

    print('\nMin')  # min in matrix
    print('A: ', A.min(initial=100))
    print('B: ', B.min(initial=100))
    print('C: ', C.min(initial=100))

    print('\nmax')  # max by each axis
    print('A: ', A.max(axis=1, initial=0))
    print('B: ', B.max(axis=1, initial=0))
    print('C: ', C.max(axis=1, initial=0))

    print('\nPercentile 25')  # percentile as in andan
    print('A: ', np.percentile(A, 25))
    print('B: ', np.percentile(B, 25))
    print('C: ', np.percentile(C, 25))

    print('\nPercentile 50')  # percentile as in andan
    print('A: ', np.percentile(A, 50))
    print('B: ', np.percentile(B, 50))
    print('C: ', np.percentile(C, 50))

    print('\nPercentile 75')  # percentile as in andan
    print('A: ', np.percentile(A, 75))
    print('B: ', np.percentile(B, 75))
    print('C: ', np.percentile(C, 75))

    print('\nUnique')  # list of unique elements
    print('A: ', np.unique(A))
    print('B: ', np.unique(B))
    print('C: ', np.unique(C))


#  second part of assignment
def run_operations():
    print('\nIn run_operations function')

    print('D * E')
    print(D * E)

    print('\nE * D')
    print(E * D)

    print('\nD + E')
    print(D + E)

    print('\nA * E')
    print(A * E)

    print('\nE * A')
    print(E * A)

    print('\nA + E')
    print(A + E)

    print('\nB * E')
    print(B * E)

    print('\nE * B')
    print(E * B)

    print('\nB + E')
    print(B + E)

    print('\nC * E')
    print(C * E)

    print('\nE * C')
    print(E * C)

    print('\nC * PI')
    print(C * np.pi)

    print('\nA / E')
    print(A / E)

    print('\nA // E')  # division without remainder
    print(A // E)

    print('\nA % E')
    print(A % E)

    print('\nA ** E')  # A^E
    print(A ** E)

    print('\nE / A')
    print(E / A)

    print('\nE - A')
    print(E - A)

    print('\nnp.log(B)')
    print(np.log(B))

    print('\nC += E')
    print(C + E)

    print('\nC *= E')
    print(C * E)

    print('\nC / E')
    print(C / E)

    print('\nB ** E')
    print(B ** E)

    print('\nA //= E')
    print(A // E)

    print('\nB %= E')
    print(B % E)

    print('\nC > 27')
    print(C > 27)

    print('\nC == 25')
    print(C == 25)

    print('\nA + C')
    print(A + C)

    print('\nA * C')
    print(A * C)

    print('\nB^T + C')
    print(np.transpose(B) + C)

    print('\nnp.dot(A, B)')
    print(np.dot(A, B))

    try:
        print('\nnp.dot(B, C^T)')
        print(np.dot(B, np.transpose(C)))
    except ValueError as e:
        print("Operation unsuccessful: " + str(e))

    print('\ndet(S)')
    print(np.linalg.det(S))

    print('\nround(det(S))')
    print(round(np.linalg.det(S)))

    print('\ninv(S)')
    print(np.linalg.inv(S))

    print('\nsolve(B, C^T)')
    try:
        print(np.linalg.solve(B, np.transpose(C)))
    except LinAlgError as e:
        print('Unsuccessful operation' + str(e))

    print('\nMatrix power (S, 3)')
    print(np.linalg.matrix_power(S, 3))

    print('\nMatrix rank')
    print(np.linalg.matrix_rank(S))


def matrix_file_upload():
    print('\nIn matrix_file_upload function')
    size = 100

    r1 = np.random.randint(1, 100, size=(size, 1))
    r2 = np.random.randint(151, 200, size=(size, 1))

    np.savetxt('r1.txt', r1, newline="\n", fmt="%d")
    np.save('r2.npy', r2)

    r1_read = np.loadtxt('r1.txt', ndmin=2)
    r2_read = np.load('r2.npy')

    print('Matrices are successfully writen to files and loaded from there')


if __name__ == '__main__':
    main()
