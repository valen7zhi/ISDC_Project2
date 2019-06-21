import math
from math import sqrt
import numbers

###


def zeroes(height, width):
    """
    Creates a matrix of zeroes.
    """
    g = [[0.0 for _ in range(width)] for __ in range(height)]
    return Matrix(g)


def identity(n):
    """
    Creates a n x n identity matrix.
    """
    I = zeroes(n, n)
    for i in range(n):
        I.g[i][i] = 1.0
    return I


def dot_product(vectorA, vectorB):
    result = 0
    for i in range(len(vectorA)):
        result += vectorA[i] * vectorB[i]
    return result


class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################

    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError,
                  "Calculating determinant not implemented for matrices largerer than 2x2.")

        # TODO - your code here

        if len(self.g) == 1:
            return self.g
        else:
            return self.g[0][0]*self.g[1][1]-self.g[0][1]*self.g[1][0]

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        # TODO - your code here
        ts = 0
        for i in range(len(self.g)):
            ts += self.g[i][i]

        return(ts)

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError,
                  "inversion not implemented for matrices larger than 2x2.")

        # TODO - your code here

        if len(self.g) == 1:

            inverseMatrix = zeroes(1, 1)
            inverseMatrix[0][0] = 1 / self.g[0][0]
            return inverseMatrix

        if len(self.g) == 2:

            inverseMatrix = zeroes(2, 2)

            inverseMatrix[0][0] = self.g[1][1]
            inverseMatrix[0][1] = -self.g[0][1]
            inverseMatrix[1][0] = -self.g[1][0]
            inverseMatrix[1][1] = self.g[0][0]

            factor = 1 / (self.g[0][0] * self.g[1][1] -
                          self.g[0][1] * self.g[1][0])

            for i in range(len(inverseMatrix.g)):
                for j in range(len(inverseMatrix.g[0])):
                    inverseMatrix[i][j] = factor * inverseMatrix[i][j]

            return inverseMatrix

    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """

        # TODO - your code here
        matrixTran = zeroes(self.w, self.h)

        for j in range(len(self.g[0])):
            for i in range(len(self.g)):
                matrixTran[j][i] = self.g[i][j]

        return matrixTran

    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################

    def __getitem__(self, idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self, other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same")
        #
        # TODO - your code here
        #
        matrixSum = zeroes(self.h, self.w)
        row = []

        for r in range(len(self.g)):
            for c in range(len(self.g[0])):
                matrixSum[r][c] = self.g[r][c] + other.g[r][c]

        return matrixSum

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #
        # TODO - your code here
        #
        row = []
        negmatrix = zeroes(self.h, self.w)
        for i in range(len(self.g)):
            for j in range(len(self.g[0])):
                negmatrix[i][j] = (-1 * self.g[i][j])

        return negmatrix

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #
        # TODO - your code here
        #
        matrixSub = zeroes(self.h, self.w)
        row = []

        for r in range(len(self.g)):
            for c in range(len(self.g[0])):
                matrixSub[r][c] = self.g[r][c] - other.g[r][c]

        return matrixSub

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #
        # TODO - your code here
        #
        product = zeroes(self.h, other.w)

        transposeB = other.T()

        # for c in range(len(other[0])):
        #     new_row = []
        #     for r in range(len(other)):
        #         new_row.append(other[r][c])
        #     transposeB.append(new_row)

        for r1 in range(len(self.g)):
            for r2 in range(len(transposeB.g)):
                product[r1][r2] = dot_product(self.g[r1], transposeB.g[r2])

        return product

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            pass
            #
            # TODO - your code here
            #

            matrixMul = zeroes(self.h, self.w)

            for i in range(len(self.g)):
                for j in range(len(self.g[0])):
                    matrixMul[i][j] = other * self.g[i][j]

            return matrixMul
