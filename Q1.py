import numpy as np


#A. Random vector of size 15 with integers in range(1-20):
vector = np.random.randint(1, 21, 15)
print("Actual Vector")
print(vector)

# Reshape the array to 3 by 5:
a= vector.reshape((3, 5))
print("Array which is reshaped:")
print(a)

# Replace the max in each row with 0
a[np.arange(len(a)), np.argmax(a, axis=1)] = 0
print("Replacing max in each row by 0:")
print(a)

# Creating a 2-dimensional array of size 4*3
a_2d = np.zeros((4, 3), dtype=np.int32)
print("Array Shape:", a_2d.shape)
print("Array Type:", type(a_2d))
print(" Data Type:", a_2d.dtype)

#B. Computing Eigen values and right Eigen vectors
K = np.array([[3, -2], [1, 0]])

eval, evector = np.linalg.eig(K)

# Print the results
print("Eigenvalues:", eval)
print("Right eigenvectors:")
print(evector)

#C. Compute the sum of the diagonal element of a given array.

arr1 = np.array([[0, 1, 2],
                [3, 4, 5]])

diagonal_sum = np.trace(arr1)
print(diagonal_sum)

#D. New shape of an array without changing its contents:

arr2 = np.array([[1, 2],
                [3, 4],
                [5, 6]])

updated_shape1 = arr2.reshape(2, 3)
updated_shape2 = arr2.reshape(3, 2)

print(updated_shape1)
print(updated_shape2)
