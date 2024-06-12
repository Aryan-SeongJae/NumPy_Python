import numpy as np


arr = np.array([1,2,3,4,5])
print(arr)
print(type(arr))

# 0-D array
arr1 = np.array(42)
print(arr1)

# 1-D array
arr2 = np.array([1,2,3,4,5])
print(arr2)

# 2-D array
arr3 = np.array([[1,2,3],[4,5,6]])
print(arr3)

# 3-D array
arr4 = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
print(arr4)


# Check the number of dimensions

print(arr.ndim)
print(arr1.ndim)
print(arr2.ndim)
print(arr3.ndim)
print(arr4.ndim)

# Higher Dimensional Arrays
arr5 = np.array([1,2,3,4], ndmin=5)
print(arr5)
print('number of dimensions :', arr5.ndim)

# NumPy Array Indexing
# Access 1-D Arrays
arr6 = np.array([1,2,3,4])
print(arr6[0])
print(arr6[1])
print(arr6[1:4:2])

# Access 2-D Arrays
arr7 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('2nd element on 1st dim: ', arr7[0, 1])

# Access 3-D Arrays
arr8 = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
print(arr8[0, 1, 2])

# Negative Indexing
arr9 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('Last element from 2nd dim: ', arr9[1, -1])
print('Last element from 1st dim: ', arr9[0, -1])

# NumPy Array Slicing
arr10 = np.array([1,2,3,4,5,6,7])
print(arr10[1:5])
print(arr10[4:])
print(arr10[:4])
print(arr10[-3:-1]) # Negative Slicing
print(arr10[1:5:2])
print(arr10[::2])

# Slicing 2-D Arrays
arr11 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(arr11[1, 1:4])
print(arr11[0:2, 2])
print(arr11[0:2, 1:4])

# NumPy Data Types
arr12 = np.array([1,2,3,4])
print(arr12.dtype)

arr13 = np.array(['apple', 'banana', 'cherry'])
print(arr13.dtype)

arr14 = np.array([1,2,3,4], dtype='S')
print(arr14)
print(arr14.dtype)

arr15 = np.array([1,2,3,4], dtype='i4')
print(arr15)

# Converting Data Type on Existing Arrays
arr16 = np.array([1.1, 2.1, 3.1])
newarr = arr16.astype('i')
print(newarr)
print(newarr.dtype)

newarr1 = np.array([1,0,3])
newarr1 = newarr1.astype(bool)

# NumPy Array Copy vs View
arr17 = np.array([1,2,3,4,5])
x = arr17.copy()
arr17[0] = 42
print(arr17)
print(x)

arr18 = np.array([1,2,3,4,5])
y = arr18.view()
arr18[0] = 42
print(arr18)
print(y)

# Check if Array Owns it's Data

arr19 = np.array([1,2,3,4,5])
x = arr19.copy()
y = arr19.view()
print(x.base)
print(y.base)

# NumPy Array Shape
arr20 = np.array([[1,2,3,4], [5,6,7,8]])
print(arr20.shape)

# Reshaping arrays
arr21 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr2 = arr21.reshape(4,3)
print(newarr2)

arr22 = np.array([1,2,3,4,5,6,7,8])
newarr3 = arr22.reshape(2,2,2)
print(newarr3)

arr23 = np.array([1,2,3,4,5,6,7,8])
newarr4 = arr23.reshape(2,2,-1)  # Unknown dimension is calculated based on the length of the array and remaining dimensions
print(newarr4)

#Array Iteration

# Iterating 1-D Arrays
arr24 = np.array([1,2,3])
for x in arr24:
    print(x)

# Iterating 2-D Arrays
arr25 = np.array([[1,2,3], [4,5,6]])
for x in arr25:
    print(x)

for x in arr25:
    for y in x:
        print(y)

# Iterating 3-D Arrays
arr26 = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
for x in arr26:
    print(x)

for x in arr26:
    for y in x:
        for z in y:
            print(z)

# Iterating Arrays Using nditer()
arr27 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
for x in np.nditer(arr27):
    print(x)

# Iterating Array With Different Data Types
arr28 = np.array([1,2,3])
for x in np.nditer(arr28, flags=['buffered'], op_dtypes=['S']):
    print(x)

# Iterating With Different Step Size
arr29 = np.array([[1,2,3,4], [5,6,7,8]])
for x in np.nditer(arr29[:, ::2]):
    print(x)

# Enumerated Iteration Using ndenumerate()
arr30 = np.array([1,2,3])
for idx, x in np.ndenumerate(arr30):
    print(idx, x)

arr31 = np.array([[1,2,3,4], [5,6,7,8]])
for idx, x in np.ndenumerate(arr31):
    print(idx, x)

# NumPy Joining Array
# Joining NumPy Arrays
arr32 = np.array([1,2,3])
arr33 = np.array([4,5,6])
arr34 = np.concatenate((arr32, arr33))
print(arr34)

arr35 = np.array([[1,2], [3,4]])
arr36 = np.array([[5,6], [7,8]])
arr37 = np.concatenate((arr35, arr36), axis=1)
print(arr37)

# Joining Arrays Using Stack Functions
arr38 = np.array([1,2,3])
arr39 = np.array([4,5,6])
arr40 = np.stack((arr38, arr39), axis=1)
print(arr40)


# Stacking Along Rows
arr41 = np.array([1,2,3])
arr42 = np.array([4,5,6])
arr43 = np.hstack((arr41, arr42))
print(arr43)

# Stacking Along Columns
arr44 = np.array([1,2,3])
arr45 = np.array([4,5,6])
arr46 = np.vstack((arr44, arr45))
print(arr46)

# Stacking Along Height (depth)
arr47 = np.array([1,2,3])
arr48 = np.array([4,5,6])
arr49 = np.dstack((arr47, arr48))
print(arr49)

# NumPy Splitting Array
# Splitting NumPy Arrays
arr50 = np.array([1,2,3,4,5,6])
newarr5 = np.array_split(arr50, 3)
print(newarr5)

# Splitting 2-D Arrays
arr51 = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
newarr6 = np.array_split(arr51, 3)
print(newarr6)

# Splitting Along Columns
newarr7 = np.hsplit(arr51, 2)
print(newarr7)

# NumPy Searching Arrays
# Searching Arrays
arr52 = np.array([1,2,3,4,5,4,4])
x = np.where(arr52 == 4)
print(x)

# Find the indexes where the values are even
arr53 = np.array([1,2,3,4,5,6,7,8])
x = np.where(arr53%2 == 0)
print(x)

