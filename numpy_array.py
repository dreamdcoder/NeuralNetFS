import numpy as np

arr=[1,2,3]

print(f"type is {type(arr)}")
arr1= np.array(arr)
print(f"{arr1}type is {type(arr1)} shape is {arr1.shape}")
arr2 = np.array([arr1])
print(f"{arr2}type is {type(arr2)} shape is {arr2.shape}")
# A row vector is a matrix whose first dimension’s size (the number of rows) equals 1 and the
# second dimension’s size (the number of columns) equals n — the vector size. In other words, it’s a
# 1×n array or array of shape (1, n):
arr3=np.expand_dims(arr1,axis=0)
print(f"{arr3}type is {type(arr3)} shape is {arr3.shape}")
arr4=np.expand_dims(arr2,axis=0)
print(f"{arr4}type is {type(arr4)} shape is {arr4.shape}")
# Where np.expand_dims() adds a new dimension at the index of the axis.
# A column vector is a matrix where the second dimension’s size equals 1, in other words, it’s an
# array of shape (n, 1):
arr5=np.expand_dims(arr1,axis=1)
print(f"{arr5}type is {type(arr5)} shape is {arr5.shape}")
#Transpose of row vector
arr6=arr2.T
print(f"{arr6}type is {type(arr6)} shape is {arr6.shape}")