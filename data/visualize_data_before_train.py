import numpy as np

print("========= bow test =============")
data = np.load('bow_test.npy', allow_pickle=True)
print(data)

print("========= bow test 1=============")
data_1 = np.load('bow_test_1.npy', allow_pickle=True)
print(data_1)


print("========= bow test 2=============")
data_2 = np.load('bow_test_2.npy', allow_pickle=True)
print(data_2)


print("========= bow train=============")
data_train = np.load('bow_train.npy', allow_pickle=True)
print(data_train)
