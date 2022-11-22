import numpy as np

one_d = np.array([1,3,4,5,6,7])
two_d = np.array([[2,4],[4,5],[6,7]])
three_d = np.array([
    [[2,4],[4,5],[6,7]],
    [[2,4],[4,5],[6,7]],
    [[2,4],[4,5],[6,7]]
])

shaped = one_d.reshape(3,2)
changed = np.flip(shaped)
add = two_d + changed
# back_to_one = add.flatten().sum()
back_to_one = add.flatten().max()
# print(add)
#dimension
print(back_to_one.dtype)
print(back_to_one)
diff_type = back_to_one.astype('f')
print(diff_type)
back_to_one_again = np.copy(back_to_one)
sorted = np.sort(back_to_one_again)
print(sorted)

# print(one_d)
# print(three_d)
# print(changed)