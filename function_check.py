# import mmpose.apis

# print(dir(mmpose.apis))
string = 'yeah ah'
import copy
# print(string.rfind('a'))

list1 = [1,2,3,4]
list2 = copy.deepcopy(list1)
list2 = [1,3,4,5]
# print(list1)
# print(list2)

import functools

def add_func(a,b):
    return a + b

def mul_func(a,b):
    return a * b

a_list = [i for i in range(1, 11)]

num = functools.reduce(add_func, a_list)
# print(num)
# num = functools.reduce(lambda x,y,z : (x+y) // z, a_list)
# print(num)

word_list = ['apple','applc']
# print(sorted(word_list))

print(int(input(),base=2))
