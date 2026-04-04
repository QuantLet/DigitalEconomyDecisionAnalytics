"""
DEDA Unit 1
Python Basic Syntax and Data Structure Introduction

Author: Junjie Hu
Editor: Isabell Fetzer and Lucas Umann
"""


"""
numeric
"""
"""  
This is a multi-line comment block. It is set in-between six quotation marks.
Python provides a straight way for numerical operations.
Try the basics:
+, -, *, /, %, **, // 
"""
# This is a one-line comment in Python. It extends to the end of the physical line.

a = 5  # initiates integer variable a which the value 5 is assigned to
b = 3  # another integer variable b which the value 3 is assigned to
b = 5  # the old value of b is overwritten by 5
a, b = 7, 3  # initialising two variables at once: a takes the value  7; b takes 3
a += 1  # short for a = a+1 so again a new value is assigned to a


"""
functions
"""
"""  
Python provides versatile set of functions for variables.
"""
print(a)  # the fct. print() prints out the value of a; the output is 8
c = round(a / b)  # the fct. round() rounds 8/3 and assigns the output to c
# Function can also take arguments, which allow us further specifications.
c = round(a / b, ndigits=4)  # ndigits specifies the number of decimal places
print(c)  # output is 2.6667
c = round(a / b, 3)  # arguments do not need to be spelled out, if placed correctly
print(c)  # output is 2.667


"""
string
"""
""" 
String is a basic type in python, it's commonly used and very powerful.
"""
w = 'Hello'  # initiates string variables by single quotation marks
x = "World"  # … or by double quotation marks
print(w + x)  # HelloWorld
y = '20'  # numbers can also be assigned as strings too
e = '10'
print(y + e)  # concatenates strings so output is ‘2010’
f = int(y) + int(e)  # the fct. int() converts strings to integers, f is 30 now
# formatting strings
# using f '….'  string, you can write variable names inside the brackets {}, directly.
greeting = f'{w} Python-{x}.'  # Hello Python-World.
greeting = f'{w} {x.upper()}.'  # Hello WORLD.


"""
comparison
"""
"""
Python provides a straight way for numerical operations. Try comparison operations ==,<,<=,>,>=,!=
"""
a = 5
a == 5  # checks if a equals 5; output will be True
a != 7  # checks if a is not equal to 7; output will be True
a <= 7  # checks if a is smaller or equal 7; output will be True
a > 5  # False
a = 'A'  # old integer value of a (which is 7) is overwritten by string value ‘A’
b = 'B'  # old integer value of b (which is 3) is overwritten by string value ‘B’
a == b  # False
a != b  # True


"""
list
"""
"""
List is a versatile Python data type to group values.
In Python a list is created by placing all the items inside square brackets [] , separated by commas.
"""
my_list = []  # initiates an empty list, which we choose to call “my_list”
my_list = [2, 3, 5, 7, 11]  # adds elements to the list
print(my_list)  # output: [2, 3, 5, 7, 11]
# indexing
my_list[0]  # the first element of my_list is called, which is 2
my_list[-1]  # the last element of my_list is called, which is 11
# slicing
my_list[:2]  # calls the first two elements of my_list [2, 3]
my_list[-3:]  # calls the last three elements [5, 7, 11]
# appending
my_list.append(13)  # [2, 3, 5, 7, 11, 13]
my_list.extend([17, 19])  # [2, 3, 5, 7, 11, 13, 17, 19]


"""
list
"""
"""
Lists can contain different types, e.g. strings, numbers, 
functions, lists, …
"""
my_list = ['Welcome', 'to', 'Python']  # adds three string elements to the list
print(my_list[0])  # Welcome
print(my_list[1])  # to
print(my_list)  # [‘Welcome', 'to', ‘Python']
# using for loop to iterate all elements in the list:
for word in my_list:  # word we choose randomly, any other name works too
    print(word)
# the output is:
# Welcome
# to
# Python


"""
list
"""
"""
Functions can help when working with lists.
"""
# to print it out in one line we need to apply the fct join() to our list:
list_as_sentence = '*'.join(my_list)  # sets * as separators
print(list_as_sentence)  # Welcome*to*Python
list_as_sentence = ' '.join(my_list)  # sets white spaces as separators
print(list_as_sentence)  # Welcome to Python
# Try to execute this code and see what happens.
list_sliced = list_as_sentence[0:12]  # slices string by indices
print(f'{list_sliced}.')  # Welcome to P.
# Try changing the indices to negative.
sentence_big = list_as_sentence.upper()
print(sentence_big)  # WELCOME TO PYTHON


"""
operations
"""
"""
Checking for possible operations: 
"""
# by using dir() function, or help() function; you may see all ops on the str object
dir(str)
# or a str instance
dir(sentence_big)
# likewise,
help(str)


"""
dictionary
"""
"""
Dictionaries are used to store data values in key:value pairs. A dictionary is a collection which is unordered, changeable and does not allow duplicates. 
"""
course = dict(name='DEDA', unit=0)  # {'name': 'DEDA', 'unit': 0}
course = {'name': 'DEDA', 'unit': 0}  # alternative
# accessing
course['unit']  # 0
# assigning
course['unit'] = 1
# get keys
course.keys()  # ['name', 'unit']
course.values()  # ['DEDA', 1]
# adding values
course.update({'lecturers': ['Chen', 'Härdle']})
# {'name': 'DEDA', 'unit': 1, 'lecturers': ['Chen', 'Härdle']}


"""
if/elif/else
"""
"""
Control Flow Tools: if/elif/else. 
Comparison statements can be used. 
"""
x = 10  # initialising a numeric variable x
if x < 0:  # checking for the value of x
    print('Negative Value')  # note the four additional level of indentation (4 spaces)
elif x == 0:
    print('Zero')
else:
    print('Positive Value')

    
"""
if/elif/else
"""
"""
Control Flow Tools: if/elif/else
"""
# Conditions can be combined or altered with: and, or, not, is, is not
p = [2, 3, 5, 7, 11]  # initialising a list p
3 in p  # True
3 in p and 4 in p  # False
3 in p or 4 in p  # True
# Empty/Missing Values can be initialised by the term “None”
y = None  # initialising an empty variable x
if y is not None:  # alternatively: if not y is None: ….
    print('Value is not None')
else:
    print('Value is None')

    
"""
for loop
"""
"""
For Loop can iterate over all elements. The i is a self-chosen variable to use to represent the current increment in a loop. 
"""
l = list([1, 2, 3, 4, 5])
for i in l:
    print(i * 2, end=' ')  # output: 2 4 6 8 10
for i in range(6):  # iterates from 0 to 5
    if i == 3:
        continue  # skip 3*2 in output
    print(i * 2, end=' ')  # output:  0 2 4 8 10
for i in 'DEDA':
    print(i, end=' ')  # output: D E D A
d = dict(a=1, b=2)
for k, v in d.items():
    print('\n{} has value {}'.format(k, v))
# a has value 1
# b has value 2


"""
while loop
"""
"""
With the while loop we can execute a set of statements as long as a condition is true. ATTENTION: Make sure not to have infinite loop (loop with tautology in condition).
"""
# Fibonacci series: sum of two preceding numbers defines next number
a, b = 0, 1
while b < 100:
    print(b, end=' ')
    a, b = b, b + a
# 1 1 2 3 5 8 13 21 34 55 89


"""
while loop
"""
"""
The < break > statement in Python terminates the current loop and resumes execution at the next statement.
"""
fib = [0, 1]  # a list called fib
while True:
    # appends the sum of the last 2 elements of our list to our new list
    fib.append(sum(fib[-2:]))
    if fib[-1] > 100:
        break
print(fib)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


"""
function
"""
"""
Define your own function using def, followed by a name, parameters in parentheses (), a colon, and a block of code.
"""
def my_first_fct():
    print('This is my first function.')  # remember the four indentation
my_first_fct()  # call the function
# another function with two parameter m and n
def my_second_fct(m, n):
    mn = m % n
    return print(f'The remainder of the devision of {m} by {n} is {mn}.')
my_second_fct(4, 3)  # call the function and pass the variables m = 4 and n = 3
my_second_fct(7, 2)  # call the function again and pass other variables


"""
function
"""
"""
Another example for functions
"""
def square_numeric(x):  # Squares numeric x
    return x ** 2
def square_iterable(x):  # Squares numerics in iterable x
    list = []
    for i in x:
        list.append(square_numeric(i))
    return list
def square_iterabel_short(x):  # Squares numerics in iterable x
    return [square_numeric(i) for i in x]
x = [1, 2, 3, 4, 5]
square_iterable(x)  # [1, 4, 9, 16, 25]
square_iterabel_short(x)  # [1, 4, 9, 16, 25]


