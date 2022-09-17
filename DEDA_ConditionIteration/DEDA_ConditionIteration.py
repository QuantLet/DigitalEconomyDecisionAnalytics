"""
DEDA Unit 2
Introducing the Condition and Iteration in Python

Authors: Isabell Fetzer and Junjie Hu
"""


"""
List 
"""
"""
Recap: List items are ordered, changeable, and allow duplicate values.
The first item has the index [0].
Using list makes Python code simple. A skilful handling of lists requires a broad knowledge of functions and methods for lists. 
"""
languages = ['C++', 'Java', 'C#']
python = ['Python 2.7', 'Python 3.6']
len(languages)  # counts items of the list. Output will be 3
len(languages[1])  # 4 items (string characters) in second item of the list 
languages.append('JavaScript')  # adds one element at the end of the list 
# ['C++', 'Java', 'C#','JavaScript']
languages.insert(0, 'Python')  # adds one element at arbitrary place
# ['Python', 'C++', 'Java', 'C#','JavaScript']
languages.extend(python)  # adds multiple elements at the end of the list
# ['Python', 'C++', 'Java', 'C#', 'JavaScript', 'Python 2.7', 'Python 3.6']


"""
List 
"""
"""
Some more built-in methods that can be used on lists in Python.
"""
languages.remove('C#') # removes specific element from list
# ['Python', 'C++', 'Java', 'JavaScript', 'Python 2.7', 'Python 3.6']
languages.pop(-1)  # removes the last element of the list …
# ['Python', 'C++', 'Java', 'JavaScript', 'Python 2.7']
del languages[-1] # … and this does so too
# ['Python', 'C++', 'Java', 'JavaScript']
languages_str = '; '.join(languages)  # converts list to string
# Python; C++; Java; JavaScript
languages_new = languages_str.split('; ')  # converts string to list 
# ['Python', 'C++', 'Java', 'JavaScript']


"""
List 
"""
"""
Some more built-in methods that can be used on lists in Python.
"""
numbers = [4, 11, 2, 95, 0]
# basic search in list
min_num = min(numbers)  # 0
max_num = max(numbers)  # 95
sum_num = sum(numbers)  # 112
index_num = numbers.index(11)   # 1
# reorder list
numbers.reverse()  # [0, 95, 2, 11, 4]
numbers.sort()  # [0, 2, 4, 11, 95]
numbers.sort(reverse=True)  # [95, 11, 4, 2, 0]
sorted_number = sorted(numbers)  # [95, 11, 4, 2, 0]
# formatting numbers of list to 2 digits
numbers_db = [format(num, '02d') for num in numbers]
# ['95', '11', '04', '02', '00']


"""
tuple
"""
"""
A tuple is an ordered and immutable sequence of Python objects. A tuple is explicitly created by () and its items are separated by commas. Implicit creation of a tuple happens when we set up multiple variables at one line. This is known as tuple packing.
"""
t = () # empty tuple
t1, t2, t3 = 'Guido van Rossum', 65, 'Dutch' # tuple is implicitly created
# tuple vs. list 
creator_tuple = ('Guido van Rossum', 65, 'Dutch')
creator_list = ['Guido van Rossum', 65, 'Dutch']
# accessing tuples works like accessing lists, however … 
# … tuple do not support item assignment while lists do
# creator_tuple[0] = 'G.v. Rossum' # TypeError! Python stops.
# creator_list[0] = 'G.v. Rossum' # ['G.v. Rossum', 65, 'Dutch']

                
"""
set and frozenset
"""
"""
Python also provides two set types, set and frozenset. A Set is an unordered collection data type that is iterable and has no duplicate items. While the set type is mutable, frozenset is not. 
"""
first_set = set()  # empty set
second_set = {1,2,3} # also a set
not_a_set = {} # empty dict
first_frozenset = frozenset()  # empty frozenset
second_frozenset = ({1,2,3}) # also a frozenset
third_frozenset = ([4,5,6]) # again: a frozenset
# four sets
ordered_mutable = {'list', 'list'} # sets will drop duplicated elements automatically
ordered_immutable = {'string', 'tuple'}
unordered_immutable = {'dictionary', 'frozenset'}
unordered_mutable = {'set'}
# merge 2 sets into 1 w/o duplicates
ordered = ordered_mutable.union(ordered_immutable) # {'string', 'tuple', 'list'}

                
"""
set and frozenset
"""
"""
The common set of arithmetic operators are available as methods for sets. 
"""
immutable = ordered_immutable.union(unordered_immutable) # a | b
# {'frozenset', 'dictionary', 'string', 'tuple'}
inter = immutable.intersection(ordered) # a & b
# {'string', 'tuple'}
diff1 = immutable.difference(ordered)  # a - b
# {'frozenset', 'dictionary'}
diff2 = ordered.difference(immutable)  # b - a
# {'list'}
sym = ordered.symmetric_difference(immutable)  # a ^ b 
# {'frozenset', 'dictionary', 'list'}
disj = ordered.isdisjoint(immutable)  
# False

               
"""
dict
"""
"""
Recap: A dictionary, dict type, is formed by 2 parts, a unique key and values.
"""
profile = {'name': 'Guido von Rossum', 'birth': '31-01-1956', 'gender': 'male', 'height': 1.73} 
print(profile['name'])  # find the value by using the key
# Guido von Rossum
keys = list(profile.keys()) # get all the keys and put into a list 
# ['name', 'birth', 'gender', ‘height']
values = list(profile.values())  # get all values and put into a list
# ['Guido von Rossum', '31-01-1956, 'male', 1.73]
key_value = list(profile.items()) # get all key-value pairs and put into a list
# [('name', 'Guido von Rossum'), ('birth', '31-01-1956'), ('gender', 'male'), ('height', 1.73)]
profile['net worth'] = 10000000 # add a key with value
profile['name'] = 'G.v.Rossum' # change value of a key
profile.update({'height': '5ft 8', 'nationality': 'Dutch'}) # alter multiple keys & values

                
"""
Conditional Execution
"""    
"""
Conditional programming, or branching, is about evaluating conditions: if …, then …  The operator "is" is used to evaluate if two 2 variables pointed to the same object, "==" is used to evaluate the equivalence of 2 variables.
"""
seq1 = [True, False, 0, 1, -1, 100, '', (), [], {}, [False], [True]]
seq2 = [True, False, 0, 1, -1, 100, '', (), [], {}, [False], [True]]
print(seq1 == seq2)  # True
print(seq1 is seq2)  # False
seq3 = seq2 # reassigning 
print(seq3 == seq2)  # True
print(seq3 is seq2)  # True
# False, 0, empty sequence will be evaluated as False
for item in seq1:
    if item: # equals: if bool_smb == TRUE
        print(item, ": This is True")
    elif not item:
        print(item, ": This is False")
              
              
"""
Iteration
"""
"""
An iterable is an object that can be iterated upon, meaning that it can return its times one by one. Example of iterables include all sequence types (i.e. lists, str, tuple) and some non-sequence types, such as dict. The for loop automatically creates a temporary variable to hold the iteration for the duration of the loop.
"""
repos = ['Introduction', 'ConditionIteration', 'WebScrapingIntro'] 
units = ['Unit1', 'Unit2', 'Unit3']
for i in range(len(repos)): # (1) inefficient and not pythonic!
 repo = repos[i] # i iterates over the two list of positions (0,1,2)
 unit = units[i]
 print(unit, repo)
for idx, repo in enumerate(repos): # (2) better, still not beautiful!
    unit = units[idx]
    print(unit, repo)
for unit, repo in zip(units, repos): # (3) beautiful Python code!
   print(unit, repo) # for even more elegance:  print(f'{unit}: {repo}')
              
              
"""
Iteration
"""
"""
Looping in a list follows a basic syntax: func(ele) for func(ele) in a_list if func(ele)
"""
# example 1
intros = [intro + ' Chapter' for intro in repos if intro.startswith('Intro') or intro.endswith('Intro')] 
print(intros) # ['Introduction Chapter', 'WebScrapingIntro Chapter']
# example 2
for repo, unit, intro  in zip(repos, units, intros):
    if repo + ' Chapter' != intro:
        print(f'{unit} is not of introductory nature.')
        break # stops whole loop
    elif repo + ' Chapter' == intro:
        continue # rejects all remaining statements and continues with next iteration
    else:
        print('Something went wrong.')

              
"""
Iteration
"""
"""
Special Loops
"""
our_class = ['Digital', 'Economy', 'and', 'Decision', 'Analytics']
# a loop embed in another loop
for word in our_class:
    if word == 'in':
        break
    else:
        for letter in word:
            if letter.isupper() is True:
                print(letter)

# infinite loop
# This will run infinitely if you remove the comment sign #
#num = 0
#while True:
    #print(num)
    #num +=1

             
"""
Import Packages
"""
"""
Python provides us with four different way of importing modules of an external package. Make sure to have the package pre-installed with pip install <module>.
"""
# (1) import <module_name> 
# for the import of the whole module 
import numpy
# (2) import <module_name> as <self-chosen_name> 
# for the import of the whole module with an alias
import numpy as np
# (3) from <module_name> import <name(s)> 
# for the import of an specific object of the module 
from numpy import array
from numpy import * # imports everything from the module numpy
# (4) from <module_name> import <name(s)> as <self-chosen_name>
# for the import of an specific object of the module with an alias
from numpy import array as ay
             
             
"""
Read & Write Files
"""
"""
Opening a file in Python is very intuitive: We call the build-in function open(), pass the filename, and specific the read mode 'r' or 'w' the write mode. To rest assure that the close() function will be called at the end, whatever error might occur, we disseminate our code with a try/finally block. readlines() will put content into a list, every line is a string in the list.
"""
try:
    shakespeare = open('shakespeare.txt', 'r', encoding='utf-8')
    for line in shakespeare.readlines():
        print(line.strip()) #removes leading and trailing whitespaces 
except:
    print("Reading the file didn't work!")
else:
    shakespeare.close()
# simplified: 
try:
    shakespeare = open('shakespeare.txt') # 'r' and 'utf-8' are set by default
    for string in shakespeare: # no need to call readlines() explicitly, it’s the default
        print(string.strip())
except:
    print("Reading the file didn't work!")
else:
    shakespeare.close()
                       
                       
"""
Read & Write Files
"""
"""
The context manager is an alternative to wrapping out code into try/finally blocks, while ensuring the secure open and close of a file. We use the syntax "with open() as <self-chosen name>” to load the content. The true beauty of this code lies in the fact that the close() function will be called automatically for us, even in case of errors. 
"""
with open('shakespeare.txt') as shakespeare:
    for string in shakespeare:
        print(string.strip())
# shakespeare.close() will be called automatically and is redundant    
with open('shakespeare.txt', 'a' ) as shakespeare:
    shakespeare.write('BY WILLIAM SHAKESPEARE')
# BY WILLIAM SHAKESPEARE will be added to last line of the .txt file 

                       
"""
Input and Output
"""
"""
The Package Pandas supports most of the common structured data formats.
The read_csv method read files as DataFrame type, which is a very powerful data structure that allow us to do almost everything with the data.
"""
import pandas as pd
# AAPL.csv file can be found in GitHub Repository of this Unit
apple_stock = pd.read_csv('AAPL.csv', index_col='date', parse_dates=True)
type(apple_stock) # <class 'pandas.core.frame.DataFrame'>
# Getting the data for year 2021
apple_stock_2021 = apple_stock.loc[apple_stock.index.year == 2021,]
 
                       
"""
Input and Output
"""
# shape of DataFrame
print(apple_stock_2021.shape) # (198, 8)
# sorting by the value of volume
apple_stock_2021.sort_values(by='volume', ascending=False, inplace=True)
print(apple_stock_2021)
# check monotonicity increasing through time
print(apple_stock_2021.index.is_monotonic_increasing) # False
# sorting by index
apple_stock_2021.sort_index(axis=0, ascending=True, inplace=True)
print(apple_stock_2021)
# reset index as numeric
apple_stock_2021.reset_index(drop=False, inplace=True)
print(apple_stock_2021)
  
                       
"""
Input and Output
""" 
# check null values
nan_rows = apple_stock_2021[apple_stock_2021['open'].isna()]
# remove null values
apple_stock_2021.dropna(axis=0, how='any', subset=['volume', 'close'], inplace=True)
# set a column as index
apple_stock_2021.set_index(keys='date', drop=True, append=False, inplace=True)
# duplicates timestamp operations
print(apple_stock_2021.index.has_duplicates)
apple_stock_2021.index.duplicated() 
                       
                       
"""
Input and Output
"""
# operations
# way1, potential risk
way1 = apple_stock_2021.drop_duplicates(keep='first', subset=['low', 'high', 'open', 'close', 'volume'])
# way2, drop by index, row operation. ~, take inverse
way2 = apple_stock_2021[~ (apple_stock_2021.index.duplicated())]
# a simple build-in plot function of pandas
apple_stock_2021['open'].plot() 
# Save the new data as json format
apple_stock_2021.to_json('AAPL_2021.json')
                         
