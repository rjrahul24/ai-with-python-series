#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install kanren


# In[2]:


pip install sympy


# In[3]:


# Import the necessary functions from the kanren library
from kanren import run, var, fact
from kanren.assoccomm import eq_assoccomm as eq
from kanren.assoccomm import commutative, associative
# Define values that will undertake the addition and multiplication operations
addition = 'add'
multiplication = 'mul'
# Define facts and properties of each operation
fact(commutative, multiplication)
fact(commutative, addition)
fact(associative, multiplication)
fact(associative, addition)
# Declare the variables that are going to form the expression
var_x, var_y, var_z = var('var_x'), var('var_y'), var('var_z')
# Build the correct pattern that the program needs to learn
match_pattern = (addition, (multiplication, 4, var_x, var_y), var_y, (multiplication, 6, var_z))
match_pattern = (addition, (multiplication, 3, 4), (multiplication, (addition, 1, (multiplication, 2, 4)),2))

# Build 3 distinct expressions to test if the function has learnt
test_expression_one = (addition, (multiplication, (addition, 1 , (multiplication, 2, var_x )), var_y) ,(multiplication, 3, var_z )) 
test_expression_two = (addition, (multiplication, var_z, 3), (multiplication, var_y, (addition, (multiplication, 2, var_x), 1)))
test_expression_three = (addition  , (addition, (multiplication, (multiplication, 2, var_x), var_y), var_y), (multiplication, 3, var_z))
# Test the evaluations of the expression on the test expressions
run(0,(var_x,var_y,var_z),eq(test_expression_one,match_pattern))


# In[4]:


run(0,(var_x,var_y,var_z),eq(test_expression_two,match_pattern))


# In[5]:


print(run(0,(var_x,var_y,var_z),eq(test_expression_three,match_pattern)))


# In[6]:


# Since the first two expressions satisfy the expression above, they return the values of individual variables. The third expression is structurally different and therefore does not match

# Running Mathematical Evaluations using SymPy
import math
import sympy
print (math.sqrt(8))


# In[7]:


# Although the Math Square Root function gives an output for the Square Root of 8, we know this is not accurate since the square root of 8 is a recursive, non-ending real number
print (sympy.sqrt(3))


# In[8]:


# Sympy on the other hand, symbolizes the output and shows it as root of 3
# In case of actual square roots like 9, SymPy gives the correct result and not a symbolic answer


# In[16]:


# Import the necessary libraries for running the Prime Number function
from kanren import membero, isvar, run
from kanren.core import goaleval, condeseq, success, fail, eq, var
from sympy.ntheory.generate import isprime, prime
import itertools as iter_one

# Defining a function to build the expression
def exp_prime (input_num): 
    if isvar(input_num):
        return condeseq([(eq, input_num, x)] for x in map(prime, iter_one.count(1)))
    else:
        return success if isprime (input_num) else fail

# Variable to use
n_test = var() 
set(run(0, n_test,(membero, n_test,(12,14,15,19,21,20,22,29,23,30,41,44,62,52,65,85)),( exp_prime, n_test)))


# In[17]:


run(7, n_test, exp_prime(n_test))


# In[18]:


# Code Snippet 2 - Functional Programming


# In[19]:


# Working with Pure Functions in Python
# Pure functions do not change the input list
def pure_func(List): 
    Create_List = [] 
    for i in List: 
        Create_List.append(i**3) 
    return Create_List 
# Test input code
Initial_List = [1, 2, 3, 4] 
Final_List = pure_func(Initial_List) 
print("The Root List:", Initial_List) 
print("The Changed List:", Final_List)


# In[20]:


# Looking at Recursion code in Python
# We implement Recursion to find the sum of a list
def Sum(input_list, iterator, num, counter): 
    if num <= iterator: 
        return counter 
    counter += input_list[iterator] 
    counter = Sum(input_list, iterator + 1, num, counter) 
    return counter 
# Driver's code 
input_list = [6, 4, 8, 2, 9] 
counter = 0
num = len(input_list) 
print(Sum(input_list, 0, num, counter))


# In[21]:


# Python demonstration of high-order functions
def func_shout(text_input): 
    return text_input.upper() 
def func_whisper(text_input): 
    return text_input.lower() 
def greet(func_var): 
    # Store the Function as a variable 
    greet_text = func_var("Hello, I was passed like an argument to a function") 
    print(greet_text) 
greet(func_shout) 
greet(func_whisper)


# In[22]:


# Code Snippet 3: Built-in Complex Functions


# In[23]:


# Implementation of Map() in Python
def addition(num): 
    return num + num
# Implement the function of doubling input numbers
input_numbers = (3, 4, 1, 2) 
final_results = map(addition, input_numbers) 
# This print statment would not return results, and only show the object type
print(final_results) 
# This will return results
for num_result in final_results: 
    print(num_result, end = " ")


# In[24]:


# Python implementation of Filter() function
# Writing a function to filter vowels 
def Check_Vowel(var): 
    vowels = ['a', 'e', 'i', 'o', 'u'] 
    if (var in vowels): 
        return True
    else: 
        return False
test_seq = ['m', 'a', 'i', 'x', 'q', 'd', 'e', 'k'] 
filter_output = filter(Check_Vowel, test_seq) 
print('The extracted vowels are:') 
for s in filter_output: 
    print(s)


# In[25]:


# Writing a lambda function in Python
result_cube = lambda a: a * a*a 
print(result_cube(8)) 
test_list = [4, 2, 1, 3, 5, 6] 
even_test = [a for a in test_list if a % 2 == 0] 
print(even_test)


# In[ ]:




