# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: t1_test_iterable.py
@Time: 2020-10-07 17:55
@Last_update: 2020-10-07 17:55
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""

a = range(20)
print(len(a))
print(type(a))

b = (i for i in range(20))
print(len(b))
print(type(b))