#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='smartlearner',
    version='0.0.1',
    author='Marc-Alexandre Côté, Adam Salvail, Mathieu Germain',
    author_email='smart-udes-dev@googlegroups.com',
    url='https://github.com/SMART-Lab/smartpy',
    packages=find_packages(),
    license='LICENSE',
    description='A machine learning library built with researchers in mind.',
    long_description=open('README.md').read(),
    install_requires=['theano']
)
