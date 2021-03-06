Author: Luke Styles, style011@umn.edu

License: GPL V2

This repository contains Python code written as coursework for the couse CSCI 5525: Machine Learning (professor: Arindam Bannerjee) at the University of Minnesota in Fall 2013. All code is the original work of the author. I have included [the UCI Mushroom dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom) for demo purposes, but the code should perform on any numerical data.

Dependency: [numpy](http://www.numpy.org)

Included Functions ----
    
-> hw2.dstumpIG('Mushroom.csv') - Random forest of decision stumps using information gain.
    
-> hw2.dstumpGI('Mushroom.csv') - Random forest of two-layer decision trees using Gini Index.
    
-> hw2.myAdaBoost('Mushroom.csv', num_stumpts) - AdaBoost using decision stumps and info gain.
