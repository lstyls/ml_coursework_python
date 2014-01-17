Author: Luke Styles, style011@umn.edu
License: GPL V2

This repository contains Python code written as coursework for the couse CSCI 5525: Machine Learning (professor: Arindam Bannerjee) at the University of Minnesota in Fall 2013. I have included [the UCI Mushroom dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom) for demo purposes, but the code should perform on any numerical data.

Dependency: [numpy](http://www.numpy.org)

Included Functions:
    
    -> dstumpIG('Mushroom.csv') - Random forest of decision stumps using information gain.
    
    -> dstumpGI('Mushroom.csv') - Random forest of two-layer decision trees using Gini Index.
    
    -> myAdaBoost('Mushroom.csv', num_stumpts) - AdaBoost using decision stumps and info gain.
