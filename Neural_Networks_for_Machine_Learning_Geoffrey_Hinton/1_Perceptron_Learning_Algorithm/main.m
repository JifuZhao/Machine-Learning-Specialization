clc; clear all;

%% load the dataset1
load dataset1

% learn weights through Perceptron algorithms
w = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)

%% load the dataset1
load dataset2

% learn weights through Perceptron algorithms
w = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)

%% load the dataset1
load dataset3

% learn weights through Perceptron algorithms
w = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)

%% load the dataset1
load dataset4

% learn weights through Perceptron algorithms
w = learn_perceptron(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas)