#  Natural language processing - ex2
## perceptron algorithm implementation.
the idea was to learn how dependency graph works and how we can (using a feature extraction function)
create a W weight vector that fits for all our train data and help us to build a tree from random sentence.

## implementation.
We choosed to create our own SparseVector because the CSR_Matrix and the lil_matrix took too much resources.
(maybe we will optimize some of the functionlaities soon)

