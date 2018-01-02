#  Natural language processing - ex2
## perceptron algorithm implementation.
the idea was to learn how dependency graph works and how we can (using a feature extraction function)
create a W weight vector that fits for all our train data and help us to build a tree from random sentence.

## implementation.
We choosed to create our own SparseVector because the CSR_Matrix and the lil_matrix took too much resources.
(maybe we will optimize some of the functionlaities soon)

the mst algorithm is implemented using networkX library. (and so all our graph representation)
we found this way much easier and clear.

to correlation between trees are made using (sets intersection length) / (set length) 
while the set contain all the relevat edges

to save some runtime, we are calculating w in this order
w1 = 0 + (t1` - t1), w2 = w1 + (t2` - t2)...
so wn = 0 + (t1` - t1) + (t2` - t2) .... (tn` - tn)
(instead of summing pretty long sparse vectors we are summing ~100 elements vectors)


## Timing
the init process take around 5 seconds.
the train process take around 1sec for 100 sentences.
the test process take around 3 seconds.

