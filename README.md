Optimisation of machine learning algorithm

Objective: find the parameters of the K-nearest neighbors classifier that maximise the cross-validation score and the accuracy

Optimisation algorithm: Indicator-Based Evolutionary Algorithm (IBEA)

Parameters to optimise: algorithm, leaf_size, n_neighbors, p, weights
(Please refer to sklearn.neighbors.KNeighborsClassifier documentation).
The decision vector is coded into binary.

Objective: minimise the negative of the cross-validation score and the accuracy score.

To run the optimisation (Matlab):
[Archive,Archive_objectives, X, Xo, samples, samples_objectives] = IBEA_binary(10, 5, 'cost_func', 30, 2, 0.1, 1, 0.01, 0.05)

To save the results:
save IBEA.mat

To visualise the results:
plot_results

To write the best solution to a file:
write_best(index_best)
