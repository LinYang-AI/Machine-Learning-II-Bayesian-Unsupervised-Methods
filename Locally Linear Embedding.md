# Locally Linear Embedding


Locally Linear Embedding (LLE) is a nonlinear dimensionality reduction technique that involves several mathematical calculations. Here is a high-level overview of the key steps involved:

1. Construct the neighborhood graph: For each data point in the high-dimensional space, identify its k-nearest neighbors and construct a graph where each data point is connected to its neighbors.
2. Compute the reconstruction weights: For each data point, compute a set of weights that can be used to reconstruct the point as a linear combination of its neighbors. This involves solving a system of linear equations that minimizes the difference between the distances between the original data points and the distances between the reconstructed points.
3. Compute the lower-dimensional representation: Use the reconstruction weights to find a lower-dimensional representation of the data that preserves the relationships between neighboring points. This involves using classical multidimensional scaling (MDS) or other dimensionality reduction techniques to find a lower-dimensional embedding that minimizes the difference between the distances between the original data points and the distances between the reconstructed points.

Mathematical deduction:

1. Construct the neighborhood graph:
   Let $X$ be a matrix of size ($n \times m$) that contains the high-dimensional data, where $n$ is the number of data points and m is the dimensionality of the data. For each data point $x_i$, identify its k-nearest neighbors and compute the corresponding weights $W_{i,j}$ such that:
   $W_{i,j} = 0$ if $x_j$ is not one of the k-nearest neighbors of $x_i$
   $W_{i,j} = argmin_w ||x_i - w_j||^2$ subject to $\sum{w} = 1$ and $W_{i,k} = 0 $for all $k$ not equal to $j$
   Here, the weights are chosen such that the reconstruction of $x_i$ is a linear combination of its neighbors:
   $x_i = \sum_j W_{i,j} * x_j$
2. Compute the lower-dimensional representation:
   Let $Y$ be a matrix of size ($n \times d$) that contains the lower-dimensional representation of the data, where d is the desired dimensionality of the embedding. Use classical MDS or other dimensionality reduction techniques to find a matrix $Y$ that minimizes the following cost function:
   $J(Y) = \sum_i ||y_i - \sum_j W_{i,j} * y_j||^2$
   Here, the cost function measures the difference between the distances between the original data points and the distances between the reconstructed points in the lower-dimensional space.
