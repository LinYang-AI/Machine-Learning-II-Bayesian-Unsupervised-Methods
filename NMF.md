# Non-Negative Factorization


Non-negative matrix factorization (NMF) is a matrix factorization method that decomposes a non-negative matrix $V$ into two non-negative matrices $W$ and $H$ such that $V ≈ WH$, where $V$ is an $n \times m$ non-negative matrix, $W$ is an $n \times k$ non-negative matrix, $H$ is a $k \times m$ non-negative matrix, and $k$ is the number of latent features.

The goal of NMF is to find $W$ and $H$ such that their product approximates the original matrix $V$ as closely as possible, subject to the constraint that all elements of $W$ and $H$ must be non-negative. The non-negativity constraint ensures that the resulting factors are additive, which makes them more interpretable.

The optimization problem of NMF can be formulated as minimizing the Frobenius norm of the difference between $V$ and the product of $W$ and $H$ subject to the non-negativity constraint:

$\min ||V - WH||_F^2$, $s.t. W \geq 0, H \geq 0$

where $||.||_F$ is the Frobenius norm.

To solve this optimization problem, we can use an iterative algorithm such as alternating least squares (ALS). In ALS, we fix one matrix (either $W$ or $H$) and update the other matrix until convergence, and then alternate the fixed and updated matrices until convergence.

To update $W$, we fix $H$ and minimize the objective function with respect to $W$. To do this, we take the derivative of the objective function with respect to $W$ and set it to zero:

$\frac{d}{dW} ||V - WH||_F^2 = -2H^T(V - WH) + 2HH^TWH = 0$

Solving for $W$, we get:

$W = WHH^T / (HH^T)$

To update $H$, we fix $W$ and minimize the objective function with respect to $H$. To do this, we take the derivative of the objective function with respect to $H$ and set it to zero:

$d/dH ||V - WH||_F^2 = -2(W^T)(V - WH) + 2(W^TWH)H = 0$

Solving for $H$, we get:

$H = W^T(V) / (W^TWH)$

We initialize $W$ and $H$ randomly and update them iteratively using the above equations until convergence.

NMF has applications in a variety of fields such as image processing, bioinformatics, and text mining. For example, in image processing, NMF can be used to decompose an image into a set of basis images and their corresponding coefficients, which can be used for image compression and feature extraction. In text mining, NMF can be used to identify latent topics in a collection of documents.
