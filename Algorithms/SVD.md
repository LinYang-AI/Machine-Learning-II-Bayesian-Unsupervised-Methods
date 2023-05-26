# Singular Value Decomposition

## Mathematical Prove

For a given matrix $A_{m\times n}$

1. Compute the singular values of $A$ by finding the eigenvalues of $A^TA$.

$A^T A$ is a symmetric matrix of size $n \times n$, so it can be diagonalized using its eigenvectors. Let $v_1, v_2, ..., v_n$ be the eigenvectors of $A^T A$, and let $λ_1, λ_2, ..., λ_n$ be their corresponding eigenvalues. The singular values of $A$ are the square roots of the eigenvalues of $A^T A$:

$σ_1 = \sqrt{λ_1}$, $σ_2 = \sqrt{λ_2}$, ..., $σ_n = \sqrt{λ_n}$

2. Compute the orthonormal matrix $V$ whose columns are the eigenvectors of $A^T A$.

The columns of $V$ are the eigenvectors of $A^T A$, normalized to unit length. That is, $v_1, v_2, ..., v_n$ are the columns of $V$, and each column $v_i$ satisfies:

$||v_i|| = 1$

and

$V^T V = I$

where $I$ is the identity matrix of size $n \times n$.

3. Compute the orthonormal matrix $U$ whose columns are the eigenvectors of $A A^T$.

The columns of $U$ are the eigenvectors of $A A^T$, normalized to unit length. That is, $u_1, u_2, ..., u_m$ are the columns of $U$, and each column $u_i$ satisfies:

$||u_i|| = 1$

and

$U^T U = I$

where $I$ is the identity matrix of size $m \times m$.

4. Compute the matrix $Σ$ whose diagonal elements are the singular values of $A$, and all other elements are zero.

The matrix $Σ$ is an $m \times n$ diagonal matrix, where the diagonal elements are the singular values of $A$:

$Σ = \begin{bmatrix}σ_{1} & & &\\ & σ_{2} & &\\ & & \ddots &\\ & & & σ_{n}\end{bmatrix}$

where $\Sigma$ represents a diagonal matrix with diagonal elements $σ_1, σ_2, ..., σ_n$ and all other elements equal to zero.

5. Verify that $A = U Σ V^T$.

Finally, we can verify that $A$ can be expressed as the product of the three matrices $U$, $Σ$, and $V^T$:

$A = U Σ V^T$

This is the singular value decomposition of $A$.

Note that the singular values in $Σ$ are arranged in decreasing order, so that $σ_1 \geq σ_2 \geq ... \geq σ_n$. The first $k$ singular values and their corresponding left and right singular vectors can be used for truncated SVD, as described in a previous answer.
