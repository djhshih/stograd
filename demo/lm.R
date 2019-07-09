# Demo application applying stochastic gradient descent
# to solve a linear model.

source("../R/stograd.R");

# Gradient for linear models
# @beta  beta   model parameter values (i.e. coefficients)
# @beta  data   a list of y and X
gradient_lm <- function(beta, data) {
	d <- unsplit_data(data);
	y <- d[[1]];
	X <- d[[2]];

	t( -2 * t(y - X %*% beta) %*% X )
}

# Squared error for linear models
objective_lm <- function(beta, data) {
	d <- unsplit_data(data);
	y <- d[[1]];
	X <- d[[2]];

	e <- y - X %*% beta;
	sum(e * e)
}

# Estimate linear model betaeters based on normal equation:
# \hat{\theta} = (X^\top X)^{-1} X^\top y
estimate_by_normal_eq <- function(y, X) {
	solve(t(X) %*% X, t(X) %*% y)
}

# Convert matrix to a list of rows
mat_to_rlist <- function(mat) {
	lapply(1:nrow(mat), function(i) mat[i,])
}

# Root mean squared error
rmse <- function(x, y) {
	sqrt(mean((x - y)^2))
}

# Split regression data into a list of observations
# y  response variable (univariate)
# X  predictor variables
split_data <- function(y, X) {
	mapply(
		function(y, X) list(y=y, X=X),
		y,
		mat_to_rlist(X),
		SIMPLIFY=FALSE
	)
}

# Reverse the split operation on regression data
unsplit_data <- function(data) {
	y <- matrix(unlist(lapply(data, function(d) d$y)), ncol=1);
	X <- matrix(unlist(lapply(data, function(d) d$X)), byrow=TRUE, nrow=length(y));
	list(y, X)
}


# Example 1 ------------------------------------------------------------------

N <- 5;
d <- 2;
X <- matrix(c(-2, -1, 2, 0, 1, 3, 0, -1, 1, 2), nrow=N, ncol=d, byrow=TRUE);
beta <- matrix(c(0.5, -1.5), ncol=1);
y <- X %*% beta;

data <- split_data(y, X);

beta0 <- matrix(rep(0, d), ncol=1);
beta.hat.sgd <- stograd(beta0, data, fn=NULL, gr=gradient_lm);
beta.hat.sgd2 <- stograd(beta0, data, fn=objective_lm);

beta.hat <- estimate_by_normal_eq(y, X);


# Example 2 ------------------------------------------------------------------

N <- 1010;
d <- 5;

sigma <- 1;

X <- matrix(rnorm(N*d), nrow=N, ncol=d);
beta <- matrix(rnorm(d), ncol=1);
error <- matrix(rnorm(N, sd=sigma), ncol=1);
y <- X %*% beta + error;

data <- split_data(y, X);

beta0 <- matrix(rep(0, d), ncol=1);
beta.hat.sgd <- stograd(beta0, data, fn=NULL, gr=gradient_lm);
beta.hat.sgd2 <- stograd(beta0, data, fn=objective_lm);

objective_lm(beta.hat.sgd, data)

beta.hat <- estimate_by_normal_eq(y, X);
beta.hat2 <- estimate_by_normal_eq(y[1:100, , drop=FALSE], X[1:100, ]);

rmse(beta.hat.sgd, beta)
rmse(beta.hat.sgd2, beta)

rmse(beta.hat, beta)
rmse(beta.hat2, beta)

gradient_lm(beta, data)
gradient_lm(beta.hat, data)
gradient_lm(beta.hat.sgd, data)

fdiff_lm <- finite_difference_gr(objective_lm, data);
fdiff_lm(beta, data)
fdiff_lm(beta.hat, data)
fdiff_lm(beta.hat.sgd, data)

