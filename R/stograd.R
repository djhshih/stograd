#' Stochastic gradient descent
#'
#' Optimize an objective function of parameters and data with respect to
#' the parameters using stochastic (mini-batch) gradient descent.
#'
#' @param param  initial guess of values for the function parameters
#' @param data   a list of data points for input into fn
#' @param fn     objective function (unnessary if \code{gr} is specified)
#' @param gr     gradient of the objective function
#' @param rate   learning rate
#' @param control  a list of control parameters:
#'                 \itemize{
#'                   \item \code{nbatches}, number of batches [10];
#'                   \item \code{bsize}, batch size;
#'                   \item \code{nepochs}, number of passes through the data;
#'                   \item \code{rate}, learning rate [0.01];
#'                   \item \code{eps}, threshold for gradient norm for early exit
#'                 }
#' @return estimate of parameter values that minimize the objective function
#' @export
#'
stograd <- function(param, data, fn, gr=NULL, control = list()) {
	N <- length(data);
	d <- length(param);

	if (is.null(control$bsize)) {
		if (is.null(control$nbatches)) {
			nbatches <- min(N, 10);
		}
		bsize <- floor(N / nbatches);
	} else {
		nbatches <- floor(N / bsize);
	}

	if (is.null(control$nepochs)) {
		nepochs <- 100;
	}

	if (is.null(control$rate)) {
		rate <- 1e-2;
	}

	if (is.null(control$eps)) {
		eps <- 1e-4;
	}

	if (is.null(gr)) {
		gr <- finite_difference_gr(fn, data);
	}

	for (i in 1:nepochs) {
		odelta <- rep(0, d);
		for (b in 1:nbatches) {
			# extract batch
			blim <- batch_limits(b, bsize, nbatches);
			batch <- data[blim[1]:blim[2]];
			# actual batch size
			B <- blim[2] - blim[1] + 1;

			# update the parameters
			delta <- rate / B * gr(param, batch);
			param <- param - delta;

			odelta <- odelta + delta;
		}

		# check for early convergence
		if (sqrt(sum(odelta*odelta)) < eps) {
			message("INFO: Early convergence at epoch ", i)
			break;
		}
	}

	param
}

# Return batch start and end indices given.
# b         batch index
# bsize     batch size
# nbatches  number of batches
batch_limits <- function(b, bsize, nbatches) {
	# extract a batch of data
	s <- (b - 1) * bsize + 1;
	e <- b * bsize;
	# group ultimate batch with the penultimate batch
	if (b == nbatches) {
		e <- N;
	}

	c(s, e)
}

# Central finite difference approximation the gradient.
# Requires 2 * d evaluations of the function
# where d is the number of parameter dimensions.
# step << 1e-6 appears to reduce accuracy.
finite_difference_gr <- function(fn, data, step=1e-6) {
	function(param, data) {
		d <- length(param);
		# perturb each of d dimensions of the parameter
		unlist(lapply(1:d,
			function(k) {
				# perturb only the kth dimension
				v <- rep(0, d);
				v[k] <- step;

				# evaluate central difference
				(fn(param + v, data) - fn(param - v, data)) / (2*step)
			}
		))
	}
}

