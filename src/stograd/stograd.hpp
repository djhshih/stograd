#ifndef _STOGRAD_HPP_
#define _STOGRAD_HPP_

#include <vector>
#include <cmath>

namespace stograd {

	using namespace std;

#ifdef DEBUG

		class Optimizable {
			public:

				/// Number of observations.
				virtual std::size_t nobs() const = 0;

				/// Number of parameters.
				virtual std::size_t nparams() const = 0;

				/// Accumulate gradient.
				///
				/// Compute the gradient of the objective function of the next data point
				/// and add it to the provided current gradient vector
				///
				/// @param grad  current gradient vector (of size `nparams`)
				virtual void accumulate(std::vector<double>& grad) = 0;

				/// Update parameter vector.
				///
				/// To minimize the objective function, use the `substract` utility
				/// function to update the internal parameter vectorc with the provided
				/// delta vector. Otherwise, use the `add` function.
				///
				/// @param delta  vector for updating the parameter vector
				virtual void update(const std::vector<double>& delta) = 0;

			protected:

				// Disallow polymorphic deletion through a base pointer
				virtual ~Optimizable() {}
		};

#endif  // DEBUG


	/**
	 * Substract vector ys from vector xs.
	 *
	 * Vectors must have the same size.
	 */
	template <typename T>
	void subtract(vector<T>& xs, const vector<T>& ys) {
		typename vector<T>::iterator xit;
		typename vector<T>::const_iterator yit, xend = xs.end();
		for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
			(*xit) -= (*yit);
		}
	}

	/**
	 * Add vector ys to vector xs.
	 *
	 * Vectors must have the same size.
	 */
	template <typename T>
	void add(vector<T>& xs, const vector<T>& ys) {
		typename vector<T>::iterator xit;
		typename vector<T>::const_iterator yit, xend = xs.end();
		for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
			(*xit) += (*yit);
		}
	}

	/**
	 * Root of sum of squares.
	 */
	template <typename T>
	double rss(vector<T>& xs) {
		typename vector<T>::iterator xit;
		typename vector<T>::const_iterator xend = xs.end();
		double r = 0.0;
		for (xit = xs.begin(); xit != xend; ++xit) {
			double x = *xit;
			r += x*x;
		}
		return sqrt(r);
	}
	

	/**
	 * Optimize an objective function.
	 *
	 * Minimize (maximize) an objective function by stochastic gradient descent
	 * (ascent) to a possibly local (but hopefully global) optimum.
	 *
	 * The Optimizable type is implicitly required to implement the Optimizable
	 * interface class given above. For production code, we use template instead 
	 * of an abstract class for superior runtime speed.
	 * 
	 * @param op      an object of a class that implements the implicit 
	 *                Optimizable interface
	 * @param bsize   batch size
	 * @param nepochs number of passes through the data
	 * @param rate    base learning rate
	 * @param eps     threshold on gradient norm for early convergence
	 * @return  number of passes through the data (including fractions),
	 *          negated if gradient did not converge to zero early
	 */
	template <typename Optimizable, typename Real>
	double optimize(Optimizable& op, size_t bsize, size_t nepochs, Real rate=1e-2, Real eps=1.0e-4) {
		size_t nobs = op.nobs();
		size_t nparams = op.nparams();

		// number of batches
		size_t nbatches = nobs / bsize;

		// number of leftover data points
		size_t nremnant = nobs - (nbatches * bsize);

		// TODO adapt the learning rate
		Real _rate = rate;

		// number of data points processed in the current epoch
		size_t nprocessed;

		// early convergence of gradient to zero
		bool converged = false;

		size_t a;
		for (a = 0; a < nepochs; ++a) {
			nprocessed = 0;

			// overall gradient across all batches
			vector<Real> ograd(nparams, 0.0);

			for (size_t b = 0; b < nbatches; ++b) {

				// last batch will include any leftover data points
				size_t _bsize;
				if (b == nbatches - 1) {
					_bsize = bsize + nremnant;	
				} else {
					_bsize = bsize;
				}

				vector<Real> grad(nparams, 0.0);

				// iterate through data points in a batch to accumulate the gradient
				for (size_t i = 0; i < _bsize; ++i) {
					op.accumulate(grad);
					++nprocessed;
				}

				// compute the delta vector, re-using grad
				typename vector<Real>::iterator it;
				typename vector<Real>::const_iterator end = grad.end();
				for (it = grad.begin(); it != end; ++it) {
					(*it) *= _rate / _bsize;
				}

				// update the parameter vector
				op.update(grad);

				// accumulate the overall gradient
				add(ograd, grad);

			} // nbatches
			
			if (rss(ograd) < eps) {
				converged = true;
				break;
			}

		} // nepochs

		double epochs = a + ((double)nprocessed / nobs);
		return converged ? epochs : -epochs;
	}


}

#endif  // _STOGRAD_HPP_
