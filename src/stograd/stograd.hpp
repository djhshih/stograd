#ifndef _STOGRAD_HPP_
#define _STOGRAD_HPP_

#include <vector>
#include <cmath>

#include <iostream>

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
				/// To minimize the objective function, use the `substract_from` utility
				/// function to update the internal parameter vectorc with the provided
				/// delta vector. Otherwise, use the `add_to` function.
				///
				/// @param delta  vector for updating the parameter vector
				virtual void update(const std::vector<double>& delta) = 0;

			protected:

				// Disallow polymorphic deletion through a base pointer
				virtual ~Optimizable() {}
		};

#endif  // DEBUG


	/**
	 * Substract vector xs from vector ys.
	 *
	 * Vectors must have the same size.
	 */
	template <typename T>
	void subtract_from(const vector<T>& xs, vector<T>& ys) {
		typename vector<T>::const_iterator xit, xend = xs.end();
		typename vector<T>::iterator yit;
		for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
			(*yit) -= (*xit);
		}
	}

	/**
	 * Add vector xs to vector ys.
	 *
	 * Vectors must have the same size.
	 */
	template <typename T>
	void add_to(const vector<T>& xs, vector<T>& ys) {
		typename vector<T>::const_iterator xit, xend = xs.end();
		typename vector<T>::iterator yit;
		for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
			(*yit) += (*xit);
		}
	}

	/**
	 * Dot product of xs and ys.
	 *
	 * Vectors must have the same size.
	 */
	template <typename T>
	T dot_product(const vector<T>& xs, const vector<T>& ys) {
		typename vector<T>::const_iterator xit, yit, xend = xs.end();
		T p = 0;
		for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
			p += (*xit) * (*yit);
		}
		return p;
	}

	/**
	 * Root of sum of squares.
	 */
	template <typename T>
	double rss(const vector<T>& xs) {
		typename vector<T>::const_iterator xit, xend = xs.end();
		double r = 0.0;
		for (xit = xs.begin(); xit != xend; ++xit) {
			double x = *xit;
			r += x*x;
		}
		return sqrt(r);
	}

	/**
	 * Sign of nuemric value.
	 */
	template <typename T> int sign(T t) {
		return (T(0) < t) - (t < T(0));
	}

	namespace stepper {

		/// Non-adaptive
		template <typename Real>
		struct constant {
			// learning rate
			Real r;

			constant(Real rate=0.01)
				: r(rate) {}

			Real operator()(Real g) {
				return r * g;
			}
		};

		/// Momemtum
		template <typename Real>
		struct momentum {
			// base learning rate
			Real r;

			// hyperparameter
			Real b;

			// first moment
			Real v;

			momentum(Real rate=0.001, Real beta=0.9)
				: r(rate), b(beta), v(0.0) {}

			Real operator()(Real g) {
				// update moment
				v = b * v + g;

				return r * v;
			}
		};

		/// RMSprop
		/// equivalent to ADAM with beta2=0 and no bias correction
		template <typename Real>
		struct rmsprop {
			// base learning rate
			Real r;

			// hyperparameters
			Real b, e;

			// second moment
			Real v;

			rmsprop(Real rate=0.001, Real beta=0.9, Real epsilon=1e-6)
				: r(rate), b(beta), e(epsilon), v(0.0) {}

			Real operator()(Real g) {
				// update moment
				v = b*v + (1 - b)*g*g;
				
				// NB epsilon is intentionally placed outside sqrt
				return r * g / (sqrt(v) + e);
			}
		};

		/// ADAM
		template <typename Real>
		struct adam {
			// base learning rate
			Real r;

			// hyperparameters
			Real b1, b2, e;

			// bias correct the moments
			// (this can cause v to blow up too quick, causing premature stopping)
			bool debias;

			// first and second moments
			Real m, v;

			// timestep
			unsigned long int t;

			adam(Real rate=0.001, Real beta1=0.9, Real beta2=0.999, Real epsilon=1e-3)
				: r(rate), b1(beta1), b2(beta2), e(epsilon), debias(false), m(0.0), v(0.0), t(0) {}
			
			Real operator()(Real g) {
				++t;

				// update moments
				m -= (1 - b1) * (m - g);
				v -= (1 - b2) * (v - g*g);
				// equivalently,
				// m = b1*m + (1 - b1)*g;
				// v = b2*v + (1 - b2)*g*g;

				if (debias) {
					// correct for bias in moments
					m = m / (1 - pow(b1, t));
					v = v / (1 - pow(b2, t));
				}

				return r * m / (sqrt(v) + e);
			}
		};

		/// YOGI
		template <typename Real>
		struct yogi {
			// base learning rate
			Real r;

			// hyperparameters
			Real b1, b2, e;

			// bias correct the moments
			// (this can cause v to blow up too quick, causing premature stopping)
			bool debias;

			// first and second moments
			Real m, v;

			// timestep
			unsigned long int t;

			yogi(Real rate=0.01, Real beta1=0.9, Real beta2=0.999, Real epsilon=1e-3)
				: r(rate), b1(beta1), b2(beta2), e(epsilon), debias(false), m(0.0), v(0.0), t(0) {}
			
			Real operator()(Real g) {
				++t;

				// update moments
				m -= (1 - b1) * (m - g);
				v -= (1 - b2) * sign(v - g*g) * g*g;

				if (debias) {
					// correct for bias in moments
					m = m / (1 - pow(b1, t));
					v = v / (1 - pow(b2, t));
				}

				return r * m / (sqrt(v) + e);
			}
		};

	}  // namespace stepper
	
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
	 * @param stepf   functor object for adapting the gradient into a step
	 *                (e.g. stepper::constant, stepper::adam)
	 * @param bsize   batch size
	 * @param nepochs number of passes through the data
	 * @param eps     threshold on gradient norm for early convergence
	 * @return  number of passes through the data,
	 *          negated if gradient did not converge to zero early
	 */
	template <typename Optimizable, typename F, typename Real>
	int optimize(Optimizable& op, F& stepf, size_t bsize, size_t nepochs, Real eps=1e-4) {
		size_t nobs = op.nobs();
		size_t nparams = op.nparams();

		// number of batches
		size_t nbatches = nobs / bsize;

		// number of leftover data points
		size_t nremnant = nobs - (nbatches * bsize);

		// early convergence of gradient to zero
		bool converged = false;

		// initialize a stepper for each parameter
		vector<F> stepfs(nparams, stepf);

		size_t a;
		for (a = 0; a < nepochs; ++a) {

			// overall gradient across all batches
			vector<Real> odelta(nparams, 0.0);

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
				}

				// call the steppers with the normalized gradient to compute the steps
				// reusing grad vector for storing the delta values
				typename vector<Real>::iterator it;
				typename vector<Real>::const_iterator end = grad.end();
				typename vector<F>::iterator sit;
				for (it = grad.begin(), sit = stepfs.begin(); it != end; ++it, ++sit) {
					// normalize the gradient by the batch size
					double g = (*it) / _bsize;
					(*it) = (*sit)(g);
				}
				vector<Real>& delta = grad;

				// update the parameter vector
				op.update(delta);

				// accumulate the overall delta vector
				add_to(delta, odelta);

			} // nbatches
			
			if (rss(odelta) < eps) {
				converged = true;
				break;
			}

		} // nepochs

		return converged ? a : -a;
	}

	/**
	 * Approximate gradient by central finite difference.
	 *
	 *
	 * @param fn  functor object that accepts vector<Real>& and returns Real;
	 *            evaluation fn(x) must have no side-effect
	 * @param x   point at which to evaluate the gradient
	 * @param g   uninitialized out parameter that will hold the evaluated value
	 * @param step  step size
	 */
	template <typename F, typename Real>
	void finite_difference_gradient(const F& fn, const vector<Real>& x, vector<Real>& g, Real step=1e-6) {
		size_t D = x.size();
		g.reserve(D);
		for (size_t d = 0; d < D; ++d) {
			vector<Real> xp(x);
			xp[d] += step;

			vector<Real> xm(x);
			xm[d] -= step;

			g.push_back( (fn(xp) - fn(xm)) / (2 * step) );
		}
	}

}

#endif  // _STOGRAD_HPP_
