#include <stograd.hpp>
#include <iostream>

using namespace std;
using namespace stograd;

namespace lm {

	struct model {
		/// Rows of predictor variable values
		vector<vector<double>> X;

		/// Observed response variable values
		vector<double> y;

		/// Index of the current observation
		size_t i;

		model() : i(0) {}
		
		/// Return squared error
		/// (y - X \beta)^\top (y - X \beta)
		/// This is not required if exact analytic gradient is available
		double objective(const vector<double>& beta) {
			double e = y[i] - dot_product(X[i], beta);
			return e * e;
		}

		/// Return gradient w.r.t. beta in vector g
		/// g is a out parameter that must be passed in uninitialized
		/// ( -2 (y - X \beta)^\top X )^\top
		void gradient(const vector<double>& beta, vector<double>& g) {
			vector<double>& xi = X[i];
			size_t D = xi.size();
			g.reserve(D);

			double e = (y[i] - dot_product(xi, beta));
			for (size_t d = 0; d < D; ++d) {
				g.push_back(-2.0 * e * xi[d]);
			}
		}

		/// Move onto next observation
		void next() {
			++i;
			if (i == y.size()) {
				i = 0;
			}
		}

		double operator()(const vector<double>& beta) {
			return objective(beta);
		}
	};

	struct optimizable {

		size_t _N;
		size_t _D;

		model m;
		vector<double> beta;

		optimizable(size_t N, size_t D)
			: _N(N), _D(D), beta(D, 0.0)
		{}
		
		size_t nobs() const {
			return _N;
		}

		size_t nparams() const {
			return _D;
		}

		/// Compute gradient based on observation i and
		/// accmuluate the current gradient
		void accumulate(vector<double>& grad) {
			vector<double> gradi;
			m.gradient(beta, gradi);

			add_to(gradi, grad);
			
			m.next();
		}

		/// Update beta based on provided delta vector
		void update(const vector<double>& delta) {
			subtract_from(delta, beta);	
		}

	};

}

int main(int argc, char* argv[]) {

	const size_t N = 5;
	const size_t D = 2;

	// ground truth beta
	vector<double> beta(D);
	beta[0] = 0.5;
	beta[1] = -1.5;

	cout << "beta: [" << beta[0] << ", " << beta[1] << "]" << endl;

	// set up model
	lm::optimizable opt(N, D);
	lm::model& m = opt.m;

	cout << "Populate example data ..." << endl;

	m.X.resize(N);

	m.X[0].resize(D);
	m.X[0][0] = -2.0;
	m.X[0][1] = -1.0;

	m.X[1].resize(D);
	m.X[1][0] =  2.0;
	m.X[1][1] =  0.0;

	m.X[2].resize(D);
	m.X[2][0] =  1.0;
	m.X[2][1] =  3.0;

	m.X[3].resize(D);
	m.X[3][0] =  0.0;
	m.X[3][1] = -1.0;
	
	m.X[4].resize(D);
	m.X[4][0] =  1.0;
	m.X[4][1] =  2.0;

	// y = X \beta
	m.y.resize(N);
	for (size_t i = 0; i < N; ++i) {
		m.y[i] = dot_product(m.X[i], beta);
	}

	cout << "Running stochastic gradient descent ..." << endl;

	// Estimate beta by using stochastic gradient descent
	// to minimize the squared error objective function
	
	double epochs = stograd::optimize(opt, 2, 1000, 1e-2, 1e-4);

	cout << "elasped epochs: " << epochs << endl;
	cout << "beta_hat: [" << opt.beta[0] << ", " << opt.beta[1] << "]" << endl;

	return 0;
}
