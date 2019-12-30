# stograd

A C++ header-only library for stochastic gradient descent.

This library provides a common interface for stochastic gradient descent for
many learning rate adaption methods:

- Momemtum
- RMSprop
- AdaDelta
- ADAM
- AdaMax
- YamAdam
- AMSGrad
- YOGI

For an example on how to implement a statistical model (comprising of an
objective function and optionally a gradient function) that will be fitted
to the data using stochastic gradient descent, see `demo/lm.cpp`.

# Dependency

- gcc >= 4.8
