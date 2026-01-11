# Physics, math and reservoir computing
Repository of some recent projects that I have done recently related to physics and reservoir computing.

As times go on I will add more and more projects that I have done over the years or that I have been doing recently related to these topics.


# Simulation of delayed stochastic differential equations

Two main scripts have been used: one for the simulation of networks of semiconductor lasers with optical feedback (delay) and delayed coupling, written in Julia; and other one to analyze and visualize time traces, with Python.

For the simulations I have used Runge-Kutta 4 method. Previous versions of this code used Heun method, which allowed for the implementation for multiplicative noise but at the cost of lower precision. In theory, the noise term in the rate equation for the electric field should be multiplicative (multiplied by the square root of the carrier number). Nevertheless, after the ignition of the laser, N~N_{th}, so additive noise multiplied by the proper coefficients and the square root of the carrier number at threshold. As a side note, carrier number has its own rate equation simulated. Nevertheless, because what we just said, sometimes not even the carrier number will be simulated and instead left as N = N_{th}. Given that is not possible to replicate exactly the evolution of the laser (we are dealing with a chaotic system and thus small perturbations grow exponentially over time) and we are just trying to end up in the same attractor, this also would be a good approximation.

There is another important thing to mention. This code uses a circular buffer for the electric with headers. When computing the next step of the simulation, we use RK4. The problem raises in k_2 and in k_3, where they ask for the value of E(t - τ + h/2), which has not been simulated given that h is our time step (that is, the smallest temporal resolution available). In theory one should use interpolation methods to estimate this value of the electric field. Nevertheless we haven't incorporated this because, again, we are satisfied with just arriving to the right attractor.



