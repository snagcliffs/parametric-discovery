# parametric-discovery

Code for discovery of parametric partial differential equations presented in;

Rusy, S., Alla, A., Brunton, B., and Kutz, J.N.

"Data-driven identification of parametric partial differential equations."


Abstract:
In this work we present a data-driven method for the discovery of parametric partial differential
equations (PDEs), thus allowing one to disambiguate between the underlying evolution equations
and their parametric dependencies. Group sparsity is used to ensure parsimonious representations
of observed dynamics in the form of a parametric PDE, while also allowing the coefficients to have
arbitrary time series, or spatial dependence. This work builds on previous methods for the identifica-
tion of constant coefficient PDEs, expanding the field to include a new class of equations which until
now have eluded machine learning based identification methods. We show that group sequentially
thresholded ridge regression outperforms group LASSO in identifying the fewest terms in the PDE
along with their parametric dependency. The method is demonstrated on four canonical models with
and without the introduction of noise.