# MacroFrontiersProblemSet
Code for the problem set of professor Villaverde's class

The answers to all questions are written in Julia. The most recent version is recommended to avoid syntax incompatibilities.
The following packages should be added before compiling : 
Optim , NLopt, Random, Distributions, NLsolve, LineSearches

_Q23.jl 

The file Q23.jl runs the code for the first two questions (comparing the different integration and optimization methods). 
Simply open and run the file after checking that the packages used at the start are added to your environment.
The printed lines in the console give the computation results for the integration methods and compare the values and convergence of the different optimization methods. Optimization in question 3 is done with the Julia Optim package as it features all the methods that are asked. 
We compare the efficiency of the methods by comparing the number of iterations by the different methods. Note that the steepest gradient descent does not converge after 1000 iterations, which is the default limit in Optim (it takes a few hundreds more). The number of iterations and convergence status is printed for each method. Conjugate gradient seems to be the most efficient in that regard, although it requires an additional linesearch.

_ParetoAllocation.jl 

The first lines of the code allow to modify the parameters. m and n as are in the question. The code is flexible and can accomodate any length of m and n as long as the Pareto weights, utility function parameters and endowments have the right size given m and n. 

After setting the desired parameters, run the full file. The printed text shows the number of iterations until convergence and the result. Given that we use an efficient method implemented in NLopt, m=10 and n=10 run reasonably well. Tweaking lambdas obviously gives a larger allocation to the individual with the largest lambda. Increasing the elasticity or increasing alpha with respect to one good tends to make that good distributed more evenly as it contributes more to utility. (A difference in the allocation of that good is "harder" to compensate with other goods).

_CompetitiveAllocation.jl 

Here we wrap our optimization within a function excessdemand, which takes price vector as input and computes the L2 norm of excess demand by solving a nonlinear system using the marginal rates of substitution. We use a gradient-free optimizer from NLopt ( LN_COBYLA ). The printed text shows the resulting allocation, price vector and the number of iterations until convergence. Again m,n and all parameters can be changed in the first lines of the code, it will run for any combination such that the vector of endowments, alpha and elasticities have the right size given m and n.
Simply run the file after setting the parameters of your choice or leaving the defaults. Output will show in the REPL.
