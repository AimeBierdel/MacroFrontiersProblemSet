using NLopt
using JuMP
using Optim
using Random, Distributions, LineSearches

Random.seed!(15)
####### Integration ##########

rho = 0.04
T = 100 
lambda = 0.02   
f(t) = -exp(-rho*t) * exp( exp(- lambda*t) - 1)

## midpoint method ##

n = 1000 # 1000 points to subdivide the interval into
w = T/n

vecMidPoint = [ f( (t- 0.5)*w ) for t in 1:n]
intMidPoint = w* sum(vecMidPoint)

println( " The integral computed with the midpoint method is equal to $intMidPoint")

## Trapezoid ##

vecTrapPoint = [ f(t * w ) for t in 1:n    ]
intTrapPoint = w*( f(0)/2  + f(T)/2 + sum(vecTrapPoint ))

println( " The integral computed with the trapezoid method is equal to $intTrapPoint")

## Simpson ##


vecSimPointEven = [ 2*f(2*t * w ) for t in 1:n/2    ]
vecSimPointOdd = [ 4*f((2*t-1) * w ) for t in 1:(n/2-1)    ]

intSimPoint = w/3*( f(0)  + f(T) + sum(vecSimPointEven )+ sum(vecSimPointOdd))

println( " The integral computed with the Simpson method is equal to $intSimPoint")

## Monte Carlo 

# Draw 1000 points between 0 and T
ndraws = 10000
draws = rand(Uniform(0,T),ndraws)

MCvec = [ f(t) for t in draws]
MCint  = T*sum(MCvec)/ndraws
 
println( " The integral computed with the Monte Carlo method is equal to $MCint")

# Then we take the 

######## Optimization #########

g(x) = 100.0*(x[2]-x[1]^2)^2 +(1.0-x[1])^2

function dg!(stor,x)
    stor[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    stor[2] = 200.0 * (x[2] - x[1]^2)
end 

function ddg!(stor,x)
    stor[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    stor[1, 2] = -400.0 * x[1]
    stor[2, 1] = -400.0 * x[1]
    stor[2, 2] = 200.0
end 

## Newton Raphson 
println("Optimizing with Newton Raphson")
resNewton = Optim.optimize(g,dg!, [1.0 , 18.0], Newton())
print(resNewton)
println( "We find the minimum at", Optim.minimizer(resNewton))


### BFGS 
println("Optimizing with BFGS")
resBFGS = Optim.optimize(g, dg!, [0.0, 0.0], LBFGS())
print(resBFGS)
println( "We find the minimum at", Optim.minimizer(resBFGS))

### steepest descent

println("Optimizing with Steepest Gradient Descent")
resSteepGD = Optim.optimize(g, dg!, [0.0, 0.0],GradientDescent(; alphaguess = LineSearches.InitialPrevious(),
linesearch = LineSearches.HagerZhang(),
P = nothing,
precondprep = (P, x) -> nothing)  )
print(resSteepGD)
println( "We stop at", Optim.minimizer(resSteepGD), "after 1000 iterations")
println(" The steepest gradient descent does not converge as fast as the other methods tested")

### Conjugate gradient descent 

println("Optimizing with Conjugate Gradient Descent with eta = 0.1")
resConjugateGD = Optim.optimize(g, dg!, [0.0, 0.0],ConjugateGradient(; alphaguess = LineSearches.InitialHagerZhang(),
linesearch = LineSearches.HagerZhang(),
eta = 0.1,
P = nothing,
precondprep = (P, x) -> nothing)  )
print(resConjugateGD)
println( "We find the minimum at ", Optim.minimizer(resConjugateGD) )
