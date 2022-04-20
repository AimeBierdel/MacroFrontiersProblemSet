using NLopt
using JuMP
using Random, Distributions, LineSearches
Random.seed!(15)

######## Computing Pareto efficient allocations

m = 3
n = 3


### Vector of endowments 

e = zeros(n,m )*1. # line for agent, column for good 

e[1,:] = [1. , 2. , 3. ]
e[2,:] = [ 2., 2. , 2. ]
e[3,:] = [5., 0., 0. ]

### social weights and agent preferences

alpha = [1,1,1 ] ### Same weight on the three goods 
w = zeros( n,m)*1.
w[1,:] = [ -0.5,-0.5,-0.5] ## Same agent preferences
w[2,:] = [ -0.5,-0.5,-0.5]
w[3,:] = [ -0.5,-0.5,-0.5]
lambda = [1,1,1] # Same weight on each agent

println( " Computing the pareto optimum with endowments", e )
println( " Pareto weights ", lambda )
println(" And individual weights w = ", w , " and alphas ", alpha )

#### Utility function and planner Objective
u(i,j,x) =  alpha[j] * ( x^(1+w[i,j]) )/(1+w[i,j]) 

Upareto(x) = sum([ lambda[i]*u(i,j,x[m*(i-1)+j ])  for i in 1:n for j in 1:n]  )

#### derivative of utility and planner objective 

du(i,j,x) = alpha[j]* x^w[i,j]



 
low = [ 0.01 for i in 1:n for j in 1:m ]
high = [ sum(e[:,j]) for j in 1:m ]
#### Constrained optimization with NLOpt package 


function U(x::Vector, grad::Vector)
    if length(grad) > 0
        for i in 1:n 
            for j in 1:m
                grad[(i-1)*m+j] = lambda[i]*du(i,j,x[(i-1)*m+j])
            end 
        end 
    end
    return Upareto(x)
end

# budget constraint for each good j
function constraint(x::Vector, grad::Vector, j)
    if length(grad) > 0
        for i in 1:n 
            for l in 1:m
                if l == j 
                    grad[(i-1)*m+l] = 1.
                else 
                    grad[(i-1)*m+l] = 0.
                end 
            end 
        end  
    end
    sum( [ x[j + (i-1)*m ] for i in 1:n ] ) - high[j]
end

## Optimizer and bounds 




#opt = NLopt.Opt(:AUGLAG,9)
#opt.local_optimizer = NLopt.Opt(:LD_LBFGS, 9)
#opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 9)
#opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 9)
opt = Opt(:LD_MMA, 9)
opt.lower_bounds = low 

# Tolerance 
opt.xtol_rel = 1e-4

# objective and constraints 
opt.max_objective = U
for j in 1:m 
    inequality_constraint!(opt, (x,g) -> constraint(x,g,j), 1e-8)
end 
maxeval!(opt,100);
x0 = [ high[1]-0.1,high[2]-0.1, high[3]-0.1, 0.01,0.01,0.01,0.01,0.01,0.01  ]
# Optimization 

(minf,minx,ret) = NLopt.optimize(opt, x0 )
numevals = opt.numevals # the number of function evaluations
println("We obtain a welfare equal to $minf ") 
println("We find the optimal individual endowments  $minx " ) 
println( "after $numevals iterations using the LD_MMA optimization method from NLopt(returned $ret)")



