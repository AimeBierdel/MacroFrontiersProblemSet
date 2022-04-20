using NLopt
using NLsolve
using Random, Distributions, LineSearches

Random.seed!(15)



######## We will bruteforce the problem by solving simultaneously for the prices and allocation (probably not recommended)


Random.seed!(15)

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


du(i,j,x) = alpha[j]* x^w[i,j]
ddu(i,j,x) = alpha[j]*w[i,j]* x^(w[i,j]-1)
### the system 





function xdemand(p0::Vector, grad::Vector)
    if length(grad)>0

    end
    p = vcat(1, p0)
    x0 = [ 1 + 0.1*i for i in 1:(n*m) ]
    b = [ p[1]*e[i,1] + p[2] * e[i,2] + p[3] * e[i,3]  for i = 1:3 ]
    function f!(F,x)
        for i in 1:n 
            for j in 1:m-1 
                F[(i-1)*m+j] = du(i,j,exp(x[(i-1)*m+j])) / du(i,j+1,exp(x[(i-1)*m+j+1])) - p[j]/p[i] 
            end 
            F[ i*m] = b[i] - exp(x[(i-1)*m+1]) - sum( [ exp(x[(i-1)*m+j])*p[j] for j in 2:m ])
        end 
    end 
    res = nlsolve(f!, x0)
    x = res.zero

    demand = [ sum([exp(x[(i-1)*m+j])  for i in 1:n]) for j in 1:m ] 
    xdemand = sqrt( sum( [ (demand[j] - sum(e[:,j]))^2 for j in 1:m ]  ) )
    return xdemand 
end 


println( " Computing the competitive allocation with endowments", e )
println(" And individual weights w = ", w , " and alphas ", alpha )

### We use a gradient-free solver with box constraint 

opt = NLopt.Opt(:LN_COBYLA, 2)
opt.lower_bounds = [0.,0.]
opt.upper_bounds = [50.,50.]

# Tolerance 
opt.xtol_rel = 1e-4

# objective and constraints 
opt.min_objective = xdemand

# Stop after 200 evaluations 
maxeval!(opt,200);
p0 = [2.,2.]
# Optimization 

(minf,minx,ret) = NLopt.optimize(opt, p0 )
numevals = opt.numevals # the number of function evaluations
p = vcat(1, minx)
println("We find prices equal to $p" ) 
println( "after $numevals iterations using the optimization method LN_COBYLA from NLopt(returned $ret)")

x0 = [ 1 + 0.1*i for i in 1:(n*m) ]

b = [ p[1]*e[i,1] + p[2] * e[i,2] + p[3] * e[i,3]  for i = 1:3 ]
function f!(F,x)
    for i in 1:n 
        for j in 1:m-1 
            F[(i-1)*m+j] = du(i,j,exp(x[(i-1)*m+j])) / du(i,j+1,exp(x[(i-1)*m+j+1])) - p[j]/p[i] 
        end 
        F[ i*m] = b[i] - exp(x[(i-1)*m+1]) - sum( [ exp(x[(i-1)*m+j])*p[j] for j in 2:m ])
    end 
end 
res = nlsolve(f!, x0)
x = exp.(res.zero)

println("The corresponding allocation is $x" )
println(" Where the first 3 elements represent the allocation of the three goods for the first agent, etc.")



### Jacobian of the system to speed it up
#function j!(J, x)
#    for i in 1:n 
#        for j in 1:m-1     
#            J[(i-1)*m+j, (i-1)*m+j] = ddu(i,j,exp(x[(i-1)*m+j])) / du(i,j+1,exp(x[(i-1)*m+j+1])) 
#            J[(i-1)*m+j, (i-1)*m+j+1] = -du(i,j,exp(x[(i-1)*m+j]))* ( ddu(i,j+1,exp(x[(i-1)*m+j+1])) / du(i,j+1,exp(x[(i-1)*m+j+1]))^2 ) 
#        end 
#        J[i*m,(i-1)*m+1 ] = -1
#       for j in 2:m 
#            J[i*m,(i-1)*m+j ] = -p[j]
#        end
#    end
#end
