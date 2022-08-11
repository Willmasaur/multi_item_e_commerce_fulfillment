using JuMP
using Gurobi
using Distances
using Random
using StatsBase
EPSILON = 0.0000001
cd(dirname(@__FILE__))
# rng = MersenneTwister(1234);

INSTS = 30 # number of random instances (order types, demand rates) to generate
SEQS_PER_INST = 30 # number of random arrival sequences to generate per instance
A = 5 # number of algorithms to test
z_safety = 0.5

REGIONS_FILENAME = "cities_warmup.csv"
FCS_FILENAME = "fulfillment_centers_warmup.csv"
I = 20 # number of items
nmax = 5 # maximum size of order
nper = 5 # number of orders of each size
T = 10000
p_carry = 0.75

# UNCOMMENT LINES BELOW TO SIMULATE BIGGER NETWORK
# REGIONS_FILENAME = "cities.csv"
# FCS_FILENAME = "fulfillment_centers.csv"
# (I, nmax, nper, T) = (100, 10, 10, 100000)
# p_carry = 0.5
# p_carry = 0.25


# READ REGIONS
city_name = []
city_state = []
city_coor = []
city_pop = []
open(REGIONS_FILENAME,"r") do f
    for line in eachline(f)
        temp = split(line, ",")
        push!(city_name, temp[1])
        push!(city_state, temp[2])
        push!(city_coor, (parse(Float64,temp[4]), parse(Float64,temp[3]))) # the coordinates are LONGITUDE first!!
        push!(city_pop, parse(Int64,temp[5]))
    end
end
J = length(city_coor) # number of regions

# READ FULFILLMENT CENTERS
fc_name = []
fc_state = []
fc_coor = []
open(FCS_FILENAME,"r") do f
    for line in eachline(f)
        temp = split(line, ",")
        push!(fc_name, temp[1])
        push!(fc_state, temp[2])
        push!(fc_coor, (parse(Float64,temp[4]), parse(Float64,temp[3])))
    end
end
K = length(fc_coor) # number of fulfillment centers

# DISTANCES AND COSTS
c_fixed = zeros(K+1, J)
c_unit = zeros(K+1, J)
dists = zeros(K, J)
for j in 1:J
    dist_max = 0.0
    for k in 1:K
        c_fixed[k,j] = 8.759
        dists[k,j] = evaluate(Haversine(), city_coor[j], fc_coor[k])/1000/1.61 # this gives distance in miles
        c_unit[k,j] = 0.423 + 0.000541*dists[k,j]
        dist_max = max(dist_max, dists[k,j])
    end
    # routing to FC K+1 corresponds to not fulfilling an item
    c_fixed[K+1,j] = 2*8.759
    c_unit[K+1,j] = 2*(0.423 + 0.000541*dist_max)
end

function indep_round(x,q,j)
    n = size(x, 2)
    fc_tried = zeros(Int64, n)
    for i in 1:n
        weights = aweights(x[:,i])
        fc_tried[i] = sample(KVec, weights)
    end
    return fc_tried
end

function dilate_round(x,q,j)
    K = size(x, 1)
    n = size(x, 2)
    fc_tried = zeros(Int64, n)
    opening_times = randexp(K)
    for i in 1:n
        observed_opening = opening_times ./ x[:,i]
        fc_tried[i] = argmin(observed_opening)
    end
    return fc_tried
end

function forceopen_round(x,q,j)
    K = size(x, 1)
    n = size(x, 2)
    fc_tried = zeros(Int64, n)
    opening_times = randexp(K)
    for i in 1:n
        observed_opening = opening_times ./ x[:,i]

        m = argmax(x[:,i])
        if rand() < (1 - x[m,i]) / (1 - x[m,i] + x[m,i]*exp(1/x[m,i]) - exp(1))
            observed_opening[m] = 1/x[m,i]
        else
            observed_opening[m] = min(opening_times[m], 1)/x[m,i]
        end

        fc_tried[i] = argmin(observed_opening)
    end
    return fc_tried
end

function JS_round(u,q,j)
    # u = [0.6 0.0 0.4 0.0; 0.3 1.0 0.5 0.3; 0.1 0.0 0.1 0.7]
    K = size(u, 1)
    n = size(u, 2)
    fc_tried = zeros(Int64, n)

    # Step 1
    tilde_u = zeros(n,K,n)
    for k in 1:K
        v = copy(u[k,:])
        for m in n:-1:1
            r = 0
            min_nonzero = 1.0
            for i in 1:n
                if v[i] > EPSILON
                    r += 1
                    min_nonzero = min(min_nonzero, v[i])
                end
            end
            # r = count(i->(i>0), v)
            @assert r <= m v, m
            if r < m
                min_nonzero = 0.0
            end
            for i in 1:n
                if v[i] > EPSILON
                    tilde_u[m,k,i] = min_nonzero
                else
                    tilde_u[m,k,i] = 0.0
                end
            end
            v .-= tilde_u[m,k,:]
        end
    end

    # Step 2
    M = zeros(n,K)
    for m in 1:n
        for k in 1:K
            M[m,k] = maximum(tilde_u[m,k,:])
        end
    end
    L = zeros(K)
    for k in 1:K
        if k > 1
            L[k] = L[k-1] + sum(M[m,k]*m/n for m in 1:n)
        else
            L[k] = sum(M[m,k]*m/n for m in 1:n)
        end
    end
    H = zeros(K,n)
    Hlengths = zeros(K,n)
    for k in 1:K
        for m in 1:n
            Hlengths[k,m] = M[m,k]*m/n
            if m > 1
                H[k,m] = H[k,m-1] + Hlengths[k,m]
            else
                H[k,m] = (k==1 ? 0.0 : L[k-1]) + Hlengths[k,m]
            end
        end
    end

    # Step 3
    tilde_u_reduced = copy(tilde_u)
    for i in 1:n
        for k in 1:K
            for m in 1:n
                if tilde_u_reduced[m,k,i] > EPSILON
                    tilde_u_reduced[m,k,i] -= Hlengths[k,m]
                    @assert tilde_u_reduced[m,k,i] > -EPSILON tilde_u_reduced[m,k,i]
                end
            end
        end
    end

    Kn = fill((0,0), K, n)
    for k in 1:K
        for m in 1:n
            Kn[k,m] = (k,m)
        end
    end

    (k,m) = sample(Kn, aweights(Hlengths))
    for i in 1:n
        if tilde_u[m,k,i] > 0
            fc_tried[i] = k
        else
            (k2,m2) = sample(Kn, aweights(transpose(tilde_u_reduced[:,:,i])))
            fc_tried[i] = k2
        end
    end

    return fc_tried
end

KVec = collect(1:K+1)
overall_performances = zeros(A)
alg_runtimes = zeros(A)
for inst in 1:INSTS
    # ORDER TYPES
    query_types = [[]]
    q_containing_i = fill([],I)
    for i in 1:I
        q_containing_i[i] = [] # need to do this redundant step so that push! later works
    end
    for size in 1:nmax
        for index in 1:nper
            dummy = collect(1:I)
            shuffle!(dummy)
            push!(query_types, dummy[1:size])
            for i in 1:size
                push!(q_containing_i[dummy[i]],length(query_types))
            end
        end
    end
    Q = length(query_types)

    # DEMANDS
    query_prob = zeros(Q)
    size_zero_prob = rand()
    size_prob = rand!(zeros(nmax))
    size_prob_tot = size_zero_prob+sum(size_prob)
    query_prob[1] = size_zero_prob/size_prob_tot
    for size in 1:nmax
        subset_prob = rand!(zeros(nper))
        subset_prob_tot = sum(subset_prob)
        for index in 1:nper
            query_prob[1+(size-1)*nper+index] = size_prob[size]/size_prob_tot*subset_prob[index]/subset_prob_tot
        end
    end

    # SCALE BY POPULATIONS
    city_pop_tot = sum(city_pop)
    lambda = zeros(Q, J)
    QJ = fill((0,0), Q, J)
    for q in 1:Q
        for j in 1:J
            lambda[q,j] = query_prob[q]*city_pop[j]/city_pop_tot
            QJ[q,j] = (q,j)
        end
    end

    # STARTING INVENTORIES
    demands = zeros(K, I)
    b = zeros(Int64,K,I)
    closest_fcs = zeros(Int64, I, J)
    for i in 1:I
        stock_rand = rand!(zeros(K))

        # for each region j, find closest FC k (possibly none) that stocks item i
        for j in 1:J
            dist_min = typemax(Float64)
            closest = K+1
            for k in 1:K
                if dists[k,j] < dist_min && stock_rand[k] < p_carry
                    dist_min = dists[k,j]
                    closest = k
                end
            end
            closest_fcs[i, j] = closest
        end

        # go through queries containing i, route each region to correct FC
        for q in 1:Q
            if i in query_types[q]
                for j in 1:J
                    if closest_fcs[i, j] <= K
                        demands[closest_fcs[i, j],i] += lambda[q,j]
                    end
                end
            end
        end

        # use safety stock formula to decide how much to stock
        for k in 1:K
            b[k,i] = round(T*demands[k,i] + z_safety*sqrt(T*demands[k,i]*(1-demands[k,i])))
        end
    end

    # SOLVE LP ONCE
    model = Model(Gurobi.Optimizer)
    @variable(model, u[1:Q,1:K+1,1:I,1:J] >= 0)
    @variable(model, y[1:Q,1:K+1,1:J] >= 0)
    for k in 1:K # no inventory constraints for fc K+1
        for i in 1:I
            @constraint(model, T*sum(sum(lambda[q,j]*u[q,k,i,j] for q in q_containing_i[i]) for j in 1:J)  <= b[k,i])
        end
    end
    for q in 1:Q
        for j in 1:J
            for i in query_types[q]
                @constraint(model, sum(u[q,k,i,j] for k in 1:K+1) == 1)
                for k in 1:K+1
                    @constraint(model, y[q,k,j] >= u[q,k,i,j])
                end
            end
        end
    end
    @objective(model, Min, T*sum(sum(lambda[q,j]*sum(c_fixed[k,j]*y[q,k,j]+c_unit[k,j]*sum(u[q,k,i,j] for i in query_types[q]) for k in 1:K+1) for j in 1:J) for q in 1:Q))
    status = optimize!(model)
    DLP = objective_value(model)
    # println("", "Objective value is ", objective_value(model))
    u_opt = value.(u)
    XMatrices = Array{Matrix{Float64}}(undef, Q, J)
    for q in 1:Q
        for j in 1:J
            XMatrices[q,j] = u_opt[q,:,:,j][:,query_types[q]]
        end
    end

    function myopic(x,q,j)
        n = length(query_types[q])
        fc_tried = zeros(Int64,n)
        for i in 1:n
            fc_tried[i] = closest_fcs[query_types[q][i],j]
        end
        return fc_tried
    end

    function alg_run(rounding_function, arrivals)
        tot_cost = 0.0
        rem_inv = copy(b)
        for t in 1:length(arrivals)
            q = arrivals[t][1]
            j = arrivals[t][2]
            if q == 1 # order is empty
                continue
            end
            fc_tried = rounding_function(XMatrices[q,j],q,j)
            fc_used = zeros(Bool, K+1)
            for i in 1:length(fc_tried)
                k = fc_tried[i]
                if k <= K && rem_inv[k,query_types[q][i]] > 0
                    rem_inv[k,query_types[q][i]] -= 1
                else
                    k = K+1
                end
                fc_used[k] = 1
                tot_cost += c_unit[k,j]
            end
            for k in 1:K+1
                if fc_used[k]
                    tot_cost += c_fixed[k,j]
                end
            end
        end
        return tot_cost
    end

    ALGORITHMS = [myopic, indep_round, JS_round, dilate_round, forceopen_round]
    performances = zeros(A)
    for repeat in 1:SEQS_PER_INST
        arrivals = sample(QJ, aweights(lambda), T)
        for a in 1:A
            alg_runtimes[a] += @elapsed performances[a] += alg_run(ALGORITHMS[a], arrivals)
        end
    end
    performances ./= SEQS_PER_INST*DLP
    overall_performances .+= performances
end
overall_performances ./= INSTS
alg_runtimes ./= INSTS
println(overall_performances)
print(alg_runtimes)
