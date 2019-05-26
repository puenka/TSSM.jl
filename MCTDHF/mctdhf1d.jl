using TSSM
using Combinatorics
using LinearAlgebra

COUNT_B = 0

function sign(p::Array{Int,1})
    n = length(p)
    k = 0    
    for i=1:n
        for j=i+1:n
            if p[i]>p[j]
                k += 1
            end
        end
    end
   (-1)^k
end


struct MultiFor
    k::Array{Int,1}
end

function Base.iterate(MF::MultiFor, k::Array{Int,1}=Int[]) 
    if k==MF.k
        return nothing
    end
    if k==Int[]
        k = ones(Int, length(MF.k))
        return(copy(k), k)
    end
    for i=1:length(k)
        if k[i]<MF.k[i]
            k[i] += 1
            for j = 1:i-1                 
                k[j] = 1       
            end
            return (copy(k), k)
        end
    end            
end


function find_xxx(v::Vector{Int}, lena::Int, slater_exchange, slater_indices; must_be_included::Int=0)
    res = Tuple{Int,Int,Float64}[]
    for i=1:lena
        for j=1:lena
            if must_be_included==0 || ((must_be_included in slater_indices[i]) || (must_be_included in slater_indices[i]))
                if v==slater_exchange[i,j][1]
                    push!(res, (i,j, slater_exchange[i,j][2]))
                end
            end
        end
    end
    res
end

function add_xxx!(d::Dict{Tuple{Int,Int}, Float64}, v::Tuple{Int,Int,Float64})
    key = (v[1],v[2])
    f = get!(d, key, 0.0) + v[3]
    if f==0
        delete!(d, key)
    else
        d[key] = f
    end
    d
end

function add_xxx!(d::Dict{Tuple{Int,Int}, Float64}, v::Vector{Tuple{Int,Int,Float64}})
    for x in v
        add_xxx!(d, x)
    end
    d
end

function init_mctdhf_combinatorics(f::Int, N::Int)
    slater_indices = collect(Combinatorics.Combinations(1:N, f))
    
    density_rules = [Tuple{Int,Int,Int}[] for j=1:N, l=1:N]
    for J = MultiFor([N for j=2:f])
        for j=1:N
            for l=j:N
                J1 = vcat(j,J)
                J2 = vcat(l,J)
                #j1 = findfirst(slater_indices, sort(J1))
                j1 = something(findfirst(isequal(sort(J1)), slater_indices), 0)
                if j1>0
                    #j2 = findfirst(slater_indices, sort(J2))
                    j2 = something(findfirst(isequal(sort(J2)), slater_indices), 0)
                    if j2>0
                        s1 = sign(J1)
                        s2 = sign(J2)
                        t = (j1, j2, s1*s2)
                        push!(density_rules[j,l], t)
                        if j!=l
                            push!(density_rules[l,j], t)
                        end
                    end
                end
            end
        end
    end    
    
    density2_rules = [Tuple{Int,Int,Int}[] for j=1:N, l=1:N, p=1:N, q=1:N]
    if f==2
        for j=1:N
            for l=j:N
                for p=1:N
                    for q=1:N
                        J1 = vcat(j, p)
                        J2 = vcat(l, q)
                        #j1 = findfirst(slater_indices, sort(J1))
                        j1 = something(findfirst(isequal(sort(J1)), slater_indices), 0)
                        if j1>0
                            #j2 = findfirst(slater_indices, sort(J2))
                            j2 = something(findfirst(isequal(sort(J2)), slater_indices), 0)
                            if j2>0
                                s1 = sign(J1)
                                s2 = sign(J2)
                                t = (j1, j2, s1*s2)
                                push!(density2_rules[j,l,p,q], t)
                                if j!=l
                                    push!(density2_rules[l,j,q,p], t)
                                end
                            end
                        end
                    end
                end
            end    
        end        
    else
    for J = MultiFor([N for j=3:f])
        for j=1:N
            for l=j:N
                for p=1:N
                    for q=1:N
                        J1 = vcat(j, p, J)
                        J2 = vcat(l, q, J)
                        #j1 = findfirst(slater_indices, sort(J1))
                        j1 = something(findfirst(isequal(sort(J1)), slater_indices), 0)
                        if j1>0
                            #j2 = findfirst(slater_indices, sort(J2))
                            j2 = something(findfirst(isequal(sort(J2)), slater_indices), 0)
                            if j2>0
                                s1 = sign(J1)
                                s2 = sign(J2)
                                t = (j1, j2, s1*s2)
                                push!(density2_rules[j,l,p,q], t)
                                if j!=l
                                    push!(density2_rules[l,j,q,p], t)
                                end
                            end
                        end
                    end
                end
            end    
        end
    end
    end
    
    lena = binomial(N,f)
    slater_exchange = [(Int[], 1) for j=1:lena, l=1:lena]
    # entry = (exchange_index_pairs, sign)
    for j=1:lena
        for l=1:lena
            u = symdiff(slater_indices[j], slater_indices[l])
            if length(u)==2
                #i1 = findfirst(slater_indices[j], u[1])
                i1 = something(findfirst(isequal(u[1]), slater_indices[j]), 0)
                #i2 = findfirst(slater_indices[l], u[2])
                i2 = something(findfirst(isequal(u[2]), slater_indices[l]), 0)
                s = (-1)^(i1+i2)
                slater_exchange[j,l] = (u, s)
            elseif length(u)==4
                u = [u[1], u[3], u[2], u[4]]
                #i1 = findfirst(slater_indices[j], u[1])
                i1 = something(findfirst(isequal(u[1]), slater_indices[j]), 0)
                #i2 = findfirst(slater_indices[l], u[2])
                i2 = something(findfirst(isequal(u[2]), slater_indices[l]), 0)
                #i3 = findfirst(slater_indices[j], u[3])
                i3 = something(findfirst(isequal(u[3]), slater_indices[j]), 0)
                #i4 = findfirst(slater_indices[l], u[4])
                i4 = something(findfirst(isequal(u[4]), slater_indices[l]), 0)
                s = (-1)^(i1+i2+i3+i4)
                slater_exchange[j,l] = (u, s)
            end
        end
    end 
    
    res = [Dict{Tuple{Int,Int},Float64}([]) for p=1:N, q=1:N]
    for p=1:N
        for q=1:N
            if p==q 
                for j=1:lena
                    if p in slater_indices[j]
                       add_xxx!(res[p,q], (j,j, 1.0))
                    end
                end
            end
            v = find_xxx([q,p], lena, slater_exchange, slater_indices)
            add_xxx!(res[p,q], v)
        end
    end
    slater1_rules = [[(key[1],key[2],val) for (key,val) in res[p,q]] for p=1:N, q=1:N]
    
    res = [Dict{Tuple{Int,Int},Float64}([]) for p=1:N, q=1:N, r=1:N, s=1:N]
    for p=1:N
        for q=1:N
            for r=1:N
                for s=1:N
                    if p==q && r==s && p!=r
                        for j=1:lena
                            if p in slater_indices[j] && r in slater_indices[j]                       
                                add_xxx!(res[p,q,r,s], (j,j, 0.5))
                            end
                        end
                    end
                    if p==s && r==q && p!=r
                        for j=1:lena
                            if p in slater_indices[j] && r in slater_indices[j]                       
                                add_xxx!(res[p,q,r,s], (j,j, -0.5)) 
                            end
                        end
                    end
                    if p==q && r!=s 
                        v = find_xxx([s,r], lena, slater_exchange, slater_indices, must_be_included=p)
                        add_xxx!(res[p,q,r,s], v)
                    end
                    if q==r && s!=p 
                        v0 = find_xxx([s,p], lena, slater_exchange, slater_indices, must_be_included=q)
                        v = [(j,k,-sigma) for (j,k, sigma) in v0]
                        add_xxx!(res[p,q,r,s], v)     
                    end     
                    v = find_xxx([s,r,q,p], lena, slater_exchange, slater_indices)
                    add_xxx!(res[p,q,r,s], v)
                    v0 = find_xxx([s,p,q,r], lena, slater_exchange, slater_indices)
                    v = [(j,k,-sigma) for (j,k, sigma) in v0]
                    add_xxx!(res[p,q,r,s], v)
                end
            end
        end
    end
    slater2_rules = [[(key[1],key[2],val) for (key,val) in res[p,q,r,s]] for p=1:N, q=1:N, r=1:N, s=1:N]
    
    orthogonalization_rules = [Tuple{Int,Int,Int}[] for p=1:N, q=1:N]
    for p=1:N
        for j=1:lena
            #k = findfirst(slater_indices[j], p)
            k = something(findfirst(isequal(p), slater_indices[j]), 0)
            if k>0
                for q=1:p-1
                    J = copy(slater_indices[j])
                    J[k] = q
                    J1 = sort(J)
                    #l = findfirst(slater_indices, J1)
                    l = something(findfirst(isequal(J1), slater_indices), 0)
                    if l>0
                        s = sign(J)*sign(J1)
                        push!(orthogonalization_rules[p,q], (j,l, s))
                    end
                end
            end
        end
    end    
    
    slater_indices, density_rules, density2_rules, slater_exchange, slater1_rules, slater2_rules, orthogonalization_rules
end


mutable struct MCTDHF1D <: TimeSplittingSpectralMethodComplex1D{Float64}
    m::Schroedinger1D
    f::Int # number of electrons
    N::Int # number of orbitals
    lena::Int # number of (independent) coefficients
    spins::Array{Int, 1}
    spin_restricted::Bool
    
    slater_indices    :: Array{Array{Int64,1},1}
    density_rules     :: Array{Array{Tuple{Int64,Int64,Int64},1},2}
    density2_rules    :: Array{Array{Tuple{Int64,Int64,Int64},1},4}
    slater_exchange   :: Array{Tuple{Array{Int64,1},Int64},2}
    slater1_rules     :: Array{Array{Tuple{Int64,Int64,Float64},1},2}
    slater2_rules     :: Array{Array{Tuple{Int64,Int64,Float64},1},4}
    orthogonalization_rules :: Array{Array{Tuple{Int64,Int64,Int64},1},2}

    Vee               :: Array{Float64,2}
    density_matrix    :: Array{Complex{Float64},2}
    density2_tensor   :: Array{Complex{Float64},4}
    u_pq              :: Array{Complex{Float64},1}
    u_pqs	      :: Array{Complex{Float64},1}

    k1 
    k2
    k3
    k4

    function MCTDHF1D(f::Integer, N::Integer, 
                      nx::Integer, xmin::Real, xmax::Real; spins::Array{Int,1}=ones(Int, N), 
                      spin_restricted::Bool=true,
                      potential1::Function=TSSM.none_1D, potential1_t::Function=TSSM.none_2D ,potential2::Function=TSSM.none_2D)
        m = Schroedinger1D(nx, xmin, xmax, potential=potential1, potential_t = potential1_t)
        lena = binomial(N,f)
        if spin_restricted
            spins = [(-1)^(k+1) for k=1:N]
        end
        slater_indices, density_rules, density2_rules, slater_exchange, slater1_rules, slater2_rules, orthogonalization_rules = 
              init_mctdhf_combinatorics(f, N)
        mm = new(m, f, N, lena, spins, spin_restricted, slater_indices, density_rules, 
                density2_rules, slater_exchange, slater1_rules, slater2_rules, orthogonalization_rules, 
                zeros(Float64, nx, nx),
                zeros(Complex{Float64},N,N),  zeros(Complex{Float64},N,N,N,N),
        zeros(Complex{Float64}, nx), zeros(Complex{Float64}, nx), nothing, nothing, nothing, nothing )
        set_potential2!(mm, potential2)
        mm
    end
end

function set_potential2!(m::MCTDHF1D, V::Function)
    xx = get_nodes(m.m)
    n = get_nx(m.m)
    for ix=1:n
        x = xx[ix]
        for iy=1:n
            y = xx[iy]
            m.Vee[ix, iy] = V(x,y)
        end
    end
end

set_potential1!(m::MCTDHF1D, V::Function) = set_potential!(m.m, V)
set_potential1_t!(m::MCTDHF1D, V::Function) = set_potential_t!(m.m, V)


mutable struct Orbital
    phi::WfSchroedinger1D
    spin::Int 
end

mutable struct WfMCTDHF1D <: WaveFunctionComplex1D{Float64}
    a::Array{Complex{Float64}, 1}
    o::Array{Orbital, 1}
    m::MCTDHF1D
    function WfMCTDHF1D(m::MCTDHF1D)
        a = zeros(Complex{Float64}, m.lena)
        if m.spin_restricted
            w = [WfSchroedinger1D(m.m) for j=1:m.N]
            o = [Orbital(w[div(j+1,2)], m.spins[j]) for j=1:m.N]
        else
            o = [Orbital(WfSchroedinger1D(m.m), m.spins[j]) for j=1:m.N]
        end
        new(a, o, m)
    end
end

function TSSM.wave_function(m::MCTDHF1D )
    WfMCTDHF1D(m) 
end

LinearAlgebra.norm(o::Orbital) = norm(o.phi)
TSSM.inner_product(o1::Orbital, o2::Orbital) = (o1.spin==o2.spin ? inner_product(o1.phi, o2.phi) : 0.0im)
TSSM.potential_matrix_element(o1::Orbital, o2::Orbital) = (o1.spin==o2.spin ? potential_matrix_element(o1.phi, o2.phi) : 0.0im)
TSSM.kinetic_matrix_element(o1::Orbital, o2::Orbital) = (o1.spin==o2.spin ? kinetic_matrix_element(o1.phi, o2.phi) : 0.0im)

function LinearAlgebra.scale!(o::Orbital, f::Number) 
    scale!(o.phi, f)
end

function LinearAlgebra.axpy!(o1::Orbital, o2::Orbital, f::Number)
    axpy!(o1.phi, o2.phi, f)
end

function LinearAlgebra.scale!(psi::WfMCTDHF1D, f::Number)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        scale!(psi.o[j], f)
    end
    psi.a[:] *= f
end

function LinearAlgebra.axpy!(psi1::WfMCTDHF1D, psi2::WfMCTDHF1D, f::Number)
    st = psi1.m.spin_restricted ? 2 : 1
    for j=1:st:psi1.m.N
        axpy!(psi1.o[j], psi2.o[j], f)
    end
    psi1.a[:] += f*psi2.a[:]
end

function Base.copy!(psi1::WfMCTDHF1D, psi2::WfMCTDHF1D)
    st = psi1.m.spin_restricted ? 2 : 1
    for j=1:st:psi1.m.N
        copy!(psi1.o[j].phi, psi2.o[j].phi)
    end
    for j=1:psi1.m.N
        psi1.o[j].spin = psi2.o[j].spin
    end
    psi1.a[:] = psi2.a[:]
end

using HDF5

function TSSM.save(psi::WfMCTDHF1D, filename::AbstractString)
    st = psi.m.spin_restricted ? 2 : 1
    for k=1:st:psi.m.N
       TSSM.save(psi.o[k].phi, filename, string("orbital_",k, "_real"),
             string("orbital_", k, "_imag", ), append=(k>1))
    end
    h5open(filename,"r+") do file 
        file["coefficients_real"] =  real(psi.a)
        file["coefficients_imag"] =  imag(psi.a)
        file["spins"] = Cint[psi.o[k].spin for k=1:psi.m.N]
        attrs(file)["number_of_particles"] = psi.m.f
        attrs(file)["number_of_orbitals"] = psi.m.N
        attrs(file)["spin_restricted"] = psi.m.spin_restricted ? 1 : 0
    end
    filename
end

function TSSM.load!(psi::WfMCTDHF1D, filename::AbstractString)
    st = psi.m.spin_restricted ? 2 : 1
    for k=1:st:psi.m.N
       load!(psi.o[k].phi, filename, string("orbital_",k, "_real"),
             string("orbital_", k, "_imag", ))
    end
    h5open(filename,"r") do file 
        psi.a = read(file["coefficients_real"])+ 1im*read(file["coefficients_imag"])
    end
    psi
end


function gen_density_matrix(psi::WfMCTDHF1D)
    m = psi.m
    N = m.N
    rho = m.density_matrix
    rho[:,:] .= 0.0
    for j=1:N
        for l=j:N
            for (u, v, s) in m.density_rules[j,l]
                rho[j,l] += s * conj(psi.a[u]) * psi.a[v]
                if l!=j
                   rho[l,j] = conj(rho[j,l])
                end
            end    
        end
    end
    nothing
end


function gen_density2_tensor(psi::WfMCTDHF1D; mult_inverse_density_matrix::Bool=true)
    m = psi.m
    N = m.N
    rho = m.density2_tensor
    rho[:,:,:,:] .= 0.0
    for j=1:N
        for l=j:N
            for p=1:N
                for q=1:N
                    for (u, v, s) in m.density2_rules[j,l,p,q]
                        rho[j,l,p,q] += s * conj(psi.a[u]) * psi.a[v]
                        if l!=j
                           rho[l,j,q,p] = conj(rho[j,l,p,q])
                        end
                    end
                end
            end    
        end
    end
    if mult_inverse_density_matrix
        X = bunchkaufman(m.density_matrix)
        for p=1:N
            for q=1:N
                rho[:,:,p,q] = X \ rho[:,:,p,q]
            end
        end
    end
    nothing
end


LinearAlgebra.norm(psi::WfMCTDHF1D) = norm(psi.a)
#Note, only correct if orbitals are ortonormal

function TSSM.inner_product(psi1::WfMCTDHF1D, psi2::WfMCTDHF1D)
    m = psi1.m
    if m ≠ psi2.m
        error("psi1 and psi2 must belong to the same method")
    end
    ip = zeros(Complex{Float64}, m.N,m.N)
    st = m.spin_restricted ? 2 : 1
    for j=1:st:m.N     
        for l=1:st:m.N
            ip[j,l] = inner_product(psi1.o[j], psi2.o[l])
            if m.spin_restricted && j+1<=m.N && l+1<=m.N
                ip[j+1,l+1] = ip[j,l]
            end
        end
    end

    ps = [(p, sign(p)) for p in (Combinatorics.permutations(1:m.f))]
    d = 0.0im
    for j=1:m.lena
        J = m.slater_indices[j]
        for l=1:m.lena
            L = m.slater_indices[l]
            aa = conj(psi1.a[j])*psi2.a[l]
            for (p,s) in ps
                L1 = getindex(L, p)
                d = d + s*aa*prod([ip[J[k],L1[k]] for k=1:m.f])
            end
        end
    end
    d
end

function TSSM.distance(psi1::WfMCTDHF1D, psi2::WfMCTDHF1D; expanded::Bool=false)  
    m = psi1.m
    if m ≠ psi2.m
        error("psi1 and psi2 must belong to the same method")
    end
    if expanded
        n2 = norm(psi1)^2+norm(psi2)^2-2*real(inner_product(psi1,psi2))
        if n2<0.0
            return sqrt(complex(n2))
        else
            return sqrt(n2)
        end     
    else
        n2 = norm(psi1.a-psi2.a).^2
        st = m.spin_restricted ? 2 : 1
        for j=1:st:m.N     
            n2 += distance(psi1.o[j].phi, psi2.o[j].phi)^2
        end
        return sqrt(n2)
    end
end


function TSSM.to_real_space!(psi::WfMCTDHF1D)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        to_real_space!(psi.o[j].phi)
    end
end


function TSSM.to_frequency_space!(psi::WfMCTDHF1D)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        to_frequency_space!(psi.o[j].phi)
    end
end


function set_zero!(psi::WfMCTDHF1D)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N 
        get_data(psi.o[j].phi, true)[:] = 0.0
    end
    psi.a[:] = 0.0
end

function TSSM.set!(psi::WfMCTDHF1D, x::Number)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N 
        set!(psi.o[j].phi, x)
    end
    psi.a[:] .= x
end



function gen_rhs1!(rhs::WfMCTDHF1D, psi::WfMCTDHF1D; include_kinetic_part::Bool=false, 
                   include_one_particle_potential_part::Bool=true)
    if !(include_kinetic_part||include_one_particle_potential_part)
        return # nothing to do
    end
    m = rhs.m
    st = m.spin_restricted ? 2 : 1
    if m ≠ psi.m
        error("rhs and psi must belong to the same method")
    end
    for q=1:st:m.N
        if include_kinetic_part
            add_apply_A!(psi.o[q].phi, rhs.o[q].phi, 1im)
        end
        if include_one_particle_potential_part
            add_apply_B!(psi.o[q].phi, rhs.o[q].phi, 1im)
        end
        h = inner_product(psi.o[q], rhs.o[q])
        for (j,l,f) in m.slater1_rules[q,q]
            rhs.a[j] += h*f*psi.a[l] 
        end
        if m.spin_restricted && q+1<=m.N 
            for (j,l,f) in m.slater1_rules[q+1,q+1]
                rhs.a[j] += h*f*psi.a[l] 
            end
        end
        for p=1:st:q-1
            if psi.o[p].spin==psi.o[q].spin
                h = inner_product(psi.o[p], rhs.o[q])
                for (j,l,f) in m.slater1_rules[q,p]
                    rhs.a[j] += h*f*psi.a[l] 
                    rhs.a[l] += conj(h)*f*psi.a[j] 
                end
                if m.spin_restricted &&  q+1<=m.N && p+1<=m.N
                    for (j,l,f) in m.slater1_rules[q+1,p+1]
                        rhs.a[j] += h*f*psi.a[l] 
                        rhs.a[l] += conj(h)*f*psi.a[j] 
                    end
                end
            end
        end
    end
end


function project_out_orbitals!(rhs:: WfMCTDHF1D, psi::WfMCTDHF1D)
    m = psi.m
    st = m.spin_restricted ? 2 : 1
    c = zeros(Complex{Float64},m.N)
    for p = 1:st:m.N
        for q = 1:st:m.N
            if psi.o[p].spin==psi.o[q].spin
                c[q] = inner_product(psi.o[q], rhs.o[p])
            end
        end
        for q = 1:st:m.N
            if psi.o[p].spin==psi.o[q].spin
                axpy!(rhs.o[p], psi.o[q], -c[q])
            end
        end
    end            
end


function gen_rhs2!(rhs::WfMCTDHF1D, psi::WfMCTDHF1D)
    m = rhs.m
    if m ≠ psi.m
        error("rhs and psi must belong to the same method")
    end
    n = get_nx(m.m)
    dx = (get_xmax(m.m)-get_xmin(m.m))/n
    u_pq = m.u_pq 
    u_pqs = m.u_pqs 
    to_real_space!(psi)
    to_real_space!(rhs)
    st = m.spin_restricted ? 2 : 1
    for p=1:st:m.N
        for q=1:st:p
            if psi.o[p].spin==psi.o[q].spin
                u_pq[:] = m.Vee * (conj(get_data(psi.o[p].phi, true)).*get_data(psi.o[q].phi, true))
                for s=1:st:m.N
                    u_pqs[:] = u_pq .* get_data(psi.o[s].phi, true)
                    for r=1:st:m.N
                        if psi.o[r].spin==psi.o[s].spin
                            
                            u = get_data(rhs.o[r].phi, true)
                            u[:] += (m.density2_tensor[r,s,p,q]*(m.f-1)*dx) * u_pqs                
                            if m.spin_restricted && s+1<=m.N && r+1<=m.N
                                u[:] += (m.density2_tensor[r+1,s+1,p,q]*(m.f-1)*dx) * u_pqs                
                            end
                            
                            h = dot(get_data(psi.o[r].phi, true), u_pqs) * dx^2
                            for (j,l,f) in m.slater2_rules[q,p,s,r]
                                rhs.a[j] += h*f*psi.a[l] 
                            end
                            if m.spin_restricted 
                                if q+1<=m.N && p+1<=m.N
                                    for (j,l,f) in m.slater2_rules[q+1,p+1,s,r]
                                        rhs.a[j] += h*f*psi.a[l] 
                                    end
                                end
                                if s+1<=m.N && r+1<=m.N
                                    for (j,l,f) in m.slater2_rules[q,p,s+1,r+1]
                                        rhs.a[j] += h*f*psi.a[l] 
                                    end
                                end
                                if q+1<=m.N && p+1<=m.N && s+1<=m.N && r+1<=m.N
                                    for (j,l,f) in m.slater2_rules[q+1,p+1,s+1,r+1]
                                        rhs.a[j] += h*f*psi.a[l] 
                                    end
                                end
                            end
                            if p!=q
                                h = conj(h)
                                for (j,l,f) in m.slater2_rules[p,q,r,s]
                                    rhs.a[j] += h*f*psi.a[l] 
                                end
                                if m.spin_restricted 
                                    if q+1<=m.N && p+1<=m.N 
                                        for (j,l,f) in m.slater2_rules[p+1,q+1,r,s]
                                            rhs.a[j] += h*f*psi.a[l] 
                                        end
                                    end
                                    if s+1<=m.N && r+1<=m.N
                                        for (j,l,f) in m.slater2_rules[p,q,r+1,s+1]
                                            rhs.a[j] += h*f*psi.a[l] 
                                        end
                                    end
                                    if q+1<=m.N && p+1<=m.N && s+1<=m.N && r+1<=m.N
                                        for (j,l,f) in m.slater2_rules[p+1,q+1,r+1,s+1]
                                            rhs.a[j] += h*f*psi.a[l] 
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                if p!=q
                    u_pq = conj(u_pq)
                    for s=1:st:m.N
                        u_pqs[:] = u_pq .* get_data(psi.o[s].phi, true)
                        for r=1:st:m.N
                            if psi.o[r].spin==psi.o[s].spin
                                u = get_data(rhs.o[r].phi, true)
                                u[:] += (m.density2_tensor[r,s,q,p]*(m.f-1)*dx) * u_pqs                
                                if m.spin_restricted && s+1<=m.N && r+1<=m.N
                                    u[:] += (m.density2_tensor[r+1,s+1,q,p]*(m.f-1)*dx) * u_pqs                
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


function gen_rhs!(rhs::WfMCTDHF1D, psi::WfMCTDHF1D; include_kinetic_part::Bool=false, 
                   include_one_particle_potential_part::Bool=true )
    global COUNT_B += 1
    if rhs.m ≠ psi.m
        error("rhs and psi must belong to the same method")
    end
    orthonormalize_orbitals!(psi)
    gen_density_matrix(psi)
    gen_density2_tensor(psi)
    set!(rhs, 0.0)
    gen_rhs1!(rhs, psi, include_kinetic_part=include_kinetic_part,
                        include_one_particle_potential_part=include_one_particle_potential_part)
    gen_rhs2!(rhs, psi)
    project_out_orbitals!(rhs, psi)
    scale!(rhs, -1im)
end



function orthonormalize_orbitals!(psi::WfMCTDHF1D)
    m = psi.m
    g = zeros(Complex{Float64}, m.N)
    st = m.spin_restricted ? 2 : 1

    for p = 1:st:m.N
        for q=1:st:p-1
            g[q] = inner_product(psi.o[q], psi.o[p])
            if m.spin_restricted && q+1<=m.N
                g[q+1] = g[q]
            end    
        end
        a1 = zeros(Complex{Float64}, m.lena)
        for q=1:st:p-1
            if psi.o[p].spin==psi.o[q].spin
                axpy!(psi.o[p], psi.o[q], -g[q])
                for (j, l, s) in m.orthogonalization_rules[p,q]
                    a1[l] += s*g[q]*psi.a[j]
                end
            end    
        end
        f = norm(psi.o[p])
        scale!(psi.o[p],1/f)
        for j=1:m.lena
            if p in m.slater_indices[j]
                a1[j] += f*psi.a[j]
            else
                a1[j] += psi.a[j]
            end
        end
        psi.a[:] = a1

        if m.spin_restricted && p+1<=m.N
            a1 = zeros(Complex{Float64}, m.lena)
            for q=1:st:p-1
                if psi.o[p].spin==psi.o[q].spin
                    if q+1<=m.N
                        for (j, l, s) in m.orthogonalization_rules[p+1,q+1]
                            a1[l] += s*g[q+1]*psi.a[j]
                        end
                    end
                end
            end    
            for j=1:m.lena
                if p+1 in m.slater_indices[j]
                    a1[j] += f*psi.a[j]
                else
                    a1[j] += psi.a[j]
                end
            end    
            psi.a[:] = a1
        end
    end
    psi
end

function LinearAlgebra.normalize!(psi::WfMCTDHF1D)
    orthonormalize_orbitals!(psi) 
    psi.a[:] /= norm(psi)
end


function potential_energy_1(psi::WfMCTDHF1D)
    m = psi.m
    V = 0.0
    st = m.spin_restricted ? 2 : 1
    for p=1:st:m.N    
        h = potential_matrix_element(psi.o[p], psi.o[p])
        for (j,l,f) in m.slater1_rules[p,p]
            V += real(h*f*psi.a[j]*conj(psi.a[l]))
        end
        if m.spin_restricted && p+1<=m.N+1
            for (j,l,f) in m.slater1_rules[p+1,p+1]
                V += real(h*f*psi.a[j]*conj(psi.a[l]))
            end
        end
        for q=1:st:p-1
            if psi.o[p].spin==psi.o[q].spin
                h = potential_matrix_element(psi.o[p], psi.o[q])
                for (j,l,f) in m.slater1_rules[p,q]
                    V += 2*real(h*f*psi.a[j]*conj(psi.a[l]))
                end
                if m.spin_restricted &&  q+1<=m.N && p+1<=m.N+1
                    for (j,l,f) in m.slater1_rules[p+1,q+1]
                        V += 2*real(h*f*psi.a[j]*conj(psi.a[l]))
                    end
                end
            end
        end
    end
    real(V)
end


function TSSM.kinetic_energy(psi::WfMCTDHF1D)
    m = psi.m
    V = 0
    st = m.spin_restricted ? 2 : 1
    for p=1:st:m.N    
        h = kinetic_matrix_element(psi.o[p], psi.o[p])
        for (j,l,f) in m.slater1_rules[p,p]
            V += real(h*f*psi.a[j]*conj(psi.a[l]))
        end
        if m.spin_restricted && p+1<=m.N+1
            for (j,l,f) in m.slater1_rules[p+1,p+1]
                V += real(h*f*psi.a[j]*conj(psi.a[l]))
            end
        end
        for q=1:st:p-1
            if psi.o[p].spin==psi.o[q].spin
                h = kinetic_matrix_element(psi.o[p], psi.o[q])
                for (j,l,f) in m.slater1_rules[p,q]
                    V += 2*real(h*f*psi.a[j]*conj(psi.a[l]))
                end
                if m.spin_restricted &&  q+1<=m.N && p+1<=m.N+1
                    for (j,l,f) in m.slater1_rules[p+1,q+1]
                        V += 2*real(h*f*psi.a[j]*conj(psi.a[l]))
                    end
                end
            end
        end
    end
    V
end


function potential_energy_2(psi::WfMCTDHF1D)
    m = psi.m
    V = 0
    n = get_nx(m.m)
    dx = (get_xmax(m.m)-get_xmin(m.m))/n
    u_pq = m.u_pq 
    u_pqs = m.u_pqs 
    to_real_space!(psi)
    st = m.spin_restricted ? 2 : 1
    for p=1:st:m.N
        for q=1:st:p
            if psi.o[p].spin==psi.o[q].spin
                u_pq[:] = m.Vee * (conj(get_data(psi.o[p].phi, true)).*get_data(psi.o[q].phi, true))
                for s=1:st:m.N
                    u_pqs[:] = u_pq .* get_data(psi.o[s].phi, true)
                    for r=1:st:m.N
                        if psi.o[r].spin==psi.o[s].spin
                            h = dot(get_data(psi.o[r].phi, true), u_pqs)
                            for (j,l,f) in m.slater2_rules[q,p,s,r]
                                V += h*f*conj(psi.a[j])*psi.a[l]
                            end
                            if m.spin_restricted 
                                if q+1<=m.N && p+1<=m.N 
                                    for (j,l,f) in m.slater2_rules[q+1,p+1,s,r]
                                        V += h*f*conj(psi.a[j])*psi.a[l]
                                    end
                                end    
                                if s+1<=m.N && r+1<=m.N
                                    for (j,l,f) in m.slater2_rules[q,p,s+1,r+1]
                                        V += h*f*conj(psi.a[j])*psi.a[l]
                                    end
                                end    
                                if q+1<=m.N && p+1<=m.N && s+1<=m.N && r+1<=m.N
                                    for (j,l,f) in m.slater2_rules[q+1,p+1,s+1,r+1]
                                        V += h*f*conj(psi.a[j])*psi.a[l]
                                    end
                                end    
                            end
                            if p!=q
                                h = conj(h)
                                for (j,l,f) in m.slater2_rules[p,q,r,s]
                                    V += h*f*conj(psi.a[j])*psi.a[l]
                                end
                                if m.spin_restricted 
                                    if q+1<=m.N && p+1<=m.N 
                                        for (j,l,f) in m.slater2_rules[p+1,q+1,r,s]
                                            V += h*f*conj(psi.a[j])*psi.a[l]
                                        end
                                    end
                                    if s+1<=m.N && r+1<=m.N
                                        for (j,l,f) in m.slater2_rules[p,q,r+1,s+1]
                                            V += h*f*conj(psi.a[j])*psi.a[l]
                                        end
                                    end
                                    if q+1<=m.N && p+1<=m.N && s+1<=m.N && r+1<=m.N
                                        for (j,l,f) in m.slater2_rules[p+1,q+1,r+1,s+1]
                                            V += h*f*conj(psi.a[j])*psi.a[l]
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    V *= dx^2
    real(V)
end



TSSM.potential_energy(psi::WfMCTDHF1D) = potential_energy_1(psi) + potential_energy_2(psi)


function TSSM.get_energy_expectation_deviation(psi::WfMCTDHF1D; tmp::WfMCTDHF1D=psi.m.k1)
    gen_rhs!(tmp, psi, include_kinetic_part=true, include_one_particle_potential_part=true )
    E = -imag(dot(psi.a, tmp.a))
    dev = norm(tmp.a+1im*E*psi.a)
    E, dev
end

TSSM.interaction_energy(psi::WfMCTDHF1D) = 0.0 #dummy


function TSSM.set_time!(psi::WfMCTDHF1D, t::Number)
   st = psi.m.spin_restricted ? 2 : 1
   for j=1:st:psi.m.N
       set_time!(psi.o[j].phi, t)
   end
end

TSSM.get_time(psi::WfMCTDHF1D) = get_time(psi.o[1].phi)
TSSM.set_propagate_time_together_with_A!(m::MCTDHF1D, flag::Bool) = set_propagate_time_together_with_A!(m.m, flag)
TSSM.get_propagate_time_together_with_A(m::MCTDHF1D) = get_propagate_time_together_with_A(m.m)

function TSSM.propagate_A!(psi::WfMCTDHF1D, dt::Number)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        propagate_A!(psi.o[j].phi, dt)
    end
end

function TSSM.propagate_B!(psi::WfMCTDHF1D, dt::Number)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        propagate_B!(psi.o[j].phi, dt)
    end
end

function TSSM.imaginary_time_propagate_A!(psi::WfMCTDHF1D, dt::Real)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        imaginary_time_propagate_A!(psi.o[j].phi, dt)
    end
end

function TSSM.imaginary_time_propagate_B!(psi::WfMCTDHF1D, dt::Real)
    st = psi.m.spin_restricted ? 2 : 1
    for j=1:st:psi.m.N
        imaginary_time_propagate_B!(psi.o[j].phi, dt)
    end
end

function TSSM.add_apply_A!(this::WfMCTDHF1D, other::WfMCTDHF1D, coefficient::Number=1.0)
    if this.m ≠ other.m
        error("this and other must belong to the same method")
    end
    st = this.m.spin_restricted ? 2 : 1
    for j=1:st:this.m.N
        add_apply_A!(this.o[j].phi, other.o[j].phi, coefficient)
    end
end

function TSSM.add_phi_A!(this::WfMCTDHF1D, other::WfMCTDHF1D, dt::Number, n::Integer, coefficient::Number=1.0)
    if this.m ≠ other.m
        error("this and other must belong to the same method")
    end
    st = this.m.spin_restricted ? 2 : 1
    for j=1:st:this.m.N
        add_phi_A!(this.o[j].phi, other.o[j].phi, dt, n, coefficient)
    end
    other.a[:] += coefficient/factorial(n)*this.a[:]
end


