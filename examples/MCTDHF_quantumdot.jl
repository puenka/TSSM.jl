# Parse parameters
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--exp"
        help = "Specify the experiment to run (groundstate, stab, conv, adapt)."
        required = true
    "--method"
        help = "Specify the method to use."
        arg_type = String
        required = true
    "--N"
        arg_type = Int
        default = 4
    "--steps"
        help = "Number of steps used in the time interval [0,12]."
        arg_type = Int
        default = 1000
    "--no-ref"
        help = "Don't use reference solution in conv experiment, instead just save resulting wf."
        action = :store_true
    "--tol"
        help = "Tolerance for the adaptivity experiment"
        arg_type = Float64
        default = 1e-5
end

parsed_args = parse_args(s)
const method_string = parsed_args["method"]
const steps = parsed_args["steps"]
const N = parsed_args["N"]
const experiment = parsed_args["exp"]
const adaptive_tol = parsed_args["tol"]

# Run experiment
using TSSM
using JLD

# The following line uses my installation directory, modify if necessary
TSSM_dir = joinpath(homedir(), ".julia/dev/TSSM")
include(joinpath(TSSM_dir, "MCTDHF/mctdhf1d.jl"))
include(joinpath(TSSM_dir, "MCTDHF/check.jl"))
include(joinpath(TSSM_dir, "examples/time_propagators.jl"))

V1(x) = 1. / 32 * (x^2)
V2(x,y) = 1/sqrt((x-y)^2+1/16)
V(x,y) = V1(x) + V1(y) + V2(x,y)

const interval_bound = 20
const nx = Int(512*interval_bound/10)
const t0 = 0
const groundstate_tol=1e-9

SHIFT_B = 0

println("N = ", N, ", nx = ", nx, " on [-", interval_bound, ",", interval_bound, "]")

function method_from_string(variant::String)
    if variant == "yoshidaRK4"
        g = [1.351207191959657634, -1.702414383919315268, 1.351207191959657634] # Yoshida
        a, b = get_coeffs_composition(g)
        method = SplittingRK4BMethod(a,b)
        println("yoshida splitting")
    elseif variant == "yoshidaMP"
        g = [1.351207191959657634, -1.702414383919315268, 1.351207191959657634] # Yoshida
        method = CompositionMethod(g, 3)
        println("yoshida splitting using MP")
    elseif variant == "suzukiRK4"
        g = [1/(4-4^(1/3)),1/(4-4^(1/3)),-4^(1/3)/(4-4^(1/3)), 1/(4-4^(1/3)), 1/(4-4^(1/3))] # Suzuki
        a, b = get_coeffs_composition(g)
        method = SplittingRK4BMethod(a,b)
        println("suzuki splitting")
    elseif variant == "suzukiMP"
        g = [1/(4-4^(1/3)),1/(4-4^(1/3)),-4^(1/3)/(4-4^(1/3)), 1/(4-4^(1/3)), 1/(4-4^(1/3))] # Suzuki
        method = CompositionMethod(g, 3)
        println("suzuki splitting using MP")
    elseif variant == "strang"
        a, b = [0.5, 0.5], [1.0, 0.0]
        method = SplittingRK4BMethod(a, b, secondorder=true)
        println("strang splitting")
    elseif variant == "krogstad"
        method = ExponentialRungeKutta(:krogstad)
        println("krogstad")
    elseif variant == "rk4"
        method = ExponentialRungeKutta(:rk4)
        println("rk4")
    elseif variant == "rklawson"
        method = ExponentialRungeKutta(:lawson)
        println("runge kutta lawson")

    elseif variant == "adamslawson31"
        method = ExponentialMultistep(3, version=2, iters=1, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("adamslawson 3 + 1it")
    elseif variant == "adamslawson40"
        method = ExponentialMultistep(4, version=2, iters=0, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("adamslawson 4 + 0it")
    elseif variant == "expmultistep31"
        method = ExponentialMultistep(3, version=1, iters=1, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("exponential multistep 3 + 1it")
    elseif variant == "expmultistep40"
        method = ExponentialMultistep(4, version=1, iters=0, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("exponential multistep 4 + 0it")

    elseif variant == "adamslawson51"
        method = ExponentialMultistep(5, version=2, iters=1, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("adamslawson 5 + 1it")
    elseif variant == "adamslawson60"
        method = ExponentialMultistep(6, version=2, iters=0, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("adamslawson without 6 + 0it")
    elseif variant == "expmultistep51"
        method = ExponentialMultistep(5, version=1, iters=1, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("exponential multistep 5 + 1it")
    elseif variant == "expmultistep60"
        method = ExponentialMultistep(6, version=1, iters=0, starting_method=ExponentialRungeKutta(:lawson), starting_subdivision=50)
        println("exponential multistep without 6 + 0it")
    else
        throw(ArgumentError("\""*variant*"\" is not a supported method"))
    end
    
    try
        s = parse(Int, variant[end-1])
        println(s, " step method needs ", s-1, " additional starting values")
        global SHIFT_B = -(s-1)*4*50
        println("shift COUNT_B by ", SHIFT_B)
    catch ArgumentError
    end
    return method
end

function conv(variant::String, steps::Int)
    #laser field
    E(t) = sin(2*t)
    tend = 12
    dt=tend/steps

    V1_t(x, t) = E(t)*x
    V_t(x,y,t) = E(t)*(x+y)

    m = MCTDHF1D(2, N, nx, -interval_bound, interval_bound, potential1=V1, potential1_t=V1_t, potential2=V2, spin_restricted=true)
    psi = wave_function(m)

    TSSM.load!(psi, "wfquantumdot_f2_n"*string(N)*"_nx"*string(nx)*"_t0_1e-9.h5")
    set_time!(psi, t0)

    method = method_from_string(variant)
    times, energies, norms = propagate_equidistant!(method, psi, t0, dt, steps)

    if !isapprox(times[end], tend)
        println("Method did not converge!")
        return
    end

    TSSM.save(psi, "results_conv/wfquantumdot_conv_f2_n"*string(N)*"_nx"*string(nx)*"_t"*string(tend)*"_"*variant*"_"*string(steps)*".h5")

    if !parsed_args["no-ref"]
        psi_ref = wave_function(m);
        TSSM.load!(psi_ref, "results_conv/wfquantumdot_conv_f2_n"*string(N)*"_nx"*string(nx)*"_t"*string(tend)*"_adamslawson51_12000.h5")
        err = distance(psi, psi_ref)
        println("err:", err)
        JLD.save("results_conv/wfquantumdot_conv_f2_n"*string(N)*"_nx"*string(nx)*"_t"*string(tend)*"_"*variant*"_"*string(steps)*".jld", "times", times, "energies", energies, "dt", dt, "err", err, "B_calls", COUNT_B+SHIFT_B)
    end
end

function stab(variant::String, steps::Int)
    #no laser field
    E(t) = 0
    tend = 70
    dt=tend/steps

    V1_t(x, t) = E(t)*x
    V_t(x,y,t) = E(t)*(x+y)

    m = MCTDHF1D(2, N, nx, -interval_bound, interval_bound, potential1=V1, potential1_t=V1_t, potential2=V2, spin_restricted=true);
    psi = wave_function(m);

    TSSM.load!(psi, "wfquantumdot_f2_n"*string(N)*"_nx"*string(nx)*"_t0_1e-9.h5")
    set_time!(psi, t0)

    method = method_from_string(variant)

    times, energies, norms = propagate_equidistant!(method, psi, t0, dt, steps)

    JLD.save("results_stab/wfquantumdot_stab_f2_n"*string(N)*"_nx"*string(nx)*"_t"*string(tend)*"_"*variant*"_"*string(steps)*".jld", "times", times, "energies", energies, "dt", dt, "norms", norms)
end

function adapt(variant::String, tol::Float64)
    E(t) = sin(2*t)
    tend = 12

    V1_t(x, t) = E(t)*x
    V_t(x,y,t) = E(t)*(x+y)

    m = MCTDHF1D(2, N, nx, -interval_bound, interval_bound, potential1=V1, potential1_t=V1_t, potential2=V2, spin_restricted=true);
    psi = wave_function(m);
    TSSM.load!(psi, "wfquantumdot_f2_n"*string(N)*"_nx"*string(nx)*"_t0_1e-9.h5")
    set_time!(psi, t0)

    times, energies, norms, stepsizes = propagate_adaptive!(psi, t0, tend, variant, tol)

    TSSM.save(psi, "results_adapt/wfquantumdot_adapt_f2_n"*string(N)*"_nx"*string(nx)*"_t"*string(tend)*"_"*variant*"_"*string(tol)*".h5")
    if !parsed_args["no-ref"]
        psi_ref = wave_function(m);
        TSSM.load!(psi_ref, "results_conv/wfquantumdot_conv_f2_n"*string(N)*"_nx"*string(nx)*"_t"*string(tend)*"_adamslawson51_12000.h5")
        err = distance(psi, psi_ref)
        println("err:", err)
        JLD.save("results_adapt/wfquantumdot_adapt_f2_n"*string(N)*"_nx"*string(nx)*"_"*variant*"_"*string(tol)*".jld", "times", times, "energies", energies, "norms", norms, "stepsizes", stepsizes, "B_calls", COUNT_B, "err", err)
    end
end

function propagate_equidistant!(method, psi, t0, dt, steps)
    times = Vector{Float64}()
    energies = Vector{Float64}()
    norms = Vector{Float64}()
    step_counter = 1
    time0 = time()
    for (step, tsi) in EquidistantTimeStepper(method, psi, t0, dt, steps)
        n = norm(psi)
        E_pot = potential_energy(psi)
        E_kin = kinetic_energy(psi)
        E_tot = E_pot+E_kin
        if step_counter % 50 == 0 || step == steps
            @printf("%5i  %14.10f  %14.10f  %14.10f  %14.10f  %14.10f  %14.10f %6i %10.2f\n",
                step_counter, get_time(psi), E_pot, E_kin, E_tot, dt, n, COUNT_B+SHIFT_B, time()-time0)
        end
        append!(times, get_time(psi))
        append!(energies, E_tot)
        append!(norms, n)
        step_counter += 1
        if n > 1e10
            break
        end
    end
    return times, energies, norms
end

function propagate_adaptive!(psi, t0, tend, variant, tol)
    version = 0
    if variant[1:end-1] == "adamslawson"
        version = 2
    elseif variant[1:end-1] == "expmultistep"
        version = 1
    else
        throw(ArgumentError("\""*variant*"\" is not a valid method"))
    end
    method = try
        AdaptiveAdamsLawson(parse(Int, variant[end]), version=version)
    catch ArgumentError
        throw(ArgumentError("The number of steps could not be parsed from \""*variant*"\". Only single digits are supported"))
    end

    times = Vector{Float64}()
    energies = Vector{Float64}()
    stepsizes = Vector{Float64}()
    norms = Vector{Float64}()
    step_counter = 1
    time0 = time()
    told = t0
    for (t, tsi) in AdaptiveTimeStepper(method, psi, t0, tend, tol, 0.00001)
        n = norm(psi)
        E_pot = potential_energy(psi)
        E_kin = kinetic_energy(psi)
        E_tot = E_pot+E_kin
        stepsize = t - told
        if step_counter % 50 == 0
            @printf("%5i  %14.10f  %14.10f  %14.10f  %14.10f  %14.10f  %14.10f  %10.2f\n",
                step_counter, get_time(psi), n, E_pot, E_kin, E_tot, stepsize, time()-time0)
        end
        append!(times, get_time(psi))
        append!(energies, E_tot)
        append!(stepsizes, stepsize)
        append!(norms, n)
        told = t
        step_counter += 1
    end
    return times, energies, norms, stepsizes
end

if experiment == "groundstate"
    println("calculate groundstate with tolerance ", groundstate_tol)
    include(joinpath(TSSM_dir, "MCTDHF/propagators.jl"))
    m = MCTDHF1D(2, N, nx, -interval_bound, interval_bound, potential1=V1, potential2=V2, spin_restricted=true)
    psi = wave_function(m)
    groundstate!(psi, dt=0.1, max_iter=500000, output_step=100, tol=groundstate_tol)
    TSSM.save(psi, "wfquantumdot_f2_n"*string(N)*"_nx"*string(nx)*"_t0_"*string(groundstate_tol)*".h5")
elseif experiment == "conv"
    println("calculate solution using method ", method_string, " with ", steps, " steps")
    conv(method_string, steps)
elseif experiment == "stab"
    stab(method_string, steps)
elseif experiment == "adapt"
    println("calculate solution using adaptive method ", method_string, " with tolerance ", adaptive_tol)
    adapt(method_string, adaptive_tol)
else
    throw(ArgumentError("\"" * experiment * "\" is not an implemented experiment type"))
end
