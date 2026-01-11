using Distributed
addprocs(47)  # Adjust the number of workers as needed
#@everwhere Pkg.activate
@everywhere using Profile
@everywhere using Random
@everywhere using Plots
@everywhere using LaTeXStrings
@everywhere using GLM
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using CSV
@everywhere using FFTW
@everywhere using Printf
@everywhere using Colors
@everywhere using DSP
@everywhere using DataStructures
@everywhere using LinearAlgebra
@everywhere using Plots



current_dir = @__DIR__

default(xlabelfontsize=24, ylabelfontsize=24, xtickfontsize=24, ytickfontsize=24, left_margin = 5Plots.mm, right_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    """
    Runge-kutta 4 method. Previous versions of this code used Heun method, which allowed for the implementation for multiplicative noise,
    but at the cost of lower precision. In theory, the noise term in the rate equation for the electric field should be multiplicative
    (multiplied by the square root of the carrier number). Nevertheless, after the ignition of the laser, N~N_{th}, so additive noise
    multiplied by the proper coefficients and the square root of the carrier number at threshold. As a side note, carrier number has its 
    own rate equation simulated. Nevertheless, because what we just said, sometimes not even the carrier number will be simulated and instead
    left as N = N_{th}. Given that is not possible to replicate exactly the evolution of the laser (we are dealing with a chaotic system and
    thus small perturbations grow exponentially over time) and we are just trying to end up in the same attractor, this also would be a good
    approximation.


    There is another important thing to mention. This code uses a circular buffer for the electric with headers. When computing the next
    step of the simulation, we use RK4. The problem raises in k_2 and in k_3, where they ask for the value of E(t - τ + h/2), which has not
    been simulated given that h is our time step (that is, the smallest temporal resolution available). In theory one should use interpolation
    methods to estimate this value of the electric field. Nevertheless we haven't incorporated this because, again, we are satisfied with just 
    arriving to the right attractor.
    """
@everywhere function RK4_method(x_E,x_N, h, sqr_h, q_E,q_N, g , 
    k_E_array_1,k_N_array_1,k_E_array_2,k_N_array_2,k_E_array_3,k_N_array_3,
    k_E_array_4,k_N_array_4, x_k_E_array, x_k_E_last, x_k_N_array, E, E_last, N, n, header_actual,
    τ_inj_index_array,τ_fb_index_array,t, K_inj_matrix_result, K_inj_matrix, h_half,u)

    #k_1
    u .= randn(n)
    k_E_array_1 .= q_E(E_last,N,E) 
    k_N_array_1 .= q_N(E_last,N)  

    @inbounds for i in Int.(union(τ_inj_index_array,τ_fb_index_array))
        x_k_E_array[i,:] .= E[i,:] .+ h_half.*k_E_array_1
    end
    x_k_E_last .= E_last .+ h_half.*k_E_array_1
    x_k_N_array .= N .+ h_half.*k_N_array_1

    t += h_half
    K_inj_matrix_result .= K_inj_matrix(t)

    #k_2

    k_E_array_2 .= q_E(x_k_E_last,x_k_N_array,x_k_E_array) 
    k_N_array_2 .= q_N(x_k_E_last,x_k_N_array)  

    @inbounds for i in Int.(union(τ_inj_index_array,τ_fb_index_array))
        x_k_E_array[i,:] .= E[i,:] .+ h_half.*k_E_array_2
    end
    x_k_E_last .= E_last .+ h_half.*k_E_array_2
    x_k_N_array .= N .+ h_half.*k_N_array_2

    #k_3

    k_E_array_3 .= q_E(x_k_E_last,x_k_N_array,x_k_E_array) 
    k_N_array_3 .= q_N(x_k_E_last,x_k_N_array)  

    @inbounds for i in Int.(union(τ_inj_index_array,τ_fb_index_array))
        x_k_E_array[i,:] .= E[i,:] .+ h.*k_E_array_3
    end
    x_k_E_last .= E_last .+ h.*k_E_array_3
    x_k_N_array .= N .+ h.*k_N_array_3

    t += h_half
    K_inj_matrix_result .= K_inj_matrix(t)
    
    #k_4

    k_E_array_4 .= q_E(x_k_E_last,x_k_N_array,x_k_E_array) 
    k_N_array_4 .= q_N(x_k_E_last,x_k_N_array)  




    x_E .= E_last .+ (h/6).*(k_E_array_1 .+ 2 .* k_E_array_2 .+ 2 .* k_E_array_3 .+ k_E_array_4) .+ g.*sqr_h.*u
    x_N .= N .+ (h/6).*(k_N_array_1 .+ 2 .* k_N_array_2 .+ 2 .* k_N_array_3 .+ k_N_array_4)
    return t
end
"""
Simulation for four lasers in a all-to-all topology (delayed coupling) with optical feedback (delay)
----------------------------------------------------------------------
Parameters
----------
κ_fb: Feedback coefficient
κ_inj: Injection coefficient
ξ_sp: Quantum internnal efficiency
f: Free-running frequency 
τ_ph: Photon lifetime 
τ_s: Carrier lifetime
g_N: Differentical gain coefficient
α: Linewidth enhancement factor
τ_1: Delay time for the optical feedback
τ_inj: Delay time for the coupling between lasers
I_th: Threshold current
I_r: Injection current
N_0: Transparency carrier density
s: Gain saturation factor
q_charge: Charge of the electron
τ_in_1: Injection time
c: Speed of light
λ: Wavelength
ω_0_1: Angular frequency
Δf: Frequency detuning
tf: Final time
h: Step size
Δt: Time step for saving results
τ_index: Index for the delay time
τ_index_inj: Index for the injection time
----------
Δt: time step for saving results
tf: final time
g(x,t): Noise term. Here represents, with the ξ, the noise caused by the population inversion. This term should be added in both
the carrier number and electric field equations, but for the population rate equation is negligible
h: step size 
q(x): Deterministic term 
"""



@everywhere function simulation_4_lasers(f_1, f_2, f_3, f_4,
                            ξ_sp, tf)
    #Parameters for the simulation
    k_inj = 0.15
    k_fb = 0.15
    τ_in = 10*10^(-12)
    s = 1 * 10^(-7)
    q_charge= 1.60217662*10^(-19)
    c = 299792458


    β_sp = 10^(-3)
        
    #Parameters more specific for the simulation
    h = 0.2e-12
    sqr_h = sqrt(h)
    Δt = 50
    time_saved = 400e-9
    tf_steps = Int(round(tf/h)) +1
    steps_saved = Int(round((time_saved) / (h))) + 1 
    steps_saved_with_jump = Int(round((time_saved) / (h*Δt))) + 1
    time_start_saving = Int.(tf_steps-steps_saved) 

    #Parameters that requiere arrays (some of them will be transformed into matrix)
    α_array = [2.25,2.3,2.27,2.3]
    τ_ph_array = [2.53 * 10^(-12), 2.26 * 10^(-12), 1.91 * 10^(-12), 2.3 * 10^(-12)]
    τ_s_array = [1.47 * 10^(-9), 1.5 * 10^(-9), 1.47 * 10^(-9), 1.42 * 10^(-9)]
    g_N_array = [23.61* 10^(3), 23.01 * 10^(3), 27.43 * 10^(3), 20.09 * 10^(3)]
    τ_array = [10.457 * 10^(-9), 10.352 * 10^(-9), 10 * 10^(-9), 10.633 * 10^(-9)]
    #τ_in_array = [10*10^(-12), 10*10^(-12), 10*10^(-12), 10*10^(-12)]
    N_0_array = [2.547 * 10^7, 2.797 * 10^7, 1.918 * 10^7, 2.247 * 10^7]
    f_array = [f_1,f_2,f_3,f_4]

    #Calculation of the currents
    I_th_array = [4.6 * 10^(-3), 5.04 * 10^(-3), 4.17 * 10^(-3), 4.66 * 10^(-3)]
    I_th_ratio_array = [1.2,1.2,1.2,1.2]
    I_r_array = I_th_array .* I_th_ratio_array

    #Creation of some matrices for the parameters
    τ_ph_array_inverse = 1 ./τ_ph_array
    T_ph_inverse_matrix = Diagonal(τ_ph_array_inverse)
    τ_s_array_inverse = 1 ./τ_s_array
    T_s_inverse_matrix = Diagonal(τ_s_array_inverse)


    α_matrix = Diagonal(0.5.*(1 .+ 1im.*α_array))
    n = length(α_array)

    f_matrix = zeros(eltype(f_array), n, n)
    for i in 1:length(f_array)
        f_matrix[:,i] .= f_array[i]  
    end

    Δf_matrix = f_array .- f_array'

    #Creation of the delay-coupling matrix
    number_tau_inj = sum(1:n-1)
    τ_inj_array = zeros(number_tau_inj)
    k=1
    for i in 1:n
        for j in (i+1):n
            τ_inj_array[k] = 0.5*(τ_array[i] + τ_array[j])
            k = k + 1
        end
    end

    τ_inj_matrix = 0.5 .* (τ_array .+ τ_array') 
    τ_inj_matrix[diagind(τ_inj_matrix)] .= 0.0


    #Here we create the delay arrays and matrices
    #One of the feedback delays has to be the largest delay between the ones from the feedback set and from those of the injection set
    τ = maximum(τ_array)
    τ_index = Int(round(τ/h))

    τ_inj_index_array = zeros(length(τ_inj_array))
    τ_fb_index_array = zeros(n)

    for i in 1:length(τ_inj_array)
        τ_inj_index_array[i] = Int(floor((τ - τ_inj_array[i]) / (h) + 1)) 
    end

    for i in 1:length(τ_array)
        τ_fb_index_array[i] = Int(round((τ - τ_array[i]) / (h) + 1)) 
    end

    τ_inj_index_array .= Int.(τ_inj_index_array)
    τ_fb_index_array .= Int.(τ_fb_index_array) 

    τ_inj_index_matrix = Int.(floor.( 0.5 .* (τ_fb_index_array .+ τ_fb_index_array')))
    τ_inj_index_matrix[diagind(τ_inj_index_matrix)] .= 0.0
    τ_inj_index_matrix .= Int.(τ_inj_index_matrix)

    #Here we stablish the matrices for the results

    E_matrix = ones(Int(round(τ_index) + 1),n) .+ 0im
    E_last = zeros(ComplexF64,n)
    E_last .=E_matrix[end,:]
    E_saved = zeros(ComplexF64, Int(steps_saved_with_jump),n)

    N_array = zeros(Float64,n)
    N_array .= 1.1 .* N_0_array
    #N_saved = zeros(Int(round(tf / (h * Δt))) + 1,n)
    #N_saved[1,:] = N_array

    #Calculation of the coefficients in the equations 
    exp_fb_array = exp.(-2*pi*1im.*f_array.*τ_array)
    K_fb_matrix = Diagonal((k_fb/τ_in).*exp_fb_array)


    K_inj_matrix_const = (k_inj/τ_in)*exp.(-2*pi*1im.*f_matrix.*τ_inj_matrix)
    K_inj_matrix_const[diagind(K_inj_matrix_const)] .= 0.0
    K_inj_matrix_result = zeros(ComplexF64,n,n)
    function K_inj_matrix(t, K_inj = K_inj_matrix_result, constant = K_inj_matrix_const, Δf = Δf_matrix)
        K_inj .=  constant .* exp.(-2*pi*1im .* Δf .* t)
        return K_inj
    end
    K_inj_matrix_result .= K_inj_matrix(0)

    I_r_array_divided_q = I_r_array./q_charge

    
    #Arrays where the results of the current time will be stored
    OutE = zeros(ComplexF64,n) #Electric field
    OutN = zeros(Float64,n) #Carrier number
    OutG = zeros(Float64,n,n) #Optical Gain

    #Functions for the electric fiels with delay from the optical feedback and the delayed-coupling
    E_delay_fb_array = zeros(ComplexF64,n)
    function E_delay_fb_array_function(E ,E_delay_fb = E_delay_fb_array,τ_fb_index = τ_fb_index_array)
        @inbounds for i in 1:n
            E_delay_fb[i] = E[Int.(τ_fb_index[i]),i] 
        end
            return E_delay_fb
    end
    
    E_delay_inj_matrix = zeros(ComplexF64, n, n)
    function E_delay_inj_matrix_function(E ,E_delay_inj = E_delay_inj_matrix,τ_inj = τ_inj_index_matrix, max = n)
        @inbounds for i in 1:max 
            @inbounds for j in 1:max
                idx = Int(τ_inj[i,j])   #
                if idx > 0
                    E_delay_inj[i,j] = E[idx,j]
                else
                    E_delay_inj[i,j] = 0 + 0im
                end
            end
        end
        return E_delay_inj
    end


    #Functions for the computation of the electric field

    #Optical gain
    function G_matrix(E_last_g_matrix, N_g_matrix,Out = OutG,
                        g_N = g_N_array,N_0 = N_0_array, s_cte = s) #he de añadir las constantes
        Out .= Diagonal(g_N .* (N_g_matrix .- N_0) ./ (1 .+ s_cte.*abs2.(E_last_g_matrix)))
        return Out
    end
    #Electric field
    function q_E_r_eq_array(E_last_q_E ,N_q_E ,E_array_q_E, Out = OutE,
                            G = G_matrix, T_ph_inverse = T_ph_inverse_matrix, K_fb = K_fb_matrix, 
                            E_delay_fb = E_delay_fb_array, K_inj = K_inj_matrix_result, E_delay_inj = E_delay_inj_matrix, 
                            α = α_matrix, E_delay_fb_function = E_delay_fb_array_function, E_delay_inj_function = E_delay_inj_matrix_function)
        Out .= α*(G(E_last_q_E,N_q_E) - T_ph_inverse)*E_last_q_E + K_fb*E_delay_fb_function(E_array_q_E,E_delay_fb) + sum(K_inj.*E_delay_inj_function(E_array_q_E), dims=2)
        return Out
    end

    #Carrier number
    function q_N_r_eq_array(E_last_q_N,N_q_N, Out = OutN,
                            I_r_divided_q = I_r_array_divided_q, T_s_inverse = T_s_inverse_matrix, G = G_matrix)
        Out .= I_r_divided_q - T_s_inverse*N_q_N - G(E_last_q_N, N_q_N)*abs2.(E_last_q_N)
        return Out
    end

    #Noise
    g = sqrt.(2 .* β_sp .* 1.1 .* N_0_array .* ξ_sp)

     
    

    k_E_array_1 = zeros(ComplexF64,n)
    k_N_array_1 = zeros(Float64,n)

    x_k_E_array = zeros(Int(round(τ_index) + 1),n) .+ 0im
    x_k_E_last = zeros(ComplexF64,n)
    x_k_N_array = zeros(Float64,n)
    
    k_E_array_2 = zeros(ComplexF64,n)
    k_N_array_2 = zeros(Float64,n)

    k_E_array_3 = zeros(ComplexF64,n)
    k_N_array_3 = zeros(Float64,n)

    k_E_array_4 = zeros(ComplexF64,n)
    k_N_array_4 = zeros(Float64,n)


    x_E = zeros(ComplexF64,n)
    x_N = zeros(Float64,n)

    h_half = 0.5*h
    largest_delay_steps = Int(round(τ_index) + 1)
    index_save = 1
    t = 0
    header_delay = 1 #this header points to the index in E_matrix with the largest delay, because all has been made referencing that value
    header_actual = largest_delay_steps #This header points to E_matrix(t). It always will be equal to h_delay -1
    #In header_delay will be substituted the value for E_matrix(t+1)

    E_last .= E_matrix[header_actual,:]
    #E_saved[1,:] = E_last

    u = zeros(n)
    
    @inbounds for i in 1:tf_steps

        t = RK4_method(x_E,x_N, h, sqr_h, q_E_r_eq_array, q_N_r_eq_array, g , 
    k_E_array_1,k_N_array_1,k_E_array_2,k_N_array_2,k_E_array_3,k_N_array_3,
    k_E_array_4,k_N_array_4, x_k_E_array, x_k_E_last, x_k_N_array, 
    E_matrix, E_last, N_array, n, header_actual,
    τ_inj_index_array,τ_fb_index_array, t, K_inj_matrix_result, K_inj_matrix, h_half,u)

        
        N_array .= x_N 
        E_last .= x_E
        E_matrix[header_actual,:] = x_E 


        if mod(i, Int(Δt))== 0 && i>=time_start_saving 
            E_saved[index_save,:] = x_E
            #N_saved[index_save,:] = x_N
            index_save = index_save + 1
        end

        #Here we actualize all headers and the matrix with the index for all dealys related to the coupling.
        τ_fb_index_array .+= 1
        τ_inj_index_array .+= 1
        τ_inj_index_matrix .+= 1
        header_actual += 1
        header_delay += 1
        
        τ_fb_index_array .= Int.(ifelse.(τ_fb_index_array .> largest_delay_steps, 1, τ_fb_index_array))
        τ_inj_index_array .= Int.(ifelse.(τ_inj_index_array .> largest_delay_steps, 1, τ_inj_index_array))
        τ_inj_index_matrix .= Int.(ifelse.(τ_inj_index_matrix .> largest_delay_steps, 1, τ_inj_index_matrix))
        header_actual = Int.(ifelse(header_actual > largest_delay_steps, 1, header_actual))
        header_delay = Int.(ifelse(header_delay > largest_delay_steps, 1, header_delay))
        

        
        τ_inj_index_matrix[diagind(τ_inj_index_matrix)] .=0
    end
        time_arr = collect(19600e-9:h*Δt:20000e-9)
    if length(E_saved[:,1]) != length(time_arr)
        push!(time_arr, tf)
    end

    x_saved = [E_saved,time_arr]
    return x_saved
end

@everywhere function running_simulations(Δf)
    f_1 = 299792458/1546.05e-9 + Δf*10^(9)
    f_2 = 299792458/1546.05e-9
    f_3 = 299792458/1546.05e-9
    f_4 = 299792458/1546.05e-9
    
    x_results = simulation_4_lasers(f_1, f_2, f_3, f_4,
                                    0,20000*10^(-9))


    df = DataFrame(E1 = x_results[1][:,1] ,E2 = x_results[1][:,2], E3 = x_results[1][:,3], E4= x_results[1][:,4], time = abs.(x_results[2]))
    CSV.write(joinpath(@__DIR__, "4_lasers_simulation_Δf_$(Δf)_RK4_no_noise_new_params_larger_wait.csv"), df)
end

begin
    Δf_list = collect(-30:0.1:30)
    pmap(running_simulations, Δf_list)
end




