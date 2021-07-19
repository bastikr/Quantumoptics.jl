module timecorrelations

export correlation, spectrum, correlation2spectrum

using QuantumOpticsBase
using ..timeevolution, ..steadystate

using FFTW
using IterativeSolvers
using LinearMaps


"""
    timecorrelations.correlation([tspan, ]rho0, H, J, op1, op2; <keyword arguments>)

Calculate two time correlation values ``⟨A(t)B(0)⟩``.

The calculation is done by multiplying the initial density operator
with ``B`` performing a time evolution according to a master equation
and then calculating the expectation value ``\\mathrm{Tr} \\{A ρ\\}``

Without the `tspan` argument the points in time are chosen automatically from
the ode solver and the final time is determined by the steady state termination
criterion specified in [`steadystate.master`](@ref).

# Arguments
* `tspan`: Points of time at which the correlation should be calculated.
* `rho0`: Initial density operator.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector of jump operators.
* `op1`: Operator at time `t`.
* `op2`: Operator at time `t=0`.
* `rates=ones(N)`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function correlation(tspan, rho0::Operator, H::AbstractOperator, J,
                     op1, op2;
                     rates=nothing,
                     Jdagger=dagger.(J),
                     kwargs...)
    function fout(t, rho)
        expect(op1, rho)
    end
    t,u = timeevolution.master(tspan, op2*rho0, H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    u
end

function correlation(rho0::Operator, H::AbstractOperator, J,
                     op1, op2;
                     tol=1e-4,
                     rates=nothing,
                     Jdagger=dagger.(J),
                     kwargs...)
    op2rho0 = op2*rho0
    exp1 = expect(op1, op2rho0)
    function fout(t, rho)
        expect(op1, rho)
    end
    t,u = steadystate.master(H, J; rho0=op2rho0, tol=tol, fout=fout,
                       rates=rates, Jdagger=Jdagger, save_everystep=true,kwargs...)
end


"""
    timecorrelations.spectrum([omega_samplepoints,] H, J, op; <keyword arguments>)

Calculate spectrum as Fourier transform of a correlation function

This is done with the Wiener-Khinchin theorem

```math
S(ω, t) = 2\\Re\\left\\{\\int_0^{∞} dτ e^{-iωτ}⟨A^†(t+τ)A(t)⟩\\right\\}
```

The argument `omega_samplepoints` gives the list of frequencies where ``S(ω)``
is caclulated. A corresponding list of times is calculated internally by means
of a inverse discrete frequency fourier transform. If not given, the
steady-state is computed before calculating the auto-correlation function.

Without the `omega_samplepoints` arguments the frequencies are chosen
automatically.

# Arguments
* `omega_samplepoints`: List of frequency points at which the spectrum
        is calculated.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector of jump operators.
* `op`: Operator for which the auto-correlation function is calculated.
* `rho0`: Initial density operator.
* `tol=1e-4`: Tracedistance used as termination criterion.
* `rates=ones(N)`: Vector or matrix specifying the coefficients for the
        jump operators.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the
        jump operators. If they are not given they are calculated automatically.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function spectrum(omega_samplepoints,
                H::AbstractOperator, J, op;
                rho0=tensor(basisstate(H.basis_l, 1), dagger(basisstate(H.basis_r, 1))),
                tol=1e-4,
                rho_ss=steadystate.master(H, J; tol=tol, rho0=rho0)[end][end],
                kwargs...)
    domega = minimum(diff(omega_samplepoints))
    dt = 2*pi/abs(omega_samplepoints[end] - omega_samplepoints[1])
    T = 2*pi/domega
    tspan = [0.:dt:T;]
    exp_values = correlation(tspan, rho_ss, H, J, dagger(op), op, kwargs...)
    S = 2dt.*fftshift(real(fft(exp_values)))
    return omega_samplepoints, S
end

function spectrum(H::AbstractOperator, J, op;
                rho0=tensor(basisstate(H.basis_l, 1), dagger(basisstate(H.basis_r, 1))),
                tol=1e-4,
                rho_ss=steadystate.master(H, J; tol=tol)[end][end],
                kwargs...)
    tspan, exp_values = correlation(rho_ss, H, J, dagger(op), op, tol=tol, kwargs...)
    dtmin = minimum(diff(tspan))
    T = tspan[end] - tspan[1]
    tspan = Float64[0.:dtmin:T;]
    n = length(tspan)
    omega = mod(n, 2) == 0 ? [-n/2:n/2-1;] : [-(n-1)/2:(n-1)/2;]
    omega .*= 2pi/T
    return spectrum(omega, H, J, op; tol=tol, rho_ss=rho_ss, kwargs...)
end


"""
    timecorrelations.correlation2spectrum(tspan, corr; normalize_spec)

Calculate spectrum as Fourier transform of a correlation function with a given correlation function.

# Arguments
* `tspan`: List of time points corresponding to the correlation function.
* `corr`: Correlation function of which the Fourier transform is to be calculated.
* `normalize_spec`: Specify if spectrum should be normalized to its maximum.
"""
function correlation2spectrum(tspan, corr; normalize_spec=false)
  n = length(tspan)
  if length(corr) != n
    ArgumentError("tspan and corr must be of same length!")
  end

  dt = tspan[2] - tspan[1]
  for i=2:length(tspan)-1
    if !isapprox(tspan[i+1] - tspan[i], dt)
      throw(ArgumentError("tspan must be equidistant!"))
    end
  end

  tmax = tspan[end] - tspan[1]
  omega = mod(n, 2) == 0 ? [-n/2:n/2-1;] : [-(n-1)/2:(n-1)/2;]
  omega .*= 2pi/tmax
  spec = 2dt.*fftshift(real(fft(corr)))

  omega, normalize_spec ? spec./maximum(spec) : spec
end

"""
    timecorrelations.spectrum_iterative(omega, rho, op1, op2, H, J,
                        method! = IterativeSolvers.bicgstabl!, args...;
                        Jdagger=dagger.(J), rates=nothing, kwargs...)

Iteratively solve the equation for the spectrum in Fourier space.
"""
function spectrum_iterative(omega, rho, op1, op2, H, J,
                    method! = IterativeSolvers.bicgstabl!, args...;
                    Jdagger=dagger.(J), rates=nothing, kwargs...)


    bl = rho.basis_l
    br = rho.basis_r
    M = length(bl)

    # Cache stuff
    rho_cache = copy(rho)
    drho = copy(rho)
    Jrho_cache = copy(rho)

    # Check reducibility
    isreducible = timeevolution.check_master(rho,H,J,Jdagger,rates)
    if isreducible
        Hnh = timeevolution.nh_hamiltonian(H,J,Jdagger,rates)
        Hnhdagger = dagger(Hnh)
        dmaster_ = (drho,rho) -> timeevolution.dmaster_nh!(drho, Hnh, Hnhdagger, J, Jdagger, rates, rho, Jrho_cache)
    else
        dmaster_ = (drho,rho) -> timeevolution.dmaster_h!(drho, H, J, Jdagger, rates, rho, Jrho_cache)
    end

    i = 1
    # Linear mapping
    function f!(y,x)
        # Reshape
        drho.data .= @views reshape(y, M, M)
        rho_cache.data .= @views reshape(x, M, M)
        # Apply function
        dmaster_(drho,rho_cache)
        # Recast data
        @views y .= reshape(drho.data, M^2)
        @. y = im*omega[i]*x - y
        return y
    end

    lm = LinearMaps.LinearMap{eltype(rho)}(f!,M^2;ismutating=true,issymmetric=false,ishermitian=false,isposdef=false)

    R0 = op1*rho
    x0 = @views reshape(R0.data, M^2)
    b = reshape(copy(R0.data), M^2)


    spec = similar(omega)
    while i <= length(omega)
        method!(x0,lm,b,args...;kwargs...)
        spec[i] = 2*real(expect(op2,R0))
        i += 1
    end
    return spec
end


end # module
