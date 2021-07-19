using Test
using QuantumOptics
using LinearAlgebra

@testset "timecorrelations" begin

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

fockbasis = FockBasis(10)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
H = Ha + Hc + Hint

Ja = embed(basis, 1, sqrt(γ)*sm)
Ja2 = embed(basis, 1, sqrt(0.5*γ)*sp)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Ja2, Jc]

Ψ₀ = basisstate(spinbasis, 2) ⊗ fockstate(fockbasis, 5)
ρ₀ = dm(Ψ₀)

tspan = [0.:10:100.;]

op = embed(basis, 1, sqrt(γ)*sz)
exp_values = timecorrelations.correlation(tspan, ρ₀, H, J, dagger(op), op)

ρ₀ = dm(Ψ₀)

tout, exp_values2 = timecorrelations.correlation(ρ₀, H, J, dagger(op), op; tol=1e-5)

@test length(exp_values) == length(tspan)
@test length(exp_values2) == length(tout)
@test norm(exp_values[1]-exp_values2[1]) < 1e-15
@test norm(exp_values[end]-exp_values2[end]) < 1e-4

n = length(tspan)
omega_sample = mod(n, 2) == 0 ? [-n/2:n/2-1;] : [-(n-1)/2:(n-1)/2;]
omega_sample .*= 2pi/tspan[end]
omega, S = timecorrelations.spectrum(omega_sample, H, J, op; rho_ss=ρ₀)

omega2, S2 = timecorrelations.spectrum(H, J, op; tol=1e-3)
@test length(omega2) == length(S2)

omegaFFT, SFFT = timecorrelations.correlation2spectrum(tspan, exp_values)

@test omega_sample == omegaFFT && S == SFFT

@test_throws ArgumentError timecorrelations.correlation2spectrum(tout, exp_values)

tspan[5] = tspan[4]
@test_throws ArgumentError timecorrelations.correlation2spectrum(tspan, exp_values)

# Test iterative spectrum
bfock = FockBasis(10)
η = 1.5
κ = 1.0

a = destroy(bfock)
H = η*(a + a')
J = [sqrt(2κ)*a]

ρs = steadystate.iterative(H, J)
@test isapprox(expect(a'*a,ρs), η^2 / κ^2; atol=1e-2)

ω = range(-15, 15, length=200)
S = timecorrelations.spectrum_iterative(ω, ρs, a, a', H, J)

fwhm_idx = findmin(abs.(S .- 0.5*maximum(S)))[2]
hwhm = abs(ω[fwhm_idx])
@test abs(κ - hwhm) < 0.3

# TODO: better tests and test lazy operators

end # testset
