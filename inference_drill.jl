using Rocket
using GraphPPL
using ReactiveMP
using AIDA
using Plots
using WAV
using JLD

function HA_drill_output(clean_path, sin_mag=0.05)

    n, fs = WAV.wavread("sound/speech/drill/drill.wav")
    s, fs = WAV.wavread("sound/speech/clean/sp01.wav")
    n = n[1:length(s)]
    x = s .+ n
    wavwrite(x, fs, "sound/speech/drill/sp01.wav")

    AR_order = 3
    inputs, outputs = ar_ssm(n, AR_order)
    γ, θ, fe = ar_inference(inputs, outputs, AR_order, 10, priors=Dict(:μθ => zeros(AR_order), :Λθ => diageye(AR_order), :aγ => 1.0, :bγ => 1.0))

    speech_seg = get_frames(x, fs)
    totseg = size(speech_seg)[1]

    speech_seg = get_frames(x, fs)
    totseg = size(speech_seg)[1]
    priors_mη, priors_vη, priors_τ = prior_to_priors(mean(θ), precision(θ), (shape(γ), rate(γ)), totseg, AR_order)
    priors_η = (priors_mη, priors_vη)

    println("Obtaining HA output")
    rmz, rvz, rmθ, rvθ, rγ, rmx, rvx, rmη, rvη, rτ, fe = batch_coupled_learning(speech_seg, priors_η, priors_τ, 10, AR_order, 30);

    s_ = get_signal(rmz, fs)
    n_ = get_signal(rmx, fs)

    WAV.wavwrite(x, fs, "sound/speech/drill/"*clean_path[findlast("/", clean_path)[1]+1:end-3]*"wav")

    JLD.save("sound/separated_jld/speech/drill/"*clean_path[findlast("/", clean_path)[1]+1:end-3]*"jld",
            "rmz", rmz, "rvz", rvz, "rmθ", rmθ, "rvθ", rvθ, "rγ", rγ, 
            "rmx", rmx, "rvx", rvx, "rmη", rmη, "rvη", rvη, "rτ", rτ,
            "fe", fe, "filename", clean_path)

    return x, s_, n_
end

x, s, n = HA_drill_output("sound/speech/clean/sp01.wav")
plot(x)
plot!(s)
wavwrite(s, "example.wav")
plot!(n)