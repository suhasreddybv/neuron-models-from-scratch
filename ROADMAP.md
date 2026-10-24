# Roadmap

One model per day, in roughly increasing order of difficulty. An automated job picks the first unchecked item, implements it under `src/`, validates it, writes a figure to `figures/`, adds a README section, and ticks the box here in the same commit. Every model follows [CONVENTIONS.md](CONVENTIONS.md): NumPy from the equations, and at least one check against something other than its own plot.

Edit this file freely: reorder, remove, or add rows. The job always takes the first unchecked row. Lines starting with `>` under an item are notes to the implementer.

## Tier 1 — Single-neuron point models

- [x] Leaky integrate-and-fire (`src/leaky_integrate_and_fire.py`)
- [x] Hodgkin–Huxley, squid axon (`src/hodgkin_huxley.py`)
- [x] FitzHugh–Nagumo (`src/fitzhugh_nagumo.py`)
  > Validate: nullclines and fixed point from the equations; Hopf bifurcation current found by sweep agrees with the linearised Jacobian prediction.
- [x] Quadratic integrate-and-fire and the theta neuron (`src/quadratic_if_theta.py`)
  > Validate: f–I curve against the closed-form rate of the theta neuron; show the two models are equivalent under the change of variables.
- [x] Izhikevich (2003) simple model (`src/izhikevich.py`)
  > Validate: reproduce the regular-spiking, intrinsically-bursting, chattering and fast-spiking regimes with the published parameter sets; compare firing rates with the paper's figure.
- [x] Exponential integrate-and-fire (EIF) (`src/exponential_if.py`)
  > Validate: rheobase against the analytical value from the nullcline; spike-initiation sharpness vs ΔT.
- [x] Adaptive exponential integrate-and-fire (AdEx) (`src/adex.py`)
  > Validate: reproduce at least four firing patterns from Naud et al. (2008) with their parameter table; spike-frequency adaptation index.
- [x] Morris–Lecar (`src/morris_lecar.py`)
  > Validate: class I vs class II excitability from the two published parameter sets; onset frequency near 0 Hz (class I) vs finite (class II).
- [ ] Hindmarsh–Rose
  > Validate: bursting vs tonic regimes as a function of I; inter-burst interval vs slow-variable time constant r.
- [ ] Connor–Stevens (HH + A-type K current)
  > Validate: class I f–I curve (continuous from 0 Hz) vs HH class II; compare with Dayan & Abbott fig. 6.1.

## Tier 2 — Biophysics and numerics

- [ ] HH with a proper integrator: RK4 and `solve_ivp` vs forward Euler
  > Validate: order-of-convergence plot (error vs dt on log–log axes) showing slope ≈ 1 for Euler and ≈ 4 for RK4; this also closes the "Forward Euler only" limitation in the README.
- [ ] HH with temperature dependence (Q10)
  > Validate: spike width and rate vs temperature 6–20 °C; compare with the 1952 paper's statement that rates scale by Q10 ≈ 3.
- [ ] Stochastic ion channels: Markov-chain Na/K channels (Gillespie or binomial) and the Fox–Lu diffusion approximation
  > Validate: as channel number N → large, the stochastic model converges to deterministic HH; spontaneous firing rate vs N.
- [ ] Wang–Buzsáki fast-spiking interneuron
  > Validate: f–I curve reaches the paper's ~400 Hz; action potential width.
- [ ] Noisy LIF: Ornstein–Uhlenbeck input and the diffusion approximation
  > Validate: firing rate against the Siegert formula; CV of inter-spike intervals vs μ/σ.
- [ ] Passive cable: analytical steady-state attenuation and a compartmental solver
  > Validate: simulated steady-state voltage along a finite cable vs the closed-form cosh solution; electrotonic length.
- [ ] Two-compartment Pinsky–Rinzel CA3 pyramidal model
  > Validate: bursting vs regular firing as a function of the coupling conductance g_c, as in the 1994 paper.
- [ ] Multi-compartment Traub-style neuron with active dendrites
  > Validate: back-propagating action potential attenuation; compare with the two-compartment result.

## Tier 3 — Synapses and plasticity

- [ ] Conductance-based synapses: AMPA, NMDA (Mg block) and GABA-A as exponential / alpha / bi-exponential kinetics
  > Validate: NMDA I–V curve shows the Jahr–Stevens magnesium block; PSP shape vs analytical alpha function.
- [ ] Short-term plasticity: Tsodyks–Markram depression and facilitation
  > Validate: steady-state PSP amplitude vs frequency against the closed-form expression.
- [ ] Spike-timing-dependent plasticity (pair-based STDP)
  > Validate: reproduce the Bi & Poo (1998) window; weight distribution after Poisson input.
- [ ] Rate-based Hebbian learning: Oja's rule recovers the first principal component
  > Validate: learned weight vector vs the top eigenvector of the input covariance (cosine similarity > 0.99).
- [ ] Triplet STDP (Pfister & Gerstner) and frequency dependence
  > Validate: reproduce the frequency-dependent plasticity curve from Sjöström et al. (2001) qualitatively.

## Tier 4 — Networks and population models

- [ ] Wilson–Cowan excitatory–inhibitory population model
  > Validate: oscillation frequency vs linearised prediction at the Hopf point; phase plane with nullclines.
- [ ] Hopfield network
  > Validate: storage capacity ≈ 0.138 N; overlap vs number of stored patterns.
- [ ] Brunel (2000) sparsely connected E/I LIF network
  > Validate: reproduce the asynchronous-irregular and synchronous-regular regimes from the phase diagram; population rate vs mean-field prediction.
- [ ] Ring attractor (continuous attractor for head direction)
  > Validate: bump stability; drift speed vs asymmetric input matches integration of velocity.
- [ ] Balanced network (van Vreeswijk & Sompolinsky)
  > Validate: firing rates scale as predicted when input is increased; CV ≈ 1.
- [ ] Liquid state machine / LIF reservoir on a simple temporal task
  > Validate: readout accuracy above chance with a held-out set; compare with a shuffled-label control.
- [ ] Kuramoto oscillators with a neural phase-response curve
  > Validate: synchronisation transition at the analytical critical coupling K_c = 2/(πg(0)).

## Tier 5 — Models of cognition and data

- [ ] Drift-diffusion model of decision making
  > Validate: simulated accuracy and mean RT against the closed-form expressions for the pure DDM.
- [ ] Leaky competing accumulator (Usher–McClelland)
  > Validate: reduces to the DDM in the appropriate limit.
- [ ] Reduced Jansen–Rit neural mass model of EEG
  > Validate: alpha-band peak in the power spectrum for the default parameters; bifurcation vs input rate.
- [ ] Simple spiking model of a Parkinsonian basal-ganglia loop (STN–GPe beta oscillations, Terman et al. 2002)
  > Validate: beta-band oscillation emerges when striatal inhibition of GPe is raised; disappears at baseline.

## Renewal note

The GitHub token used by the daily job is a fine-grained PAT scoped to this repository. Check its expiry on github.com → Settings → Developer settings; the job will fail at `git push` once it lapses.
