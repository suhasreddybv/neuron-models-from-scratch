# neuron-models-from-scratch

Hodgkin–Huxley and leaky integrate-and-fire neurons implemented in NumPy from the original equations, with each checked against something other than its own plot.

**Result.** The LIF simulation matches the closed-form firing-rate curve to 1.7% mean relative error across 0–50 nA. The Hodgkin–Huxley spike train is converged in timestep: shrinking dt tenfold (0.01 → 0.001 ms) moves spike times by at most 0.019 ms.

![Hodgkin–Huxley response to a 10 µA/cm² step: membrane potential, gating variables, conductances, injected current](figures/hh_simulation.png)

## Method

**Hodgkin–Huxley** (`src/hodgkin_huxley.py`). A single-compartment model with Na⁺, K⁺ and leak currents. It uses the squid-axon parameters of Hodgkin & Huxley (1952), written in the modern convention with rest at −65 mV: C_m = 1 µF/cm², ḡ_Na = 120, ḡ_K = 36, g_L = 0.3 mS/cm², E_Na = 50, E_K = −77, E_L = −54.4 mV. The gating variables m, h and n follow first-order kinetics with voltage-dependent α/β rates and start at their steady-state values. Integration is forward Euler at dt = 0.01 ms. The stimulus is a 10 µA/cm² step from 10 to 110 ms, which gives 7 action potentials (~70 Hz) and a return to rest after the step ends.

**Leaky integrate-and-fire** (`src/leaky_integrate_and_fire.py`). τ_m dV/dt = −(V − E_L) + R_m·I_e with τ_m = 20 ms, E_L = V_reset = −65 mV, V_th = −50 mV, a 2 ms absolute refractory period, and forward Euler at dt = 0.1 ms. The script plots a voltage trace and inter-spike intervals at I_e = 20 nA. It then sweeps I_e from 0 to 50 nA and compares the simulated firing rate with the analytical rate r = 1 / (τ_ref + τ_m ln[(V_∞ − V_reset)/(V_∞ − V_th)]), where V_∞ = E_L + R_m·I_e.

![LIF voltage trace, inter-spike intervals, and simulated vs analytical f–I curve](figures/lif_simulation.png)

## Validation

| Check | Script | Outcome |
|---|---|---|
| LIF f–I curve vs closed form, 0–50 nA | `leaky_integrate_and_fire.py` | Rheobase at 15 nA in both; mean relative error 1.7% on firing points, max 2.45 Hz |
| HH timestep convergence, dt 0.01 vs 0.001 ms | `check_hh_timestep.py` | Same spike count (7); max spike-time shift 0.019 ms |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/hodgkin_huxley.py            # writes figures/hh_simulation.png
python src/leaky_integrate_and_fire.py  # writes figures/lif_simulation.png
python src/check_hh_timestep.py         # prints convergence check
```

Tested with Python 3.9, NumPy 1.26 and Matplotlib 3.9. No external data is needed.

## Limitations and failure cases

- **Forward Euler only.** Convergence is shown for one HH stimulus protocol, not for all inputs. There is no comparison against an adaptive or higher-order solver (e.g. `scipy.integrate.solve_ivp`). A stiffer protocol or a larger dt could diverge without warning.
- **HH is not compared quantitatively with published traces.** Spike shape, the ~+40 mV first peak and the ~70 Hz rate look qualitatively right for these parameters. No curve from the 1952 paper or a reference simulator (NEURON, Brian2) has been overlaid.
- **Singular rate functions.** α_m and α_n divide by zero at exactly V = −40 mV and −55 mV. No guard exists, despite the comment in the code. Floating-point trajectories do not hit these values in practice, but a clamped-voltage experiment at those holding potentials would return NaN.
- **LIF rate quantisation.** The simulated rate is a spike count over 500 ms, so it is only resolved to 2 Hz, and that accounts for most of the 1.7% gap. The +20 mV "spike" is drawn for display and is not part of the model.
- **Scope.** These are single-compartment, deterministic, temperature-independent models: no noise, dendrites, synapses or networks. The scripts are procedural and have no unit tests.

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117(4), 500–544.
- Dayan, P., & Abbott, L. F. (2001). *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems*, ch. 5. MIT Press.

## Licence

MIT
