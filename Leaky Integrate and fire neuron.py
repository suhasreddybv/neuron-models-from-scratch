# Block 1: Imports and Parameteres

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1 #time step in ms - note time step is 10 times larger than in the HH model
T = 200.0 #total simulation time in ms
time = np.arange(0,T,dt)

# LIF parameteres
tau_m = 20.0 #membrane time constant in ms
R_m = 1.0 # membrane resistance in GΩ with current in nA and voltage in mV
E_L = -65.0 # resting potential in mV
V_thresh = -50.0 #spike threshold in mV
V_reset = -65.0 #reset potential in mV
tau_ref = 2.0 #refractory period in ms

#Derived
t_ref_steps = int(tau_ref/dt) #number of time steps in the refractory period


## Block 2: Single neuron simulation function
def simulate_lif(I_e,T=200.0,dt=0.1):
    """
    Simulate a single LIF neuron with injected current I_ext
    returns: time array, voltage trace,spike times
    """
    time = np.arange(0,T,dt)
    V=np.zeros(len(time))
    V[0]=E_L #start at rest
    spikes = [] #list to store spike times
    refractory_counter = 0 

    for i in range(1,len(time)):
        # If in refractory period, hold at reset potential
        if refractory_counter>0:
            V[i]=V_reset
            refractory_counter -=1
            continue

        #LIF update - Euler step
        dVdt = (-(V[i-1]-E_L)+R_m*I_e)/tau_m
        V[i]=V[i-1]+dVdt*dt

        #Check for spike
        if V[i]>=V_thresh:
            V[i]=20.0 #Cosmetic spike - reset to a high value
            spikes.append(time[i])
            refractory_counter = t_ref_steps

    return time,V,spikes


## Block 3: run single trace (data for plotting)
I_e = 20.0  # nA - above threshold
time_arr, V_arr, spike_times = simulate_lif(I_e)
print(f"Number of spikes: {len(spike_times)}")
print(f"Mean firing rate: {len(spike_times)/(T/1000):.1f} Hz")


## Block 4: f-I curve data
def fI_analytical(I_e):
    V_inf = E_L + R_m*I_e
    if V_inf <= V_thresh:
        return 0.0
    ISI = tau_ref + tau_m * np.log(
        (V_inf - V_reset)/(V_inf - V_thresh)
    )
    return 1000.0/ISI  # convert to Hz

I_range = np.linspace(0, 50, 100)  # range of injected currents in nA
firing_rates_sim = []
for I in I_range:
    _, _, spikes = simulate_lif(I, T=500.0)
    rate = len(spikes) / (500.0/1000.0)
    firing_rates_sim.append(rate)
firing_rates_theory = [fI_analytical(I) for I in I_range]


## Single window: all plots + save as one image
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

# Panel 1: Voltage trace
axes[0].plot(time_arr, V_arr, color='black', lw=1.2)
axes[0].axhline(V_thresh, color='red', ls='--', lw=1, label='Threshold')
axes[0].axhline(E_L, color='blue', ls='--', lw=1, label='Rest')
axes[0].set_ylabel('Voltage (mV)')
axes[0].set_title(f'LIF Neuron - Ie = {I_e} nA')
axes[0].legend()
for st in spike_times:
    axes[0].axvline(st, color='orange', alpha=0.3, lw=0.8)

# Panel 2: ISI analysis
if len(spike_times) > 1:
    ISIs = np.diff(spike_times)
    axes[1].bar(range(len(ISIs)), ISIs, color='steelblue', alpha=0.7)
    axes[1].axhline(np.mean(ISIs), color='red', ls='--', lw=1, label=f'Mean ISI = {np.mean(ISIs):.1f} ms')
axes[1].set_ylabel('ISI (ms)')
axes[1].set_xlabel('Spike number')
axes[1].legend()

# Panel 3: f-I curve
axes[2].plot(I_range, firing_rates_sim, 'o', ms=4, color='steelblue', label='Simulated', alpha=0.8)
axes[2].plot(I_range, firing_rates_theory, '-', color='red', lw=2, label='Analytical')
axes[2].set_xlabel('Injected Current (nA)')
axes[2].set_ylabel('Firing Rate (Hz)')
axes[2].set_title('LIF Neuron: Simulation vs Analytical f-I curve')
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_xlim(I_range.min(), I_range.max())
axes[2].set_ylim(0, max(max(firing_rates_sim), max(firing_rates_theory)) * 1.1)

plt.tight_layout()
fig.savefig('LIF_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

