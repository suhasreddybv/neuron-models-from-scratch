# Timestep convergence check for the Hodgkin-Huxley simulation.
# Re-runs the same step-current protocol with forward Euler at dt = 0.01 ms (the value
# used in hodgkin_huxley.py) and dt = 0.001 ms, and compares spike times.
import numpy as np

# HH parameters (identical to hodgkin_huxley.py)
C_m, g_Na, g_K, g_L = 1.0, 120.0, 36.0, 0.3
E_Na, E_K, E_L = 50.0, -77.0, -54.4

def alpha_m(V): return 0.1*(V+40.0)/(1.0-np.exp(-(V+40.0)/10.0))
def beta_m(V):  return 4.0*np.exp(-(V+65.0)/18.0)
def alpha_h(V): return 0.07*np.exp(-(V+65.0)/20.0)
def beta_h(V):  return 1.0/(1.0+np.exp(-(V+35.0)/10.0))
def alpha_n(V): return 0.01*(V+55.0)/(1.0-np.exp(-(V+55.0)/10.0))
def beta_n(V):  return 0.125*np.exp(-(V+65.0)/80.0)

def spike_times(dt, T=150.0, I_amp=10.0):
    time = np.arange(0, T, dt)
    I_ext = np.where((time >= 10.0) & (time <= 110.0), I_amp, 0.0)
    V = -65.0
    m = alpha_m(V)/(alpha_m(V)+beta_m(V))
    h = alpha_h(V)/(alpha_h(V)+beta_h(V))
    n = alpha_n(V)/(alpha_n(V)+beta_n(V))
    V_arr = np.empty(len(time))
    V_arr[0] = V
    for i in range(1, len(time)):
        I_ion = g_Na*m**3*h*(V-E_Na) + g_K*n**4*(V-E_K) + g_L*(V-E_L)
        dV = (I_ext[i] - I_ion)/C_m
        dm = alpha_m(V)*(1-m) - beta_m(V)*m
        dh = alpha_h(V)*(1-h) - beta_h(V)*h
        dn = alpha_n(V)*(1-n) - beta_n(V)*n
        V, m, h, n = V+dV*dt, m+dm*dt, h+dh*dt, n+dn*dt
        V_arr[i] = V
    # spike = local maximum above 0 mV
    peaks = np.where((V_arr[1:-1] > 0) & (V_arr[1:-1] > V_arr[:-2]) & (V_arr[1:-1] >= V_arr[2:]))[0] + 1
    return time[peaks]

coarse = spike_times(0.01)
fine = spike_times(0.001)
print(f"Spikes: dt=0.01 ms -> {len(coarse)}, dt=0.001 ms -> {len(fine)}")
n = min(len(coarse), len(fine))
print(f"Max spike-time difference: {np.abs(coarse[:n]-fine[:n]).max():.3f} ms")
