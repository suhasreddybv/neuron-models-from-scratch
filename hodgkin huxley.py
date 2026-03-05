# Hodgkin-Huxley model of a neuron
#Block 1: Imports and Parameteres
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.01 #time step in ms
T = 150.0 #total simulation time in ms
time = np.arange(0,T,dt) #time array

# HH model parameters 
# Conductance in mS/cm^2, voltages in mV, capacitance in uF/cm^2
C_m = 1.0 #membrane capacitance
g_Na = 120.0 # maximum sodium conductance
g_K = 36.0 # maximum potassium conductance
g_L = 0.3 # leak conductance


E_Na = 50.0 # reversal potential for sodium
E_K = -77.0 # reversal potential for potassium
E_L = -54.4 # reversal potential for leak


# Block 2: Alpha and Beta functions
# gating variable rate functions
def alpha_m(V):
    #avoid multiplication by zero when V = -40.0
    return 0.1*(V+40.0)/(1.0-np.exp(-(V+40.0)/10.0))

def beta_m(V):
    return 4.0*np.exp(-(V+65.0)/18.0)

def alpha_h(V):
    return 0.07*np.exp(-(V+65.0)/20.0)

def beta_h(V):
    return 1.0/(1.0+np.exp(-(V+35.0)/10.0))

def alpha_n(V):
    return 0.01*(V+55.0)/(1.0-np.exp(-(V+55.0)/10.0))

def beta_n(V):
    return 0.125*np.exp(-(V+65.0)/80.0)

#Block 3: Initial conditions at rest (V=-65.0mV0)
# Initial conditions
V_0 = -65.0 # initial membrane potential in mV
m_0 = alpha_m(V_0)/(alpha_m(V_0)+beta_m(V_0)) # ~= 0.053
h_0 = alpha_h(V_0)/(alpha_h(V_0)+beta_h(V_0)) # ~= 0.60
n_0 = alpha_n(V_0)/(alpha_n(V_0)+beta_n(V_0)) # ~= 0.32

print(f"Initial conditions: V = {V_0} mV, m_0 = {m_0:.3f}, h_0 = {h_0:.3f}, n_0 = {n_0:.3f}")

#Block 4: Defining injected current
# Injected current Ie drives the neuron. We will use a step current which is zero for 10ms and then a constant positive value for 100 ms, then zero again.
#Injected current(step pulse)

I_ext = np.zeros(len(time))
I_ext[(time>=10.0)&(time<=110.0)]=10.0 #10mA/cm^2

#The value of 10.0mA/cm^2 is above the threshold for action potential generation.

# Block 5: The main simulation loop
#Storage arrays
V_arr = np.zeros(len(time))
m_arr = np.zeros(len(time))
h_arr = np.zeros(len(time))
n_arr = np.zeros(len(time))

# Initial conditions
V_arr[0] = V_0
m_arr[0] = m_0
h_arr[0] = h_0
n_arr[0] = n_0

# Euler integration loop
for i in range(1, len(time)):
    # Pull current state
    V = V_arr[i-1]
    m = m_arr[i-1]
    h = h_arr[i-1]
    n = n_arr[i-1]

    # Compute conductances from gating variables
    g_Na_now = g_Na * m**3 * h
    g_K_now = g_K * n**4

    # Ionic currents (positive = outward)
    I_Na = g_Na_now * (V - E_Na)
    I_K = g_K_now * (V - E_K)
    I_L = g_L * (V - E_L)

    # Rates of change (ODEs)
    dVdt = (I_ext[i] - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V)*(1 - m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1 - h) - beta_h(V)*h
    dndt = alpha_n(V)*(1 - n) - beta_n(V)*n

    # Euler step — advance all variables
    V_arr[i] = V + dVdt*dt
    m_arr[i] = m + dmdt*dt
    h_arr[i] = h + dhdt*dt
    n_arr[i] = n + dndt*dt

#Block 6: Plotting all the results
fig, axes = plt.subplots(4,1,figsize=(12,10),sharex=True)

#panel1: Membrane voltage
axes[0].plot(time, V_arr, color='black',lw=1.5)
axes[0].set_ylabel('Voltage (mV)')
axes[0].set_title('Hodgkin-Huxley Model - action potential')
axes[0].axhline(E_Na,color='blue',ls='--',lw=0.8,label='ENa')
axes[0].axhline(E_K,color='red',ls='--',lw=0.8,label='EK')
axes[0].axhline(E_L,color='green',ls='--',lw=0.8,label='EL')
axes[0].legend(loc='upper right',fontsize=8)

#panel2: Gating variables
axes[1].plot(time, m_arr, color='blue',lw=1.5, label='m(Na activation)')
axes[1].plot(time, h_arr, color='red',lw=1.5, label='h(Na inactivation)')
axes[1].plot(time, n_arr, color='green',lw=1.5, label='n(K activation)')
axes[1].set_ylabel('Gating Variables')
axes[1].set_ylim(-0.05,1.05)
axes[1].legend(loc='upper right',fontsize=8)


#panel3:Conductances
axes[2].plot(time,g_Na * m_arr**3 * h_arr, color='blue',lw=1.5, label='gNa')
axes[2].plot(time,g_K * n_arr**4, color='red',lw=1.5, label='gK')
axes[2].set_ylabel('Conductances (mS/cm^2)')
axes[2].legend(loc='upper right',fontsize=8)

# Panel 4: Injected current
axes[3].plot(time, I_ext, color='purple', lw=1.5, label='I_inj')
axes[3].set_ylabel('Injected current (µA/cm²)')
axes[3].set_xlabel('Time (ms)')
axes[3].legend(loc='upper right',fontsize=8)

plt.tight_layout()
plt.savefig('HH simulation.png',dpi=150)
plt.show()
