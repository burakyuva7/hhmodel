"""
Script for simulating cell action potentials via Hodgkin-Huxley (HH) Model.

Question 1:
```
python main.py --pulse_duration=1.0
```

Question 2:
```
python main.py --pulse_duration=1.5
python main.py --pulse_duration=2.0
python main.py --pulse_duration=3.0
```

Question 3:
```
python main.py --pulse_duration=1.0 --g_ACH_control_init=0.108 --search_increment 1.0 --tmax 25.0
```
"""
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Script for simulating cell action potentials via Hodgkin-Huxley (HH) model.')

# Arguments representing parameters for the simulation
parser.add_argument('--E_K', default=-82.0, type=float, help='')
parser.add_argument('--E_Na', default=45.0, type=float, help='')
parser.add_argument('--E_leak', default=-59.4001, type=float, help='')
parser.add_argument('--V_rest', default=-70.0, type=float, help='')
parser.add_argument('--g_bar_K', default=36.0, type=float, help='')
parser.add_argument('--g_bar_Na', default=120.0, type=float, help='')
parser.add_argument('--g_leak', default=0.3, type=float, help='')
parser.add_argument('--C_mem', default=1.0, type=float, help='')
parser.add_argument('--V_threshold', default=-55.0, type=float, help='')
parser.add_argument('--search_increment', default=0.001, type=float, help='')
parser.add_argument('--g_ACH_control_init', default=0.0, type=float, help='')
parser.add_argument('--pulse_duration', default=1.0, type=float, help='')
parser.add_argument('--tmax', default=20.0, type=float, help='')
args = parser.parse_args()

# Define equations 3-11 as given in the assignment
def alpha_m(v):
	numerator = 0.1 * (25 - v)
	denominator = math.exp(0.1 * (25 - v)) - 1
	return numerator / denominator

def beta_m(v):
	return 4 * math.exp(-v / 18)

def alpha_n(v):
	numerator = 0.01 * (10 - v)
	denominator = math.exp(0.1 * (10 - v)) - 1
	return numerator / denominator

def beta_n(v):
	return 0.125 * math.exp(-v / 80)

def alpha_h(v):
	return 0.07 * math.exp(-v / 20)

def beta_h(v):
	numerator = 1
	denominator = math.exp(0.1 * (30 - v)) + 1
	return numerator / denominator

def tau_m(v):
	numerator = 1
	denominator = alpha_m(v) + beta_m(v)
	return numerator / denominator

def m_inf(v):
	numerator = alpha_m(v)
	denominator = alpha_m(v) + beta_m(v)
	return numerator / denominator

def tau_n(v):
	numerator = 1
	denominator = alpha_n(v) + beta_n(v)
	return numerator / denominator

def n_inf(v):
	numerator = alpha_n(v)
	denominator = alpha_n(v) + beta_n(v)
	return numerator / denominator

def tau_h(v):
	numerator = 1
	denominator = alpha_h(v) + beta_h(v)
	return numerator / denominator

def h_inf(v):
	numerator = alpha_h(v)
	denominator = alpha_h(v) + beta_h(v)
	return numerator / denominator


# Times are given in units of ms
tstep = 0.001  # 1 nanosecond
tmin = 0.0 
tmax = args.tmax  # max duration as specified in the assignment

# Duration of simulation
ts = np.arange(tmin, tmax, tstep)

# Control variable for finding the g_ACH values
g_ACH_control = args.g_ACH_control_init
g_ACH = np.zeros(len(ts) + 1)
g_ACH[0] = g_ACH_control

# Create NumPy arrays for storing and plotting all variables
# NOTE: We also set the initial values for the variables
V_mem = np.zeros(len(ts) + 1)
V_mem[0] = args.V_rest
v = np.zeros(len(ts) + 1)

m = np.zeros(len(ts) + 1)
m[0] = m_inf(v[0])
n = np.zeros(len(ts) + 1)
n[0] = n_inf(v[0])
h = np.zeros(len(ts) + 1)
h[0] = h_inf(v[0])

i_mem = np.zeros(len(ts) + 1)
i_C = np.zeros(len(ts) + 1)
i_ACH = np.zeros(len(ts) + 1)
i_ACH[0] = g_ACH[0] * (V_mem[0] - args.E_K) + g_ACH[0] * (V_mem[0] - args.E_Na)
i_leak = np.zeros(len(ts) + 1)
i_leak[0] = args.g_leak * (V_mem[0] - args.E_leak)
i_K = np.zeros(len(ts) + 1)
i_K[0] = args.g_bar_K * (n[0] ** 4) * (V_mem[0] - args.E_K)
i_Na = np.zeros(len(ts) + 1)
i_Na[0] = args.g_bar_Na * (m[0] ** 3) * (V_mem[0] - args.E_Na)


# If g_ACH init is 0.0, we are searching for a value --> Questions 1 and 2
if g_ACH_control == 0.0:
	# Boolean flag to control the simulation
	action_potential = False

	# Keep running simulations until action potential is reached
	while not action_potential:
		g_ACH_control += args.search_increment
		print('g_ACH_control: %0.3f' % g_ACH_control)

		# Start the simulation for the current value of g_ach
		for i, t in enumerate(tqdm(ts, desc='Running Hodgkin-Huxley (HH) Simulation')):
			# Control ACH switch closing and opening depending on specified pulse duration
			if 1.0 <= t < 1.0 + args.pulse_duration:
				g_ACH[i + 1] = g_ACH_control
			else:
				g_ACH[i + 1] = 0.0

			# Update Hodgkin-Huxley simulation parameters
			m[i + 1] = m[i] + tstep * ((m_inf(v[i]) - m[i]) / tau_m(v[i]))
			n[i + 1] = n[i] + tstep * ((n_inf(v[i]) - n[i]) / tau_n(v[i]))
			h[i + 1] = h[i] + tstep * ((h_inf(v[i]) - h[i]) / tau_h(v[i]))

			# Update the membrane potentials
			V_mem[i + 1] = V_mem[i] + tstep * (i_mem[i] / args.C_mem)
			v[i + 1] = V_mem[i + 1] - args.V_rest

			# Compute current components in the circuit
			i_C[i + 1] = args.C_mem * tstep * (V_mem[i + 1] - V_mem[i])
			i_ACH[i + 1] = g_ACH[i + 1] * (V_mem[i + 1] - args.E_K) + g_ACH[i + 1] * (V_mem[i + 1] - args.E_Na)
			i_leak[i + 1] = args.g_leak * (V_mem[i + 1] - args.E_leak)
			i_K[i + 1] = args.g_bar_K * (n[i + 1] ** 4) * (V_mem[i + 1] - args.E_K)
			i_Na[i + 1] = args.g_bar_Na * ((m[i + 1] ** 3) * h[i + 1]) * (V_mem[i + 1] - args.E_Na)

			# Compute total current in membrane
			i_mem[i + 1] = i_C[i + 1] - i_ACH[i + 1] - i_leak[i + 1] - i_K[i + 1] - i_Na[i + 1]

			# Check if the membrane potential reaches the threshold, if so exit the simulation
			if V_mem[i + 1] > args.V_threshold:
				action_potential = True

	print('Minimum g_ACH: %0.3f' % g_ACH_control)

	plt.grid()
	plt.plot(ts, i_mem[1:])
	plt.title('Membrane Current vs. Time')
	plt.ylabel('I (mA)')
	plt.xlabel('Time (ms)')
	plt.xticks(np.arange(tmin, tmax + 1.0, 1.0))
	plt.savefig("I-vs-t_ps=%0.1f.png" % args.pulse_duration)
	plt.clf()

	plt.grid()
	plt.plot(ts, V_mem[1:])
	plt.title('Membrane Potential vs. Time')
	plt.ylabel(r'$V_{mem}$ (mV)')
	plt.xlabel('Time (ms)')
	plt.xticks(np.arange(tmin, tmax + 1.0, 1.0))
	plt.savefig("V-vs-t_ps=%0.1f.png" % args.pulse_duration)
	plt.clf()

	plt.grid()
	plt.plot(ts, g_ACH[1:])
	plt.axhline(g_ACH_control, linestyle='--', color='red', label=r'$g_{ACH, min} = %0.3f$' % g_ACH_control)
	plt.title('ACH Resistance vs. Time')
	plt.ylabel(r'$g_{ACH}$')
	plt.xlabel('Time (ms)')
	plt.xticks(np.arange(tmin, tmax + 1.0, 1.0))
	plt.legend()
	plt.savefig("gACH-vs-t_ps=%0.1f.png" % args.pulse_duration)
	plt.clf()

# If g_ACH init is not 0.0, we are just running simulations for one value of g_ACH --> Question 3
else:
	# Boolean flag to control the simulation
	action_potential = False
	action_potential_count = 0

	# Start searches for a new pulse from 2.0 ms
	t_new = 2.0

	# Keep running simulations until two action potentials are reached
	while action_potential_count != 2:
		# Reset the action potential flag and counter
		action_potential = False
		action_potential_count = 0

		t_new += args.search_increment
		print('t_new: %0.3f' % t_new)

		# Start the simulation for the current value of g_ach
		for i, t in enumerate(tqdm(ts, desc='Running Hodgkin-Huxley (HH) Simulation')):
			# Control ACH switch closing and opening depending on specified pulse duration
			if 1.0 <= t < 1.0 + args.pulse_duration:
				g_ACH[i + 1] = g_ACH_control
			# The below control is for the second (new) pulse
			elif t_new <= t < t_new + args.pulse_duration:
				g_ACH[i + 1] = g_ACH_control
			else:
				g_ACH[i + 1] = 0.0

			# Update Hodgkin-Huxley simulation parameters
			m[i + 1] = m[i] + tstep * ((m_inf(v[i]) - m[i]) / tau_m(v[i]))
			n[i + 1] = n[i] + tstep * ((n_inf(v[i]) - n[i]) / tau_n(v[i]))
			h[i + 1] = h[i] + tstep * ((h_inf(v[i]) - h[i]) / tau_h(v[i]))

			# Update the membrane potentials
			V_mem[i + 1] = V_mem[i] + tstep * (i_mem[i] / args.C_mem)
			v[i + 1] = V_mem[i + 1] - args.V_rest

			# Compute current components in the circuit
			i_C[i + 1] = args.C_mem * tstep * (V_mem[i + 1] - V_mem[i])
			i_ACH[i + 1] = g_ACH[i + 1] * (V_mem[i + 1] - args.E_K) + g_ACH[i + 1] * (V_mem[i + 1] - args.E_Na)
			i_leak[i + 1] = args.g_leak * (V_mem[i + 1] - args.E_leak)
			i_K[i + 1] = args.g_bar_K * (n[i + 1] ** 4) * (V_mem[i + 1] - args.E_K)
			i_Na[i + 1] = args.g_bar_Na * ((m[i + 1] ** 3) * h[i + 1]) * (V_mem[i + 1] - args.E_Na)

			# Compute total current in membrane
			i_mem[i + 1] = i_C[i + 1] - i_ACH[i + 1] - i_leak[i + 1] - i_K[i + 1] - i_Na[i + 1]

			# Check if the membrane potential reaches the threshold, if so exit the simulation
			if not action_potential and V_mem[i + 1] > args.V_threshold:
				action_potential = True
				action_potential_count += 1
			elif action_potential and V_mem[i + 1] < args.V_threshold:
				action_potential = False

	print('Second pulse at: ', t_new)
	print('Set g_ACH: %0.3f' % g_ACH_control)

	plt.grid()
	plt.plot(ts, i_mem[1:])
	plt.title('Membrane Current vs. Time')
	plt.ylabel('I (mA)')
	plt.xlabel('Time (ms)')
	plt.xticks(np.arange(tmin, tmax + 1.0, 1.0))
	plt.savefig("Q3-I-vs-t_ps=%0.1f.png" % args.pulse_duration)
	plt.clf()

	plt.grid()
	plt.plot(ts, V_mem[1:])
	plt.title('Membrane Potential vs. Time')
	plt.ylabel(r'$V_{mem}$ (mV)')
	plt.xlabel('Time (ms)')
	plt.xticks(np.arange(tmin, tmax + 1.0, 1.0))
	plt.savefig("Q3-V-vs-t_ps=%0.1f.png" % args.pulse_duration)
	plt.clf()

	plt.grid()
	plt.plot(ts, g_ACH[1:])
	plt.axhline(g_ACH_control, linestyle='--', color='red', label=r'$g_{ACH, min} = %0.3f$' % g_ACH_control)
	plt.title('ACH Resistance vs. Time')
	plt.ylabel(r'$g_{ACH}$')
	plt.xlabel('Time (ms)')
	plt.xticks(np.arange(tmin, tmax + 1.0, 1.0))
	plt.legend()
	plt.savefig("Q3-gACH-vs-t_ps=%0.1f.png" % args.pulse_duration)
	plt.clf()
