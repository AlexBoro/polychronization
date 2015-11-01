#!/usr/bin/python
import pyNN.spiNNaker as sim
from pyNN.random import RandomDistribution
import pylab
import numpy
from numpy.random import randint
from numpy import where
import time
import os


def random_thalamic_input(run_time, pop_size):
  spike_times = []
  for idx in range(pop_size):
    spike_times.append([])
  
  for step in range(run_time):
    spike_times[randint(0, pop_size+1)] = step

  return spike_times
  
def split_times_pops(full_times, pop1_size, pop2_size):
  spike_times_1 = []
  spike_times_2 = []
  for idx in range(pop1_size):
    spike_times_1.append(full_times[idx])

  for idx in range(pop1_size, pop1_size + pop2_size):
    spike_times_2.append(full_times[idx])

  return spike_times_1, spike_times_2


#millisecond*second*minutes*hour in a day
runtime = 1000*60*60*24 

weight_to_spike = 6.

timestep = 1
min_delay = 1
max_delay = 20

max_weight = 10.
min_weight = 0.001
a_plus = 0.1
a_minus = 0.12
tau = 20.

num_exc = 800
num_inh = 200
total_neurons = num_exc + num_inh
max_conn_per_neuron = 100
conn_prob = float(max_conn_per_neuron)/float(total_neurons)

cell_type = sim.IZK_curr_exp

exc_params = {'a': 0.02,
              'b': 0.2,
              'c': -65,
              'd': 8,
              'v_init': -65,
              'u_init': 0.2*(-65),
             }
init_exc_weight = 6

inh_params = {'a': 0.1,
              'b': 0.2,
              'c': -65,
              'd': 2,
              'v_init': -65,
              'u_init': 0.2*(-65),
             }
init_inh_weight = 5

total_stim = random_thalamic_input(run_time, total_neurons)
exc_stim, inh_stim = 

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau, tau_minus=tau,
                                        nearest=True),
    weight_dependence=sim.MultiplicativeWeightDependence(w_min=min_weight, w_max=max_weight,
                                                   A_plus=a_plus, A_minus=a_minus)
  )


exc_pop = sim.Population(num_exc, cell_type, exc_params, 
                         label="excitatory neurons")


inh_pop = sim.Population(num_inh, cell_type, inh_params, 
                         label="excitatory neurons")

stimE_pop = sim.Population(num_exc, sim.SpikeSourceArray,
                          {'spike_times': stimE_spike_times},
                          label="exc network stimulation")

stimI_pop = sim.Population(num_inh, sim.SpikeSourceArray,
                          {'spike_times': stimI_spike_times},
                          label="inh network stimulation")

rand_delays = sim.RandomDistribution(boundaries=(min_delay, max_delay))

conn_exc = sim.FixedProbabilityConnector(conn_prob, allow_self_connections=False,
                                         weights=init_exc_weight,
                                         delays=rand_delays)
                                
conn_inh = sim.FixedProbabilityConnector(conn_prob, allow_self_connections=False,
                                         weights=init_inh_weight,
                                         delays=rand_delays)

a2a_conn = sim.AllToAllConnector(weights=weight_to_spike)

e2e_proj = sim.Projection(exc_pop, exc_pop, conn_exc)
e2i_proj = sim.Projection(exc_pop, inh_pop, conn_exc)
i2i_proj = sim.Projection(inh_pop, inh_pop, conn_inh)
i2e_proj = sim.Projection(inh_pop, exc_pop, conn_inh)

s2e_proj = sim.Projection(stimE_pop, exc_pop, a2a_conn) 
s2i_proj = sim.Projection(stimI_pop, inh_pop, a2a_conn)







sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)
