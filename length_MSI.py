#!/usr/bin/python
import pyNN.spiNNaker as sim
from pyNN.random import RandomDistribution
import pylab
import numpy
from numpy.random import randint
from numpy import where
import time
import os



runtime = 10000

weight_to_spike = 3.

timestep = 1.
min_delay = 1.
max_delay = 20.


sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)

max_weight = 10.
min_weight = 0.000001
a_plus = 0.1
a_minus = 0.12
tau = 20.

conn_prob = 10./100.

num_neurons = 200

cell_type = sim.IZK_curr_exp

exc_params = {'a': 0.02,
              'b': 0.2,
              'c': -65,
              'd': 8,
              'v_init': -65,
              'u_init': 0.2*(-65),
             }
init_exc_weight = 6.


master_pop = sim.Population(num_exc, cell_type, exc_params, 
                            label="master neurons")

slave_pop = sim.Population(num_exc, cell_type, exc_params, 
                           label="slave neurons")

inhibit_pop = sim.Population(num_exc, cell_type, exc_params, 
                             label="inhibitory neurons")


stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau, tau_minus=tau,
                                        nearest=True),
    weight_dependence=sim.MultiplicativeWeightDependence(w_min=min_weight, w_max=max_weight,
                                                   A_plus=a_plus, A_minus=a_minus)
  )
  
rand_weights = sim.RandomDistribution(boundaries=(min_weight, max_weight))

prob_conn_10pc = sim.FixedProbabilityConnector(conn_prob, allow_self_connections=False,
                                               weights=rand_weights)

m2s_proj = sim.Projection(master_pop,  slave_pop,   prob_conn_10pc, target="excitatory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model))
s2i_proj = sim.Projection(slave_pop,   inhibit_pop, prob_conn_10pc, target="excitatory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model))
i2s_proj = sim.Projection(inhibit_pop, slave_pop,   prob_conn_10pc, target="inhibitory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model))



#max_delay = 1
#
#for m2s_delay in xrange(max_delay):
  
