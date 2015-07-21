#!/usr/bin/python
import pyNN.spiNNaker as sim
from pyNN.random import RandomDistribution
from pyNN.space import Grid2D
import pylab
import numpy
from numpy import where
import time
import os

izk_types = {'RS':  {'a': 0.02, 'b': 0.2,  'c': -65., 'd':  8.},
             'IB':  {'a': 0.02, 'b': 0.2,  'c': -55., 'd':  4.},
             'CH':  {'a': 0.02, 'b': 0.2,  'c': -50., 'd':  2.},
             'FS':  {'a': 0.1,  'b': 0.2,  'c': -65., 'd':  2.},
             'TC':  {'a': 0.02, 'b': 0.25, 'c': -65., 'd':  0.05},
             'RZ':  {'a': 0.1,  'b': 0.26, 'c': -60., 'd': -1.},
             'LTS': {'a': 0.02, 'b': 0.25, 'c': -65., 'd':  2.},
            }
            

max_delay = 32.
min_delay = 1.
timestep  = 1.
sim_runtime = 1000.
connection_probability = 0.2

img_width = 28
img_height = 28
img_neurons = img_width*img_height
num_neurons = img_neurons

celltype = sim.IZK_curr_exp

exc_cell_type = 'RS'
exc_start_v = -65.
cell_params_izk_exc = {'a': izk_types[exc_cell_type]['a'],
                       'b': izk_types[exc_cell_type]['b'],
                       'c': izk_types[exc_cell_type]['c'],
                       'd': izk_types[exc_cell_type]['d'],
                       'v_init': exc_start_v,
                       'u_init': 0.2*exc_start_v,
                       'i_offset': 0.0,
                       #'tau_syn_E': 2,
                       #'tau_syn_I': 2,
                      }
num_exc = num_neurons

inh_cell_type = 'FS'
inh_start_v = -65.
cell_params_izk_inh = {'a': izk_types[inh_cell_type]['a'],
                       'b': izk_types[inh_cell_type]['b'],
                       'c': izk_types[inh_cell_type]['c'],
                       'd': izk_types[inh_cell_type]['d'],
                       'v_init': inh_start_v,
                       'u_init': 0.2*inh_start_v,
                       'i_offset': 0.0,
                       #'tau_syn_E': 2,
                       #'tau_syn_I': 2,
                      }
num_inh = int(num_neurons*(1./4.))

rngseed = int(time.time())
#rngseed = 1
rng = sim.NumpyRNG(seed=rngseed)
neuron_model = sim.IZK_curr_exp

sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)

############ stdp

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                        nearest=True),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=10,
                                                   A_plus=0.02, A_minus=0.02)
)

############ random object

exc_delay_dist = RandomDistribution(distribution='uniform', parameters=[0,20])
inh_delay_dist = RandomDistribution(distribution='uniform', parameters=[0,img_neurons])
exc_weight_dist = RandomDistribution(distribution='uniform', parameters=[0,6])
inh_weight_dist = RandomDistribution(distribution='uniform', parameters=[0,5])
############ structure

grid = Grid2D()

############ populations 

input_layer = sim.Population(img_neurons, neuron_model, cell_params_izk_exc, 
                             label="Input layer - exc", structure=grid)

learn_layer = {}
learn_layer['exc'] = sim.Population(num_exc, neuron_model, cell_params_izk_exc, 
                                    label="Learn layer - exc", structure=grid)
learn_layer['inh'] = sim.Population(num_inh, neuron_model, cell_params_izk_inh, 
                                    label="Learn layer - inh", structure=grid)

id_layer = sim.Population(img_neurons, neuron_model, cell_params_izk_exc, 
                          label="Output layer - exc", structure=grid)

############ connections

connections = {}

distance_conn = sim.DistanceDependentProbabilityConnector("exp(-(d*d))", 
                                                          allow_self_connections=False)

prob_conn = {}
prob_conn['exc2exc'] = sim.FixedProbabilityConnector(connection_probability,
                                                     delays=exc_delay_dist,
                                                     weights=exc_weight_dist,
                                                     allow_self_connections=False)
prob_conn['exc2exc'] = sim.FixedProbabilityConnector(connection_probability,
                                                     delays=exc_delay_dist,
                                                     weights=exc_weight_dist,
                                                     allow_self_connections=False)


connections['in2exc'] = sim.Projection(input_layer, learn_layer['exc'], 
                                       probability_conn, stdp_model)
connections['in2inh'] = sim.Projection(input_layer, learn_layer['inh'], 
                                       probability_conn, stdp_model)

connections['exc2exc'] = sim.Projection(learn_layer['exc'], learn_layer['exc'], 
                                        probability_conn, stdp_model)
connections['inh2inh'] = sim.Projection(learn_layer['inh'], learn_layer['inh'], 
                                        probability_conn, stdp_model)

############ start sim

sim.run(sim_runtime)

############ get data out

############ finish sim

sim.end()

