#!/usr/bin/python
import pyNN.spiNNaker as sim
import spynnaker_external_devices_plugin.pyNN as ExternalDevices
from spynnaker_external_devices_plugin.pyNN.connections\
.spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection

from pyNN.random import RandomDistribution
from pyNN.space import Grid2D, Grid3D, Cuboid, BaseStructure
import pylab
import numpy
from numpy import where, exp, power, sqrt
import time
import os
import glob
import cPickle as pickle
from threading import Condition
from generate_populations import *
from generate_connections import *
from spike_adjustments import *


def plot_spikearray(spike_array):
  x = []
  y = []
  for neuron_id in xrange(len(spike_array)):
    for spike_time in spike_array[neuron_id]:
      y.append(neuron_id)
      x.append(spike_time)
  
  pylab.scatter(x, y, s=0.5)

def send_input(label, sender):
  ms_count = 0
  spikes_per_ms = 5
  neurons = []
  spike_count = 0
  curr_time = 0
  prev_time = 0
  ms = 1./1000.
  for n in xrange(num_samples):
    num_focal_spikes = len(focal_spikes[n]['spk'])
    max_spikes = int(num_focal_spikes*0.3)
    for rep in xrange(num_periods):
      prev_time = time.clock()
      for spike_idx in xrange(max_spikes):
        neuron_id, coeff, layer = focal_spikes[n]['spk'][spike_idx]
        
        if layer < 3:# == in_id:
          neurons.append(neuron_id)
          
          spike_count += 1
        
        if spike_count == spikes_per_ms:
          print_condition.acquire()
          print "Sending spikes", neurons, "sample ", n
          print_condition.release()
          sender.send_spikes(label, neurons)
          ms_count += 1
          spike_count = 0
          neurons[:] = []
          curr_time = time.clock()
          time_diff = curr_time - prev_time
          prev_time = curr_time
          if time_diff < ms:
            time.sleep(ms - time_diff)


          

      print "repeat! ", rep    
      time.sleep(period_length/1000.)
      
    time.sleep(period_length/1000.)
#######################################################################

print_condition = Condition()

#~ pars=[0.02      0.2     -65      6       14 ;...    % tonic spiking
      #~ 0.02      0.25    -65      6       0.5 ;...   % phasic spiking
      #~ 0.02      0.2     -50      2       15 ;...    % tonic bursting
      #~ 0.02      0.25    -55     0.05     0.6 ;...   % phasic bursting
      #~ 0.02      0.2     -55     4        10 ;...    % mixed mode

      #~ 0.01      0.2     -65     8        30 ;...    % spike frequency adaptation
      #~ 0.02      -0.1    -55     6        0  ;...    % Class 1
      #~ 0.2       0.26    -65     0        0  ;...    % Class 2
      
      #~ 0.02      0.2     -65     6        7  ;...    % spike latency
      #~ 0.05      0.26    -60     0        0  ;...    % subthreshold oscillations
      #~ 0.1       0.26    -60     -1       0  ;...    % resonator
      
      #~ 0.02      -0.1    -55     6        0  ;...    % integrator
      #~ 0.03      0.25    -60     4        0;...      % rebound spike
      #~ 0.03      0.25    -52     0        0;...      % rebound burst
      
      #~ 0.03      0.25    -60     4        0  ;...    % threshold variability
      #~ 1         1.5     -60     0      -65  ;...    % bistability
        #~ 1       0.2     -60     -21      0  ;...    % DAP
        
      #~ 0.02      1       -55     4        0  ;...    % accomodation
     #~ -0.02      -1      -60     8        80 ;...    % inhibition-induced spiking
     #~ -0.026     -1      -45     0        80];       % inhibition-induced bursting



izk_types = {'RS':  {'a': 0.02,  'b': 0.2,  'c': -65., 'd':  8.,   'I': 0},
             'IB':  {'a': 0.02,  'b': 0.2,  'c': -55., 'd':  4.,   'I': 0},
             'CH':  {'a': 0.02,  'b': 0.2,  'c': -50., 'd':  2.,   'I': 0},
             'FS':  {'a': 0.1,   'b': 0.2,  'c': -65., 'd':  2.,   'I': 0},
             'TC':  {'a': 0.02,  'b': 0.25, 'c': -65., 'd':  0.05, 'I': 0},
             'RZ':  {'a': 0.1,   'b': 0.26, 'c': -60., 'd': -1.,   'I': 0},
             'LTS': {'a': 0.02,  'b': 0.25, 'c': -65., 'd':  2.,   'I': 0},
###############################
             'TS':  {'a': 0.02,   'b': 0.2,  'c': -65., 'd':  6.,   'I': 0},
             'PS':  {'a': 0.02,   'b': 0.25, 'c': -65., 'd':  6.,   'I': 0},
             'TB':  {'a': 0.02,   'b': 0.2,  'c': -50., 'd':  2.,   'I': 0},
             'PB':  {'a': 0.02,   'b': 0.25, 'c': -55., 'd':  0.05, 'I': 0},
             'MM':  {'a': 0.02,   'b': 0.2,  'c': -55., 'd':  4.,   'I': 0},
             'SFA': {'a': 0.01,   'b': 0.2,  'c': -65., 'd':  8.,   'I': 0},
             'C1':  {'a': 0.02,   'b': -0.1, 'c': -55., 'd':  6.,   'I': 0},
             'C2':  {'a': 0.2,    'b': 0.26, 'c': -65., 'd':  0.,   'I': 0},
             'SL':  {'a': 0.02,   'b': 0.2,  'c': -65., 'd':  6.,   'I': 0},
             'STO': {'a': 0.05,   'b': 0.26, 'c': -60., 'd':  0.,   'I': 0},
             #RZ
             'INT': {'a': 0.02,   'b': -0.1, 'c': -55., 'd':  6.,   'I': 0},
             'RBS': {'a': 0.03,   'b': 0.25, 'c': -60., 'd':  4.,   'I': 0},
             'RBB': {'a': 0.03,   'b': 0.25, 'c': -52., 'd':  0.,   'I': 0},
             'TV':  {'a': 0.03,   'b': 0.25, 'c': -60., 'd':  4.,   'I': 0},
             'BI':  {'a': 1.,     'b': 1.5,  'c': -60., 'd':  4.,   'I': 0},
             'DAP': {'a': 1.,     'b': 0.2,  'c': -60., 'd': -21.,  'I': 0},
             'ACC': {'a': 0.02,   'b': 1.,   'c': -55., 'd':  4.,   'I': 0},
             'ISS': {'a': -0.02,  'b': -1,   'c': -60., 'd':  8.,   'I': 80},
             'ISB': {'a': -0.026, 'b': -1,   'c': -45., 'd':  0.,   'I': 80},
            }


in_id = 1
new_conn_list = True


max_delay = 32.
min_delay = 1.
timestep  = 1.
sim_runtime = 1000.
exc_conn_probability = 0.2
inh_conn_probability = 0.3

period_length = 200 #ms
num_periods   = 4

img_width = 28
img_height = 28
img_neurons = img_width*img_height
cube_z = 1.5
num_neurons = img_neurons*cube_z*3.

exc_cell_type = 'RS'
exc_start_v = start_v
cell_params_izk_exc = {'a': izk_types[exc_cell_type]['a'],
                       'b': izk_types[exc_cell_type]['b'],
                       'c': izk_types[exc_cell_type]['c'],
                       'd': izk_types[exc_cell_type]['d'],
                       'v_init': exc_start_v,
                       'u_init': 0.2*exc_start_v,
                       'i_offset': izk_types[exc_cell_type]['I'],
                       #'tau_syn_E': 2,
                       #'tau_syn_I': 2,
                      }
#~ num_exc = num_neurons

inh_cell_type = 'FS'
inh_start_v = start_v
cell_params_izk_inh = {'a': izk_types[inh_cell_type]['a'],
                       'b': izk_types[inh_cell_type]['b'],
                       'c': izk_types[inh_cell_type]['c'],
                       'd': izk_types[inh_cell_type]['d'],
                       'v_init': inh_start_v,
                       'u_init': 0.2*inh_start_v,
                       'i_offset': izk_types[inh_cell_type]['I'],
                       #'tau_syn_E': 2,
                       #'tau_syn_I': 2,
                      }
                      
inv_cell_type = 'ISS'
inv_start_v = start_v
cell_params_izk_inv = {'a': izk_types[inv_cell_type]['a'],
                       'b': izk_types[inv_cell_type]['b'],
                       'c': izk_types[inv_cell_type]['c'],
                       'd': izk_types[inv_cell_type]['d'],
                       'v_init': inh_start_v,
                       'u_init': 0.2*inh_start_v,
                       'i_offset': izk_types[inv_cell_type]['I'],
                       #'tau_syn_E': 2,
                       #'tau_syn_I': 2,
                      }

inj_cell_params = {
  'port': 12345,
}
inj_cell_type = ExternalDevices.SpikeInjector


#~ num_inh = int(num_neurons*(1./4.))

rngseed = int(time.time())
#rngseed = 1
rng = sim.NumpyRNG(seed=rngseed)
neuron_model = sim.IZK_curr_exp

sim.set_number_of_neurons_per_core(neuron_model, 128)
sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)

############# stdp

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                        nearest=True),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0.001, w_max=3.,
                                                   A_plus=0.02, A_minus=0.02)
  )
############ populations 

pop_2d = grid2d_coords(img_neurons)
pop_3d = rand_pop_3d_coords(num_neurons, 
                            space_width=img_width*1.1, 
                            space_height=img_height*1.1, 
                            space_depth=cube_z)
num_inh = int(num_neurons*0.2)
num_exc = int(num_neurons*0.8)
inh_3d = pop_3d[:num_inh]
exc_3d = pop_3d[num_inh:]

num_inv = int(img_neurons*0.5)
inv_3d = rand_pop_3d_coords(num_inv,
                            space_width=img_width*1.1, 
                            space_height=img_height*1.1, 
                            space_depth=0.2)
inv_3d[:,2] -= 0.1 #translate back 0.1 units

min_z = (pop_3d[:,2].min() - 0.2)*numpy.ones((pop_2d.shape[0], 1))
pop_2d_3d = hstack((pop_2d, min_z))

input_layer = {}
#~ for i in xrange(len(input_spikes)):
  #~ input_layer[i] = sim.Population(img_neurons, sim.SpikeSourceArray, 
                                  #~ {"spike_times": input_spikes[i]}, 
                                  #~ structure=grid) 


input_layer[in_id] = sim.Population(img_neurons, inj_cell_type,
                                    inj_cell_params, 
                                    label="spike_injector")

live_spikes_connection_send = SpynnakerLiveSpikesConnection(
                                  receive_labels=None, local_port=19999,
                                  send_labels=["spike_injector"])

live_spikes_connection_send.add_start_callback("spike_injector", send_input)



#~ input_layer[in_id] = sim.Population(img_neurons, sim.SpikeSourceArray, 
                                    #~ {"spike_times": input_spikes[in_id]}
                                   #~ ) 
learn_layer = {}

learn_layer['exc'] = sim.Population(num_exc+1, neuron_model, cell_params_izk_exc, 
                                    label="Learn layer - exc")
learn_layer['inh'] = sim.Population(num_inh+1, neuron_model, cell_params_izk_inh, 
                                    label="Learn layer - inh")
learn_layer['inv'] = sim.Population(img_neurons, neuron_model, cell_params_izk_inv,
                                    label="Learn layer - inv")
                                    
learn_layer['exc'].record()
learn_layer['inh'].record()
learn_layer['inv'].record()

ExternalDevices.activate_live_output_for(learn_layer['exc'], 
                                         database_notify_host="localhost",
                                         database_notify_port_num=19996)

ExternalDevices.activate_live_output_for(learn_layer['inh'], 
                                         database_notify_host="localhost",
                                         database_notify_port_num=19997)

ExternalDevices.activate_live_output_for(learn_layer['inv'], 
                                         database_notify_host="localhost",
                                         database_notify_port_num=19998)

print("Total neurons: %s"%(2*img_neurons+num_exc*2+num_inh+1))

############ connectors



input_strength = 1.8

my_connectors = {}
my_connectors['in2exc'] = dist_dep_conn_3d_3d(pop_2d_3d, exc_3d, 
                                             dist_rule = "exp(-(d**2))",
                                             conn_prob = 0.5, 
                                             weight=input_strength,
                                             new_format=new_conn_list)

my_connectors['in2inh'] = dist_dep_conn_3d_3d(pop_2d_3d, inh_3d, 
                                             dist_rule = "exp(-(d**2))",
                                             conn_prob = 0.5, 
                                             weight=input_strength,
                                             new_format=new_conn_list)
                             
my_connectors['in2inv'] = dist_dep_conn_3d_3d(pop_2d_3d, inv_3d, 
                                             dist_rule = "exp(-(d**2))",
                                             conn_prob = 0.2, 
                                             weight=input_strength,
                                             new_format=new_conn_list)

my_connectors['inv2inh'] = dist_dep_conn_3d_3d(inv_3d, inh_3d, 
                                               dist_rule = "exp(-(d**2))",
                                               conn_prob = 0.1, 
                                               new_format=new_conn_list)

my_connectors['inv2exc'] = dist_dep_conn_3d_3d(inv_3d, exc_3d, 
                                               dist_rule = "exp(-(d**2))",
                                               conn_prob = 0.1, 
                                               new_format=new_conn_list)


my_connectors['inh2exc'] = dist_dep_conn_3d_3d(inh_3d, exc_3d, 
                                              dist_rule = "exp(-(d**2))",
                                              conn_prob = 0.1,
                                              weight=2.0,
                                              new_format=new_conn_list)

my_connectors['inh2inh'] = dist_dep_conn_3d_3d(inh_3d, inh_3d, 
                                              dist_rule = "exp(-(d**2))",
                                              conn_prob = 0.1, 
                                              new_format=new_conn_list)

my_connectors['exc2exc'] = dist_dep_conn_3d_3d(exc_3d, exc_3d, 
                                              dist_rule = "exp(-sqrt(d*d))",
                                              conn_prob = 0.2, 
                                              new_format=new_conn_list)
                                            
my_connectors['exc2inh'] = dist_dep_conn_3d_3d(exc_3d, inh_3d, 
                                              dist_rule = "exp(-sqrt(d*d))",
                                              conn_prob = 0.2, 
                                              new_format=new_conn_list)




connectors = {}
for k in my_connectors:
  connectors[k] = sim.FromListConnector(my_connectors[k])

############ connections

connections = {}

#~ for k in input_layer:
  #~ connections['in2exc_%s'%k] = sim.Projection(input_layer[k], learn_layer['exc'], 
                                              #~ prob_conn['in'], stdp_model,
                                              #~ target='excitatory')
#~ 
  #~ connections['in2inh_%s'%k] = sim.Projection(input_layer[k], learn_layer['inh'], 
                                              #~ prob_conn['in'], stdp_model, 
                                              #~ target='excitatory')

### Excitatory connections
connections['in2exc'] = sim.Projection(input_layer[in_id], learn_layer['exc'],
                                       connectors['in2exc'], stdp_model,
                                       target='excitatory')

connections['in2inh'] = sim.Projection(input_layer[in_id], learn_layer['inh'],
                                       connectors['in2inh'], stdp_model,
                                       target='excitatory')
                                       
                                       
connections['in2inv'] = sim.Projection(input_layer[in_id], learn_layer['inh'],
                                       connectors['in2inv'], stdp_model,
                                       target='excitatory')

connections['exc2exc'] = sim.Projection(learn_layer['exc'], learn_layer['exc'], 
                                        connectors['exc2exc'], stdp_model,
                                        target='excitatory')

connections['exc2inh'] = sim.Projection(learn_layer['exc'], learn_layer['inh'], 
                                        connectors['exc2inh'], stdp_model,
                                        target='excitatory')

connections['inv2exc'] = sim.Projection(learn_layer['inv'], learn_layer['exc'], 
                                        connectors['inv2exc'], stdp_model,
                                        target='excitatory')

connections['inv2inh'] = sim.Projection(learn_layer['inv'], learn_layer['inh'], 
                                        connectors['inv2inh'], stdp_model,
                                        target='excitatory')

### Inhibitory connections
connections['inh2inh'] = sim.Projection(learn_layer['inh'], learn_layer['inh'], 
                                        connectors['inh2inh'], stdp_model,
                                        target='inhibitory')

connections['inh2exc'] = sim.Projection(learn_layer['inh'], learn_layer['exc'], 
                                        connectors['inh2exc'], stdp_model, 
                                        target='inhibitory')



#### changed spynnaker/pyNN/models/pynn_projection.py 
    #~ def __len__(self):
        #~ """Return the total number of local connections."""
        #~ rows = self._host_based_synapse_list.get_rows()
        #~ count = 0
        #~ for pre_atom in xrange(len(rows)):
          #~ count += len(rows[pre_atom].target_indices)
        #~ return count
#~ #        raise NotImplementedError
print("\n\n")
total_connections = 0
for k in sorted(connections.keys()):
  num_conn = len(connections[k])
  print( "%s\t\tconnections: %s"%(k, num_conn) )
  total_connections += num_conn
print("-------------------------------")  
print("Total connections: %s"%(total_connections))
print("\n\n")

time.sleep(1)
############ start sim

sim.run(sim_runtime)

############ get data out

any_spikes_recorded = True

sim_spikes = {}
try:
  
  #~ sim_spikes['in'] = {}
  #~ for l in input_layer:
    #~ sim_spikes['in'][l] = input_layer[l].getSpikes(compatible_output=True)
    
  sim_spikes['id']  = id_layer.getSpikes(compatible_output=True)
  sim_spikes['exc'] = learn_layer['exc'].getSpikes(compatible_output=True)
  sim_spikes['inh'] = learn_layer['inh'].getSpikes(compatible_output=True)
except IndexError:
  print("No spikes?")
  any_spikes_recorded = False

############ plot

if any_spikes_recorded:

  symbols = {'id': 's', 'exc': '.', 'inh': 'x', 'in': '+'}
  colours = {'id': 'blue', 'exc': 'green', 'inh': 'red', 'in': 'black'}
  titles = {'id': 'Identification layer', 'exc': 'Excitatory neurons',
            'inh': 'Inhibitory neurons', 'in': 'Input layer'}
  layer = 0
  #fig = pylab.figure()
  spike_times = []
  spike_ids = []
  wrap_spikes = []
  fig = pylab.figure()
  for k in sim_spikes:
    spike_times[:] = []
    spike_ids[:] = []
    ax = pylab.subplot(2,2, layer)
    ax.set_title(titles[k])
    
    spike_times = [spike_time for (neuron_id, spike_time) in sim_spikes[k]]
    spike_ids  = [neuron_id for (neuron_id, spike_time) in sim_spikes[k]]
    pylab.plot(spike_times, spike_ids, symbols[k], markerfacecolor="None",
               markeredgecolor=colours[k], markersize=2)



    layer += 1


  ax = pylab.subplot(2,2, 3)
  ax.set_title(titles["in"])
  
  plot_spikearray(full_input_spikes)

############ finish sim

pylab.show()

sim.end()

#~ pickle.dump( sim_spikes, open( "all-spikes.p", "wb" ) )
