#!/usr/bin/python
import pyNN.spiNNaker as sim
import pylab
import numpy
from numpy import where
import time
import os

def find_sequence(sequence, spike_times, spike_ids):
  sorted_indices = numpy.argsort(spike_times)
  
  ids = 0
  times = 1
  seq_length = len(sequence[ids])
  seq_idx = 0
  positives = []
  first_idx = 0
  test_seq = []
  
  prev_idx = sorted_indices[0]
  first_found = False
  for idx in sorted_indices:
    if seq_idx == seq_length:
      print("Success - New test!!!")
      positives.append(test_seq)
      seq_idx = 0
      test_seq = []
      first_found = False
      
    if spike_ids[idx] == sequence[ids][seq_idx]:
      test_seq.append(idx)
      seq_idx += 1
      first_found = True
      print test_seq
    else:
      if first_found:
        prev_time = spike_times[test_seq[seq_idx - 1]]
      else:
        prev_time = 0
      
      curr_time = spike_times[idx]
      time_diff = curr_time - prev_time
      #print("time_diff -> %s, seq_diff -> %s"%(time_diff, sequence[times][seq_idx] ))
      if time_diff > sequence[times][seq_idx] + 1:
        #print("New test!!!")
        first_found = False
        test_seq = []
        seq_idx = 0
      

    
  return positives


#######################################################################
#######################################################################
#######################################################################

izk_types = {'RS':  {'a': 0.02, 'b': 0.2,  'c': -65., 'd':  8.},
             'IB':  {'a': 0.02, 'b': 0.2,  'c': -55., 'd':  4.},
             'CH':  {'a': 0.02, 'b': 0.2,  'c': -50., 'd':  2.},
             'FS':  {'a': 0.1,  'b': 0.2,  'c': -65., 'd':  2.},
             'TC':  {'a': 0.02, 'b': 0.25, 'c': -65., 'd':  0.05},
             'RZ':  {'a': 0.1,  'b': 0.26, 'c': -60., 'd': -1.},
             'LTS': {'a': 0.02, 'b': 0.25, 'c': -65., 'd':  2.},
            }
            

use_poisson = True
use_stdp = False
max_delay = 32.
min_delay = 1.
timestep  = 1.
sim_runtime = 3600000.

num_neurons = 5

time_padding = 1
stim_spike_times = [[time_padding + 0], 
                    [time_padding + 2], 
                    [time_padding + 4],
                    [time_padding + 2],
                    [time_padding + 2]]

                    
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


rngseed = int(time.time())
#rngseed = 1
rng = sim.NumpyRNG(seed=rngseed)

celltype = sim.IZK_curr_exp



sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)


#~ normal_distribution = RandomDistribution('normal', [0., 1.], rng=rng)

sim_pop   = sim.Population(num_neurons, celltype, cell_params_izk_exc, 
                           label="Polychronous toy - 5 -")


if use_poisson:
  stim_pop  = sim.Population(num_neurons,
                             sim.SpikeSourcePoisson,
                             {'rate': 1., 'start': 1., 'duration': sim_runtime,},
                             label="Stimulation for net")
else:
  stim_pop  = sim.Population(num_neurons,
                             sim.SpikeSourceArray,
                             {'spike_times': stim_spike_times},
                             label="Stimulation for net")


conn_weight = 6.
conn_list = [(0, 1, conn_weight, 6.),
             (0, 2, conn_weight, 4.),
             (0, 3, conn_weight, 2.),
             (0, 4, conn_weight, 2.),
             (1, 0, conn_weight, 2.),
             (1, 2, conn_weight, 7.),
             (1, 3, conn_weight, 7.),
             (1, 4, conn_weight, 2.),
             (2, 0, conn_weight, 4.),
             (2, 1, conn_weight, 2.),
             (2, 3, conn_weight, 2.),
             (2, 4, conn_weight, 7.),
             (3, 0, conn_weight, 10.),
             (3, 1, conn_weight, 3.),
             (3, 2, conn_weight, 5.),
             (3, 4, conn_weight, 3.),
             (4, 0, conn_weight, 5.),
             (4, 1, conn_weight, 4.),
             (4, 2, conn_weight, 4.),
             (4, 3, conn_weight, 4.),]


stim_conn_weight = 26.
stim_conn_list = [
                  (0, 0, stim_conn_weight, 1.),
                  (1, 1, stim_conn_weight, 1.),
                  (2, 2, stim_conn_weight, 1.),
                  (3, 3, stim_conn_weight, 1.),
                  (4, 4, stim_conn_weight, 1.),
                 ]


sim_to_sim_conn = sim.FromListConnector(conn_list)

if use_stdp:
  stdp_model = sim.STDPMechanism(
      timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                          nearest=True),
      weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=10,
                                                     A_plus=0.02, A_minus=0.02)
  )

  sim_to_sim_proj = sim.Projection(sim_pop, sim_pop, sim_to_sim_conn,
                                   synapse_dynamics=sim.SynapseDynamics(slow=stdp_model),
                                   target="excitatory")
else:
  sim_to_sim_proj = sim.Projection(sim_pop, sim_pop, sim_to_sim_conn,
                                   target="excitatory")



stim_to_sim_conn = sim.FromListConnector(stim_conn_list)
#stim_to_sim_conn = sim.OneToOneConnector()

stim_to_sim_proj = sim.Projection(stim_pop, sim_pop, stim_to_sim_conn,
                                  target="excitatory")



if use_poisson:
  stim_pop.record()

sim_pop.record()

sim.run(sim_runtime)




any_spikes_recorded = True

try:
  sim_spikes = sim_pop.getSpikes(compatible_output=True)
except IndexError:
  print("No spikes?")
  any_spikes_recorded = False

try:
  stim_spikes = stim_pop.getSpikes(compatible_output=True)
except IndexError:
  print("No spikes?")
  any_spikes_recorded = False


sim.end()



if any_spikes_recorded == True:
  spike_times = [spike_time for (neuron_id, spike_time) in sim_spikes]
  #spike_ids  = [num_neurons - neuron_id for (neuron_id, spike_time) in sim_spikes]
  spike_ids  = [neuron_id for (neuron_id, spike_time) in sim_spikes]
  
  fig = pylab.figure()
  ax = fig.gca()
  #~ ax.set_aspect("equal")
  #~ ax.set_xticks(numpy.arange(0,  sim_runtime + 1, 100))
  #~ ax.set_yticks(numpy.arange(-1, num_neurons, 1.) )
  pylab.xlim([0,sim_runtime+1])
  pylab.ylim([-0.2,num_neurons+0.2])
  pylab.xlabel('Time (ms)')
  pylab.ylabel('Spikes')
  #~ pylab.grid()

  #~ pylab.subplot(2,1,1)
  pylab.plot(spike_times, spike_ids, "o", markerfacecolor="None",
             markeredgecolor="Blue", markersize=10)


  sequence = [[4, 1, 0, 2, 1],[0, 4, 1, 5, 1]]
  positives = find_sequence(sequence, spike_times, spike_ids)
  
  if len(positives) > 0:
    spike_times[:] = []
    spike_ids[:] = []
    for s in positives:
      count = 0
      for idx in s:
        spike_times.append(idx)
        spike_ids.append(sequence[0][count])
        count += 1
    
    pylab.plot(spike_times, spike_ids, "x", markerfacecolor="Green",
               markeredgecolor="Green", markersize=10)
    
  if use_poisson:
    spike_times = [spike_time for (neuron_id, spike_time) in stim_spikes]
    #spike_ids  = [num_neurons - neuron_id for (neuron_id, spike_time) in stim_spikes]
    spike_ids  = [neuron_id for (neuron_id, spike_time) in stim_spikes]
    pylab.plot(spike_times, spike_ids, "s", markerfacecolor="None",
               markeredgecolor="Red", markersize=2)
  else:
    spike_id = 0
    for spike_array in stim_spike_times:
      num_spikes = len(spike_array)
      if num_spikes > 0:
        spike_ids = numpy.ones(num_spikes)*(num_neurons - spike_id)
        pylab.plot(spike_array, spike_ids, "s", markerfacecolor="None",
                   markeredgecolor="Red")#, markersize=2)
      spike_id += 1
    
 
  dirname = "results"
  if not(os.path.isdir(dirname)):
    os.mkdir(dirname)
  filename = 'toy_polychronous_fig-%s.png'%(time.strftime("%Y-%m-%d_%I-%M"))
  fig_file = open(os.path.join(dirname,filename), 'w')
  pylab.savefig(fig_file)

  pylab.show()

  
  
  
