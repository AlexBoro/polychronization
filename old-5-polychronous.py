#!/usr/bin/python
import pyNN.spiNNaker as sim
import pylab
import numpy
from numpy import where
import time
import os
import pickle

izk_types = {'RS':  {'a': 0.02, 'b': 0.2,  'c': -65., 'd':  8.},
             'IB':  {'a': 0.02, 'b': 0.2,  'c': -55., 'd':  4.},
             'CH':  {'a': 0.02, 'b': 0.2,  'c': -50., 'd':  2.},
             'FS':  {'a': 0.1,  'b': 0.2,  'c': -65., 'd':  2.},
             'TC':  {'a': 0.02, 'b': 0.25, 'c': -65., 'd':  0.05},
             'RZ':  {'a': 0.1,  'b': 0.26, 'c': -60., 'd': -1.},
             'LTS': {'a': 0.02, 'b': 0.25, 'c': -65., 'd':  2.},
            }
sim_weight = 6.



exc_cell_type = 'RS'
exc_start_v = -65.
cell_params_izk_exc = {'a': izk_types[exc_cell_type]['a'],
                       'b': izk_types[exc_cell_type]['b'],
                       'c': izk_types[exc_cell_type]['c'],
                       'd': izk_types[exc_cell_type]['d'],
                       'v_init': exc_start_v,
                       'u_init': izk_types[exc_cell_type]['b']*exc_start_v,
                       'i_offset': 0.0,
                       #'tau_syn_E': 10.,
                       #'tau_syn_I': 2.,
                      }


num_neurons = 5

poisson_stim = True

rngseed = int(time.time())
#rngseed = 1
rng = sim.NumpyRNG(seed=rngseed)

celltype = sim.IZK_curr_exp

timestep = 1.
min_delay = 1.
max_delay = 10.

sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)


#~ normal_distribution = RandomDistribution('normal', [0., 1.], rng=rng)

sim_pop   = sim.Population(num_neurons, celltype, cell_params_izk_exc, 
                           label="Polychronous toy")

if poisson_stim:
  sim_runtime = 360000.
  stim_pop  = sim.Population(num_neurons,
                             sim.SpikeSourcePoisson,
                             {'rate': 20., 'start': 1., 'duration': sim_runtime,},
                             label="Stimulation for net")
else:
  time_padding = 3.
  sim_runtime = 30.
  stim_spike_times = [[time_padding + 0], 
                      [time_padding + 2], 
                      [time_padding + 4],
                      [time_padding + 2],
                      [time_padding + 2]]

  #~ stim_spike_times = [[0.], 
                      #~ [0.], 
                      #~ [0.],
                      #~ [0.],
                      #~ [0.]]

  #~ stim_spike_times = [[],
                      #~ [], 
                      #~ [],
                      #~ [],
                      #~ [3.]]


  stim_pop  = sim.Population(num_neurons,
                             sim.SpikeSourceArray,
                             {'spike_times': stim_spike_times},
                             label="Stimulation for net")



stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                        nearest=True),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9,
                                                   A_plus=0.02, A_minus=0.02)
)

stim_to_sim_conn = sim.OneToOneConnector(weights=20., delays=1.)

sim_to_sim_conn  = sim.FromListConnector(sim_conn)

stim_to_sim_proj = sim.Projection(stim_pop, sim_pop, stim_to_sim_conn,
                                  target="excitatory")
sim_to_sim_proj  = sim.Projection(sim_pop, sim_pop, sim_to_sim_conn,
                                  synapse_dynamics=sim.SynapseDynamics(slow=stdp_model),
                                  target="excitatory")


sim_pop.record()
if poisson_stim:
  stim_pop.record()

sim.run(sim_runtime)

any_spikes_recorded = True

try:
    sim_spikes = sim_pop.getSpikes(compatible_output=True)
    if poisson_stim:
      stim_spikes = stim_pop.getSpikes(compatible_output=True)
except IndexError:
    print("No spikes?")
    any_spikes_recorded = False


sim.end()

if any_spikes_recorded == True:
  spike_times = [spike_time for (neuron_id, spike_time) in sim_spikes]
  spike_ids  = [neuron_id for (neuron_id, spike_time) in sim_spikes]

  fig = pylab.figure()

  pylab.grid()
  pylab.xlim([-0.1, sim_runtime + 1])
  pylab.ylim([-0.1, num_neurons + 0.1])
  pylab.xlabel('Time (ms)')
  pylab.ylabel('Spikes')
  
  spike_times = numpy.array(spike_times) 
  
  if not poisson_stim:
    spike_times -= time_padding + 3.0#spiking delay
    
  pylab.plot(spike_times, spike_ids, "bo", alpha=0.6, markersize=4)
  
  if poisson_stim:
    spike_times = [spike_time for (neuron_id, spike_time) in stim_spikes]
    spike_ids  = [neuron_id for (neuron_id, spike_time) in stim_spikes]
    pylab.plot(spike_times, spike_ids, "rs", alpha=0.6, markersize=4)
    
  else:
    ax = fig.gca()
    ax.set_xticks(numpy.arange(0, sim_runtime + 1, 2.))
    ax.set_yticks(numpy.arange(0, num_neurons, 1.) )
    spike_id = 0
    for spike_array in stim_spike_times:
      num_spikes = len(spike_array)
      if num_spikes > 0:
        spike_ids = numpy.ones(num_spikes)*spike_id
        spike_array = numpy.array(spike_array) - time_padding
        pylab.plot(spike_array, spike_ids, "rs", alpha=0.6, markersize=4)
      spike_id += 1
    
  dirname = "results"
  if not(os.path.isdir(dirname)):
    os.mkdir(dirname)
  filename = 'toy_polychronous_fig-%s.png'%(time.strftime("%Y-%m-%d_%H-%M-%S"))
  fig_file = open(os.path.join(dirname,filename), 'w')
  pylab.savefig(fig_file)
  
  filename = 'toy_polychronous_spikes-%s.p'%(time.strftime("%Y-%m-%d_%H-%M-%S"))
  pickle.dump(sim_spikes, open(os.path.join(dirname,filename), 'wb'))
  pylab.show()
