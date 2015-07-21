#!/usr/bin/python
import pyNN.spiNNaker as sim
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
sim_runtime = 80.

num_neurons = 5

quiet_time = 20.
#~ stim_spike_times = [[], 
                    #~ [0., 0+quiet_time+8, 0+8+2*quiet_time], 
                    #~ [3., 3+quiet_time+4, 3+4+2*quiet_time],
                    #~ [7., 7+quiet_time+0, 7+0+2*quiet_time],
                    #~ []]
                    
stim_spike_times = [[], 
                    [1, 8+quiet_time, 0+8+2*quiet_time], 
                    [1, 4+quiet_time, 3+8+2*quiet_time],
                    [1, 0+quiet_time, 7+8+2*quiet_time],
                    []]
                    
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

#~ inh_cell_type = 'FS'
#~ inh_start_v = -65
#~ cell_params_izk_inh = {'a': izk_types[inh_cell_type]['a'],
                       #~ 'b': izk_types[inh_cell_type]['b'],
                       #~ 'c': izk_types[inh_cell_type]['c'],
                       #~ 'd': izk_types[inh_cell_type]['d'],
                       #~ 'v_init': inh_start_v,
                       #~ 'u_init': izk_types[inh_cell_type]['b']*inh_start_v,
                       #~ 'i_offset': 0.0,
                      #~ }





#rngseed = int(time.time())
rngseed = 1
rng = sim.NumpyRNG(seed=rngseed)

celltype = sim.IZK_curr_exp



sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)


#~ normal_distribution = RandomDistribution('normal', [0., 1.], rng=rng)

sim_pop   = sim.Population(num_neurons, celltype, cell_params_izk_exc, 
                           label="Polychronous toy")


#~ stim_pop  = sim.Population(num_neurons,
                           #~ sim.SpikeSourcePoisson,
                           #~ {'rate': 10., 'start': 0., 'duration': sim_runtime,},
                           #~ label="Stimulation for net")

stim_pop  = sim.Population(num_neurons,
                           sim.SpikeSourceArray,
                           {'spike_times': stim_spike_times},
                           label="Stimulation for net")


conn_weight = 3.
conn_list = [(1, 0, conn_weight, 1.),
             (2, 0, conn_weight, 5.),
             (3, 0, conn_weight, 9.),
             (1, 4, conn_weight, 8.),
             (2, 4, conn_weight, 5.),
             (3, 4, conn_weight, 1.),
            ]



stim_conn_weight = 26.
stim_conn_list = [(1, 1, stim_conn_weight, 1.),
                  (2, 2, stim_conn_weight, 1.),
                  (3, 3, stim_conn_weight, 1.),
                 ]


sim_to_sim_conn = sim.FromListConnector(conn_list)

sim_to_sim_proj = sim.Projection(sim_pop, sim_pop, sim_to_sim_conn,
                                  target="excitatory")

stim_to_sim_conn = sim.FromListConnector(stim_conn_list)

stim_to_sim_proj = sim.Projection(stim_pop, sim_pop, stim_to_sim_conn,
                                  target="excitatory")



sim_pop.record()

sim.run(sim_runtime)

any_spikes_recorded = True

try:
    sim_spikes = sim_pop.getSpikes(compatible_output=True)
except IndexError:
    print("No spikes?")
    any_spikes_recorded = False


sim.end()

if any_spikes_recorded == True:
  spike_times = [spike_time for (neuron_id, spike_time) in sim_spikes]
  spike_ids  = [num_neurons - neuron_id for (neuron_id, spike_time) in sim_spikes]

  fig = pylab.figure()
  ax = fig.gca()
  #ax.set_aspect("equal")
  ax.set_xticks(numpy.arange(0,  sim_runtime + 1, 5))
  ax.set_yticks(numpy.arange(-1, num_neurons + 1, 1.) )
  pylab.xlim([0,sim_runtime+1])
  pylab.ylim([-0.2,num_neurons+0.2])
  pylab.xlabel('Time (ms)')
  pylab.ylabel('Spikes')
  pylab.grid()

  #~ pylab.subplot(2,1,1)
  pylab.plot(spike_times, spike_ids, "o", markerfacecolor="None",
             markeredgecolor="Blue")#, markersize=2)
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
