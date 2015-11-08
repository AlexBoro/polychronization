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
    spike_times[randint(0, pop_size)].append( step )

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
#runtime = 1000*60*60*24 
runtime = 10000

weight_to_spike = 1.

timestep = 1.
min_delay = 1.
max_delay = 20.


sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)

max_weight = 10.
min_weight = 0.000001
a_plus = 0.1
a_minus = 0.12
tau = 20.

num_exc = 800
num_inh = 200
total_neurons = num_exc + num_inh
max_conn_per_neuron = 100
conn_prob = 0.1#float(max_conn_per_neuron)/float(total_neurons)

cell_type = sim.IZK_curr_exp

exc_params = {'a': 0.02,
              'b': 0.2,
              'c': -65,
              'd': 8,
              'v_init': -65,
              'u_init': 0.2*(-65),
             }
init_exc_weight = 6.

inh_params = {'a': 0.1,
              'b': 0.2,
              'c': -65,
              'd': 2,
              'v_init': -65,
              'u_init': 0.2*(-65),
             }
init_inh_weight = 5.

total_stim = random_thalamic_input((runtime*1)/4, total_neurons)
exc_stim_times, inh_stim_times = split_times_pops(total_stim, num_exc, num_inh)


exc_pop = sim.Population(num_exc, cell_type, exc_params, 
                         label="excitatory neurons")


inh_pop = sim.Population(num_inh, cell_type, inh_params, 
                         label="excitatory neurons")

stimE_pop = sim.Population(num_exc, sim.SpikeSourceArray,
                          {'spike_times': exc_stim_times},
                          label="exc network stimulation")

stimI_pop = sim.Population(num_inh, sim.SpikeSourceArray,
                          {'spike_times': inh_stim_times},
                          label="inh network stimulation")

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau, tau_minus=tau,
                                        nearest=True),
    weight_dependence=sim.MultiplicativeWeightDependence(w_min=min_weight, w_max=max_weight,
                                                   A_plus=a_plus, A_minus=a_minus)
  )

rand_delays = sim.RandomDistribution(boundaries=(min_delay, max_delay))


#~ conn_exc = sim.FixedProbabilityConnector(conn_prob, allow_self_connections=False,
                                         #~ weights=init_exc_weight,
                                         #~ delays=rand_delays)
                                
#~ conn_inh = sim.FixedProbabilityConnector(conn_prob, allow_self_connections=False,
                                         #~ weights=init_inh_weight,
                                         #~ delays=rand_delays)

conn_exc = sim.FixedNumberPostConnector(max_conn_per_neuron, allow_self_connections=False,
                                        weights=init_exc_weight,
                                        delays=rand_delays)
                                
conn_inh = sim.FixedNumberPostConnector(max_conn_per_neuron, allow_self_connections=False,
                                        weights=init_inh_weight,
                                        delays=rand_delays)

a2a_conn = sim.AllToAllConnector(weights=weight_to_spike)

e2e_proj = sim.Projection(exc_pop, exc_pop, conn_exc, target="excitatory")
e2i_proj = sim.Projection(exc_pop, inh_pop, conn_exc, target="excitatory")
i2i_proj = sim.Projection(inh_pop, inh_pop, conn_inh, target="inhibitory")
i2e_proj = sim.Projection(inh_pop, exc_pop, conn_inh, target="inhibitory")

s2e_proj = sim.Projection(stimE_pop, exc_pop, a2a_conn, target="excitatory") 
s2i_proj = sim.Projection(stimI_pop, inh_pop, a2a_conn, target="excitatory")


exc_pop.record()
inh_pop.record()

sim.run(runtime)

exc_spikes_found = True
try:
    exc_spikes = exc_pop.getSpikes(compatible_output=True)
except IndexError:
    print("No spikes?")
    exc_spikes_found = False

inh_spikes_found = True
try:
    inh_spikes = inh_pop.getSpikes(compatible_output=True)
except IndexError:
    print("No spikes?")
    inh_spikes_found = False


#pylab.ion()

if exc_spikes_found or inh_spikes_found:
  fig = pylab.figure()
  #~ ax = fig.gca()
  #~ ax.set_xticks(numpy.arange(0,  runtime + 1, 5))
  #~ ax.set_yticks(numpy.arange(-1, num_neurons + 1, 1.) )
  #pylab.xlim([0,runtime+1])
  pylab.ylim([-0.2,total_neurons+0.2])

  if exc_spikes_found:
    exc_spike_times = [spike_time for (neuron_id, spike_time) in exc_spikes]
    exc_spike_ids   = [neuron_id  for (neuron_id, spike_time) in exc_spikes]
    pylab.plot(exc_spike_times, exc_spike_ids, ".", markerfacecolor="None",
               markeredgecolor="Blue", markersize=1)

  if inh_spikes_found:
    inh_spike_times = [spike_time for (neuron_id, spike_time) in inh_spikes]
    inh_spike_ids   = [neuron_id + num_exc for (neuron_id, spike_time) in inh_spikes]
    pylab.plot(inh_spike_times, inh_spike_ids, ".", markerfacecolor="None",
               markeredgecolor="Red", markersize=1)

  dirname = "results"
  if not(os.path.isdir(dirname)):
    os.mkdir(dirname)
  filename = 'toy_polychronous_fig-%s.png'%(time.strftime("%Y-%m-%d_%I-%M"))
  fig_file = open(os.path.join(dirname,filename), 'w')
  pylab.savefig(fig_file)

  pylab.show()

# delta < 4 Hz == 250 ms or more
# alpha 8 to 12 Hz == 125 to 83 ms
# gamma > 32 Hz == 32 ms or less



sim.end()
