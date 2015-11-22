#!/usr/bin/python
import pyNN.spiNNaker as sim
from pyNN.random import RandomDistribution
import pylab
import numpy
from numpy.random import randint, uniform
from numpy import where
import time
import os
from fixed_number_post_connector import FixedNumberPostConnector


def connections(max_conn, num_exc, num_inh, max_delay):
  e2e_conns = []
  e2i_conns = []
  i2e_conns = []
  i2i_conns = []
  
  num_neurons = num_exc + num_inh
  
  for pre_idx in range(num_neurons):
    if pre_idx < num_exc: #excitatory to any
      sample = randint(0, num_neurons, max_conn) 
    else: #inhibitory to excitatory only
      sample = randint(0, num_exc, max_conn)
    
    if pre_idx in sample:
      del_idx = numpy.where(sample == pre_idx)
      numpy.delete(sample, del_idx)
    
    if pre_idx < num_exc:
      delays = randint(0, max_delay, max_conn)
      
    delay_idx = 0
    for post_idx in sample:
      delay = delays[delay_idx] if pre_idx < num_exc else 1. #if pre is exc, delay
      weight = 6. if pre_idx < num_exc else 5. # if pre is exc, w=6.; else w=5.
      pre = pre_idx if pre_idx < num_exc else pre_idx - num_exc # reset idx for pre inh
      post = post_idx if post_idx < num_exc else post_idx - num_exc # reset idx for post inh
      
      connection = (pre, post, weight, delay)

      if pre_idx < num_exc:
        if post_idx < num_exc:
          e2e_conns.append(connection)
        else:
          e2i_conns.append(connection)
      else:
        if post_idx < num_exc:
          i2e_conns.append(connection)
        else:
          i2i_conns.append(connection)
      
      delay_idx += 1

  return e2e_conns, e2i_conns, i2e_conns, i2i_conns


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
runtime  = 1000#*60*10 #10 min
stimtime = 500
weight_to_spike = 8.

timestep = 1.
min_delay = 1.
max_delay = 20.


sim.setup(timestep=timestep, min_delay = min_delay, max_delay = max_delay)

max_weight = 10.
min_weight = 0.0
a_plus = 0.1
a_minus = 0.12
tau_plus = 20.
tau_minus = 20.

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

total_stim = random_thalamic_input(stimtime, total_neurons)
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
    timing_dependence=sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus,
                                        nearest=True),
    weight_dependence=sim.AdditiveWeightDependence(w_min=min_weight, w_max=max_weight,
                                                   A_plus=a_plus, A_minus=a_minus)
  )

rng = sim.NumpyRNG(seed=int(time.time()))
#rng = sim.NumpyRNG(seed=1)

e2e_lst, e2i_lst, i2e_lst, i2i_lst = connections(max_conn_per_neuron, 
                                                 num_exc, num_inh, max_delay)

e2e_conn = sim.FromListConnector(e2e_lst)
e2i_conn = sim.FromListConnector(e2i_lst)
i2e_conn = sim.FromListConnector(i2e_lst)
i2i_conn = sim.FromListConnector(i2i_lst)

o2o_conn = sim.OneToOneConnector(weights=weight_to_spike, delays=1.)

#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Excitatory to Excitatory connections")
#~ print("-----------------------------------------------------------------")
e2e_proj = sim.Projection(exc_pop, exc_pop, e2e_conn, target="excitatory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model)
                         )

#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Excitatory to Inhibitory connections")
#~ print("-----------------------------------------------------------------")
e2i_proj = sim.Projection(exc_pop, inh_pop, e2i_conn, target="excitatory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model)
                         )

#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Inhibitory to Excitatory connections")
#~ print("-----------------------------------------------------------------")
i2e_proj = sim.Projection(inh_pop, exc_pop, i2e_conn, target="inhibitory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model)
                         )

#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Inhibitory to Inhibitory connections")
#~ print("-----------------------------------------------------------------")
i2i_proj = sim.Projection(inh_pop, inh_pop, i2i_conn, target="inhibitory",
                          synapse_dynamics = sim.SynapseDynamics(slow = stdp_model)
                         )



s2e_proj = sim.Projection(stimE_pop, exc_pop, o2o_conn, target="excitatory") 
s2i_proj = sim.Projection(stimI_pop, inh_pop, o2o_conn, target="excitatory")


#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Exc to Exc Weights")
#~ print("-----------------------------------------------------------------")
#~ print(e2e_proj.getWeights())


#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Exc to Inh Weights")
#~ print("-----------------------------------------------------------------")
#~ print(e2i_proj.getWeights())


#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Inh to Inh Weights")
#~ print("-----------------------------------------------------------------")
#~ print(i2i_proj.getWeights())


#~ print("-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Inh to Exc Weights")
#~ print("-----------------------------------------------------------------")
#~ print(i2e_proj.getWeights())

print("-----------------------------------------------------------------")
print("-----------------------------------------------------------------")
print("Stim to Inh Weights")
print("-----------------------------------------------------------------")
print(s2i_proj.getWeights())

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
spike_times = []
spike_ids = []
time_window = 1000

if exc_spikes_found or inh_spikes_found:
  
  for start_time in xrange(0, runtime, time_window): #plot 1 sec at a time
    fig = pylab.figure()
    
    #~ ax = fig.gca()
    #~ ax.set_xticks(numpy.arange(0,  runtime + 1, 5))
    #~ ax.set_yticks(numpy.arange(-1, num_neurons + 1, 1.) )
    #pylab.xlim([0,runtime+1])
    #pylab.ylim([-0.2,total_neurons+0.2])
    end_time = start_time + time_window
    fig.suptitle("From %s ms to %s ms"%(start_time, end_time))
    if exc_spikes_found:
      spike_times[:] = []
      spike_ids[:] = []

      spike_times = [spike_time for (neuron_id, spike_time) in exc_spikes \
                         if start_time <= spike_time < end_time]
      spike_ids   = [neuron_id  for (neuron_id, spike_time) in exc_spikes \
                         if start_time <= spike_time < end_time]
      pylab.plot(spike_times, spike_ids, ".", markerfacecolor="None",
                 markeredgecolor="Blue", markersize=1)

    if inh_spikes_found:
      spike_times[:] = []
      spike_ids[:] = []

      spike_times = [spike_time for (neuron_id, spike_time) in inh_spikes \
                         if start_time <= spike_time < end_time]
      spike_ids   = [neuron_id + num_exc for (neuron_id, spike_time) in inh_spikes \
                         if start_time <= spike_time < end_time]
      pylab.plot(spike_times, spike_ids, ".", markerfacecolor="None",
                 markeredgecolor="Red", markersize=1)

    if len(total_stim) > start_time:
      spike_times[:] = []
      spike_ids[:] = []
      neuron_id = 0
      for neuron_spikes in total_stim:
        for spike_time in neuron_spikes:
          if start_time <= spike_time < end_time:
            spike_times.append(spike_time)
            spike_ids.append(neuron_id)

        neuron_id += 1
      pylab.plot(spike_times, spike_ids, "o", markerfacecolor="None",
                 markeredgecolor="Green", markersize=4)
      
    dirname = "results"
    if not(os.path.isdir(dirname)):
      os.mkdir(dirname)
    filename = 'full_polychronous_fig-%s-%s.png'%(time.strftime("%Y-%m-%d_%I-%M"), start_time)
    fig_file = open(os.path.join(dirname,filename), 'w')
    pylab.savefig(fig_file)

  pylab.show()

# delta < 4 Hz == 250 ms or more
# alpha 8 to 12 Hz == 125 to 83 ms
# gamma > 32 Hz == 32 ms or less

#~ print("\n\n\n\n\n\n-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Exc to Exc Weights")
#~ print("-----------------------------------------------------------------")
#~ print(e2e_proj.getWeights())
#~ 
#~ 
#~ print("\n\n\n\n\n\n-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Exc to Inh Weights")
#~ print("-----------------------------------------------------------------")
#~ print(e2i_proj.getWeights())
#~ 
#~ 
#~ print("\n\n\n\n\n\n-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Inh to Inh Weights")
#~ print("-----------------------------------------------------------------")
#~ print(i2i_proj.getWeights())
#~ 
#~ 
#~ print("\n\n\n\n\n\n-----------------------------------------------------------------")
#~ print("-----------------------------------------------------------------")
#~ print("Inh to Exc Weights")
#~ print("-----------------------------------------------------------------")
#~ print(i2e_proj.getWeights())

sim.end()
