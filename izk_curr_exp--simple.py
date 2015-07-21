"""
the simple model paper simulation
"""
#!/usr/bin/python
from pyNN.spiNNaker import *
import pylab
from numpy import where, zeros
import time




setup(timestep=1, min_delay = 1.0, max_delay = 32.0)
set_number_of_neurons_per_core("IZK_curr_exp", 70)

nNeurons = 700 # number of neurons in each population
excToInhRatio = 4.
nExcNeurons = int(round((nNeurons*excToInhRatio/(1+excToInhRatio))))
nInhNeurons = nNeurons - nExcNeurons
runtime = 1000

izk_types = {'RS':  {'a': 0.02, 'b': 0.2,  'c': -65, 'd': 8},
             'IB':  {'a': 0.02, 'b': 0.2,  'c': -55, 'd': 4},
             'CH':  {'a': 0.02, 'b': 0.2,  'c': -50, 'd': 2},
             'FS':  {'a': 0.1,  'b': 0.2,  'c': -65, 'd': 2},
             'TC':  {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
             'RZ':  {'a': 0.1,  'b': 0.26, 'c': -60, 'd': -1},
             'LTS': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
            }

exc_cell_type = 'RS'
exc_start_v = -65
cell_params_izk_exc = {
    'a': izk_types[exc_cell_type]['a'],
    'b': izk_types[exc_cell_type]['b'],
    'c': izk_types[exc_cell_type]['c'],
    'd': izk_types[exc_cell_type]['d'],
    'v_init': exc_start_v,
    'u_init': izk_types[exc_cell_type]['b']*exc_start_v,
    'i_offset': 0.0,
    }

inh_cell_type = 'FS'
inh_start_v = -65
cell_params_izk_inh = {
    'a': izk_types[inh_cell_type]['a'],
    'b': izk_types[inh_cell_type]['b'],
    'c': izk_types[inh_cell_type]['c'],
    'd': izk_types[inh_cell_type]['d'],
    'v_init': inh_start_v,
    'u_init': izk_types[inh_cell_type]['b']*inh_start_v,
    'i_offset': 0.0,
    }


celltype = IZK_curr_exp

exc_cells = Population(nExcNeurons, celltype, cell_params_izk_exc, label="Excitatory_Cells")
inh_cells = Population(nInhNeurons, celltype, cell_params_izk_inh, label="Inhibitory_Cells")

#rngseed  = 98766987
rngseed = int(time.time())
rng = NumpyRNG(seed=rngseed)

input_type = 'const-spikes'
if input_type == 'rand-spikes':
    randSpikesBase = rng.rand(runtime)
    exc_spike_times = where(randSpikesBase > 0.6)[0].tolist()
    randSpikesBase = rng.rand(runtime)
    inh_spike_times = where(randSpikesBase > 0.4)[0].tolist()
elif input_type == 'const-spikes':
    exc_spike_times = range(runtime)
    inh_spike_times = range(runtime)

exc_spike_source = Population(1, SpikeSourceArray, {'spike_times': exc_spike_times},label='inputSpikes_exc')
inh_spike_source = Population(1, SpikeSourceArray, {'spike_times': inh_spike_times},label='inputSpikes_inh')

# poissonParameters = {
# 'duration':	runtime,
# 'start':	0.1,
# 'rate':	100.0,
# }
# spike_source = Population(1, SpikeSourcePoisson,poissonParameters, label='inputSpikes_1')

rngseed = int(time.time())
rng = NumpyRNG(seed=rngseed)

uniformWeightDistr = RandomDistribution('uniform', [0.05, 0.5], rng=rng)
exc_conn = AllToAllConnector(weights=uniformWeightDistr)
#exc_conn = FixedProbabilityConnector(0.2)

uniformWeightDistr1 = RandomDistribution('uniform', [0.0, 0.1], rng=rng)
exc_conn1 = AllToAllConnector(weights=uniformWeightDistr1, allow_self_connections=False)
#exc_conn1 = FixedProbabilityConnector(0.02, allow_self_connections=False)

uniformWeightDistrInh = RandomDistribution('uniform', [0.1, 1.], rng=rng)
inh_conn = AllToAllConnector(weights=uniformWeightDistrInh, allow_self_connections=False)
#inh_conn = FixedProbabilityConnector(0.2, allow_self_connections=False)


weightDistrSrcInh = RandomDistribution('normal', [0.5, 1.], rng=rng)
weightDistrSrcExc = RandomDistribution('normal', [0.2, 1.], rng=rng)
exc_src_conn = AllToAllConnector(weights=weightDistrSrcExc)
inh_src_conn = AllToAllConnector(weights=weightDistrSrcInh)

connections={}
connections['e2e'] = Projection(exc_cells, exc_cells, exc_conn1, target='excitatory', rng=rng)
connections['e2i'] = Projection(exc_cells, inh_cells, exc_conn, target='excitatory', rng=rng)
connections['i2e'] = Projection(inh_cells, exc_cells, inh_conn, target='inhibitory', rng=rng)
connections['i2i'] = Projection(inh_cells, inh_cells, inh_conn, target='inhibitory', rng=rng)
connections['se2e'] = Projection(exc_spike_source, exc_cells, exc_src_conn, target='excitatory')
connections['si2i'] = Projection(inh_spike_source, inh_cells, inh_src_conn, target='excitatory')
conn_tag = "-".join(connections.keys())
exc_cells.record()
inh_cells.record()


run(runtime)


# import pylab

import os

dirname = 'Izk_Results'
if not(os.path.isdir(dirname)):
    os.mkdir(dirname)

any_exc_spikes = True
any_inh_spikes = True

try:
    exc_spikes = exc_cells.getSpikes(compatible_output=True)
except IndexError:
    print("No spikes?")
    any_exc_spikes = False

try:
    inh_spikes = inh_cells.getSpikes(compatible_output=True)
except IndexError:
    print("No spikes?")
    any_inh_spikes = False



end()

#print "inhibitory spikes array  ---------------------------------"
#print inh_spikes
if any_inh_spikes == True:
    inh_time = [time for (neuron_id, time) in inh_spikes]
    inh_ids = [neuron_id + nExcNeurons + 1 for (neuron_id, time) in inh_spikes]

if any_exc_spikes == True:
    exc_time = [time for (neuron_id, time) in exc_spikes]
    exc_ids = [neuron_id for (neuron_id, time) in exc_spikes]


# filename = 'Izk_exc.txt'
# exc_results_file = open(os.path.join(dirname,filename), 'w')
# for (neuron_id, time) in exc_spikes:
#     exc_results_file.write("%d, %d \n"%(time, neuron_id))
# exc_results_file.close()

# filename = 'Izk_inh.txt'
# inh_results_file = open(os.path.join(dirname,filename), 'w')
# for (neuron_id, time) in inh_spikes:
#     inh_results_file.write("%d, %d \n"%(time, neuron_id))
# inh_results_file.close()


pylab.figure()
#pylab.subplot(3,1,1)
if any_exc_spikes == True:
    pylab.plot(exc_time, exc_ids, "b.", markersize=1)

if any_inh_spikes == True:
    pylab.plot(inh_time, inh_ids, "r.", markersize=1)

pylab.xlim([0,runtime])
pylab.ylim([0,nNeurons])
pylab.xlabel('Time/ms')
pylab.ylabel('spikes')
pylab.title('src: %s  conn: %s'%(input_type, conn_tag))

pylab.grid()

# input_spikes_exc = zeros(runtime)
# input_spikes_exc[exc_spike_times] = 1
# pylab.subplot(3,1,2)
# pylab.plot(range(runtime), input_spikes_exc, "b.", markersize=1)
# pylab.ylim([0,1.1])

# input_spikes_inh = zeros(runtime)
# input_spikes_inh[inh_spike_times] = 1
# pylab.subplot(3,1,3)
# pylab.plot(range(runtime), input_spikes_inh, "r.", markersize=1)
# pylab.ylim([0,1.1])


import time
filename = 'Izk_simple_-%s-_-%s-_fig-%s.png'%(input_type,conn_tag,time.strftime("%Y-%m-%d_%I-%M"))
fig_file = open(os.path.join(dirname,filename), 'w')
pylab.savefig(fig_file)



pylab.show()
