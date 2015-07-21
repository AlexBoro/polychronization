"""
the simple model paper simulation
"""
#!/usr/bin/python
from pyNN.spiNNaker import *
import pylab
from numpy import where
import time

#rngseed  = 98766987
rngseed = int(time.time())

setup(timestep=1, min_delay = 1.0, max_delay = 32.0)
set_number_of_neurons_per_core("IZK_curr_exp", 70)

nNeurons = 3 # number of neurons in each population
excToInhRatio = 4.
nExcNeurons = int(round((nNeurons*excToInhRatio/(1+excToInhRatio))))
nInhNeurons = nNeurons - nExcNeurons
runtime = 1000
# cell_params_izk_exc = {
#     'a': 0.02,
#     'b': 0.2,
#     'c': -65,
#     'd': 8,
#     'v_init': -65,
#     'u_init': 0.2*(-65),
#     'i_offset': 5.0,
#     }

# cell_params_izk_inh = {
#     'a': 0.1,
#     'b': 0.2,
#     'c': -65,
#     'd': 2,
#     'v_init': -65,
#     'u_init': 0.2*(-65),
#     'i_offset': 2.0,
#     }


izk_types = {'RS':  {'a': 0.02, 'b': 0.2,  'c': -65, 'd': 8},
             'IB':  {'a': 0.02, 'b': 0.2,  'c': -55, 'd': 4},
             'CH':  {'a': 0.02, 'b': 0.2,  'c': -50, 'd': 2},
             'FS':  {'a': 0.1,  'b': 0.2,  'c': -65, 'd': 2},
             'TC':  {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
             'RZ':  {'a': 0.1,  'b': 0.26, 'c': -65, 'd': 2},
             'LTS': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
            }

cell_type = 'RZ'
start_v = -65
cell_params_izk = {
    'a': izk_types[cell_type]['a'],
    'b': izk_types[cell_type]['b'],
    'c': izk_types[cell_type]['c'],
    'd': izk_types[cell_type]['d'],
    'v_init': start_v,
    'u_init': izk_types[cell_type]['a']*start_v,
    'i_offset': 0.0,
    }


celltype = IZK_curr_exp

exc_cells = Population(nExcNeurons, celltype, cell_params_izk, label="Excitatory_Cells")
#inh_cells = Population(nInhNeurons, celltype, cell_params_izk_inh, label="Inhibitory_Cells")


rng = NumpyRNG(seed=rngseed)

input_type = 'const-spikes'
if input_type == 'rand-spikes':
    randSpikesBase = rng.rand(runtime)
    spike_times = where(randSpikesBase > 0.4)[0].tolist()
elif input_type == 'const-spikes':
    spike_times = range(10, runtime)

spikeArray = {'spike_times': spike_times}
spike_source = Population(1, SpikeSourceArray, spikeArray,label='inputSpikes_1')

# poissonParameters = {
# 'duration':	runtime,
# 'start':	0.1,
# 'rate':	100.0,
# }
# spike_source = Population(1, SpikeSourcePoisson,poissonParameters, label='inputSpikes_1')



#uniformWeightDistr = RandomDistribution('uniform', [0., 0.5], rng=rng)
#exc_conn = AllToAllConnector(weights=uniformWeightDistr)
uniformWeightDistr = RandomDistribution('uniform', [0.5, 1.], rng=rng)
#inh_conn = AllToAllConnector(weights=uniformWeightDistrInh)

connections={}
#connections['e2e'] = Projection(exc_cells, exc_cells, exc_conn, target='excitatory', rng=rng)
#connections['e2i'] = Projection(exc_cells, inh_cells, exc_conn, target='excitatory', rng=rng)
#connections['i2e'] = Projection(inh_cells, exc_cells, inh_conn, target='inhibitory', rng=rng)
#connections['i2i'] = Projection(inh_cells, inh_cells, inh_conn, target='inhibitory', rng=rng)
connections['s2e'] = Projection(spike_source, exc_cells, AllToAllConnector(weights=uniformWeightDistr), target='excitatory')
#connections['s2i'] = Projection(spike_source, inh_cells, AllToAllConnector(weights=0.6))

exc_cells.record_v()
#inh_cells.record()


run(runtime)


# import pylab

import os

dirname = 'Izk_Results'
if not(os.path.isdir(dirname)):
    os.mkdir(dirname)

any_exc_spikes = True
any_inh_spikes = True

try:
    exc_voltages = exc_cells.get_v(compatible_output=True)
except IndexError:
    print("No voltage?")
    any_exc_spikes = False

# try:
#     inh_spikes = inh_cells.getSpikes(compatible_output=True)
# except IndexError:
#     print("No spikes?")
#     any_inh_spikes = False



end()


# if any_inh_spikes == True:
#     inh_time = [time for (neuron_id, time) in inh_spikes]
#     inh_ids = [neuron_id + nExcNeurons + 1 for (neuron_id, time) in inh_spikes]

if any_exc_spikes == True:
#     exc_time = [time for (neuron_id, time, voltage) in exc_voltages if neuron_id == 0]
#     exc_v = [voltage for (neuron_id, time, voltage) in exc_voltages if neuron_id == 0]
#     exc_ids = [neuron_id for (neuron_id, time, voltage) in exc_voltages if neuron_id == 0]
    exc_time = [time for (neuron_id, time, voltage) in exc_voltages]
    exc_v = [voltage for (neuron_id, time, voltage) in exc_voltages]
    exc_ids = [neuron_id for (neuron_id, time, voltage) in exc_voltages]


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
pylab.xlabel('Time/ms')
if any_exc_spikes == True:
    numSubplots = int(round(nNeurons/1))
    for i in range(nNeurons):
        start_idx = i*runtime
        end_idx = start_idx + runtime
        pylab.subplot(numSubplots, 1, int(round(i/10)))
        pylab.plot(exc_time[start_idx:end_idx], exc_v[start_idx:end_idx], markersize=1)
        pylab.ylabel('volts')
# if any_inh_spikes == True:
#     pylab.plot(inh_time, inh_ids, "r.", markersize=1)

pylab.xlim([0,runtime])
#pylab.ylim([0,nNeurons])


pylab.title('voltage')

pylab.grid()

import time
filename = 'Izk_simple-%s-voltage_%s-input_fig-%s.png'%(cell_type, input_type, time.strftime("%Y-%m-%d_%I-%M"))
fig_file = open(os.path.join(dirname,filename), 'w')
pylab.savefig(fig_file)
pylab.show()


