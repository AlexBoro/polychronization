from spynnaker.pyNN.models.neural_projections.connectors.abstract_connector \
    import AbstractConnector
from spynnaker.pyNN.models.neural_properties.synaptic_list import SynapticList
from spynnaker.pyNN.models.neural_properties.synapse_row_info \
    import SynapseRowInfo
from spynnaker.pyNN.models.neural_properties.randomDistributions \
    import generate_parameter_array
from pyNN.random import RandomDistribution
import numpy
import random
from spinn_front_end_common.utilities import exceptions


class FixedNumberPreConnector(AbstractConnector):
    """
    Each post-synaptic neuron is connected to exactly n pre-synaptic neurons
    chosen at random.
    
    NOT THE SAME! Connects a fixed number of pre-synaptic neurons selected at randoom,
    to all post-synaptic neurons

    :param `int` n:
        number of random pre-synaptic neurons connected to output
    :param `bool` allow_self_connections:
        if the connector is used to connect a
        Population to itself, this flag determines whether a neuron is
        allowed to connect to itself, or only to other neurons in the
        Population.
    :param weights:
        may either be a float, a !RandomDistribution object, a list/
        1D array with at least as many items as connections to be
        created. Units nA.
    :param delays:
        If `None`, all synaptic delays will be set
        to the global minimum delay.
    :param `pyNN.Space` space:
        a Space object, needed if you wish to specify distance-
        dependent weights or delays - not implemented
    """
    def __init__(self, n, weights=0.0, delays=1,
                 allow_self_connections=True):
        """
        Creates a new FixedNumberPreConnector
        """
        if isinstance(n, (int, long, float, complex)):
          self._n_pre = int(n)
        elif isinstance(n, RandomDistribution):
          self._n_pre = n
        
        if isinstance(weights, (int, long, float, complex)):
          self._weights = float(weights)
        if isinstance(weights, RandomDistribution):
          self._weights = weights
          
        if isinstance(delays, (int, long, float, complex)): 
          self._delays = int(delays)
        elif isinstance(delays, RandomDistribution):
          self._delays = delays
        
        if isinstance(allow_self_connections, bool):
          self._allow_self_connections = allow_self_connections
        else:
          self._allow_self_connections = True
          
    def generate_synapse_list(
            self, presynaptic_population, postsynaptic_population, delay_scale,
            weight_scale, synapse_type):

        prevertex = presynaptic_population._get_vertex
        postvertex = postsynaptic_population._get_vertex
        n_post_atoms = postvertex.n_atoms
        n_pre_atoms = prevertex.n_atoms
        
        id_lists = list()
        weight_lists = list()
        delay_lists = list()
        type_lists = list()
        for _ in range(0, n_pre_atoms):
            id_lists.append(list())
            weight_lists.append(list())
            delay_lists.append(list())
            type_lists.append(list())

        if not 0 <= self._n_pre <= n_pre_atoms:
            raise exceptions.ConfigurationException(
                "Sample size has to be a number less than the size of the "
                "population but greater than zero")

        for post_atom in range(n_post_atoms):
            pre_synaptic_neurons = random.sample(range(0, n_pre_atoms),
                                                 self._n_pre)

            if (not self._allow_self_connections and
                presynaptic_population == postsynaptic_population and
                post_atom in pre_synaptic_neurons):
                pre_synaptic_neurons.remove(post_atom)

            #~ print("Connections going to POST neuron %s (%s)"%(post_atom, len(pre_synaptic_neurons)))
            #~ print(pre_synaptic_neurons)
            #~ print("----------------------------------------------------")
            
            for pre_atom in pre_synaptic_atoms:
                id_lists[pre_atom].append(post_atom)
                
                
                
        for pre_atom in range(n_pre_atoms):
            present = id_lists[pre_atom]
            n_present = len(id_lists[pre_atom])
            
            weight_lists[pre_atom] = (generate_parameter_array(
                      self._weights, n_present, present) * weight_scale)

            delay_lists[pre_atom] = (generate_parameter_array(
                        self._delays, n_present, present) * delay_scale)

            type_lists[pre_atom] = generate_parameter_array(
                                       synapse_type, n_present, present)

        connection_list = [SynapseRowInfo(id_lists[i], weight_lists[i],
                                          delay_lists[i], type_lists[i])
                           for i in range(0, n_pre_atoms)]

        return SynapticList(connection_list)
