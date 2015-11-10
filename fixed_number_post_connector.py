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


class FixedNumberPostConnector(AbstractConnector):
    """
    Each pre-synaptic neuron is connected to exactly n post-synaptic neurons
    chosen at random.
    
    If n is less than the size of the post-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If n is greater than the size of the post-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.
    """
    
    def __init__(self, n, weights=0.0, delays=1,
                 allow_self_connections=True):
        """
        Create a new connector.
        
        `n` -- either a positive integer, or a `RandomDistribution` that produces
               positive integers. If `n` is a `RandomDistribution`, then the
               number of post-synaptic neurons is drawn from this distribution
               for each pre-synaptic neuron.
        `allow_self_connections` -- if the connector is used to connect a
               Population to itself, this flag determines whether a neuron is
               allowed to connect to itself, or only to other neurons in the
               Population.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        if isinstance(n, (int, long, float, complex)):
          self._n_post = int(n)
        elif isinstance(n, RandomDistribution):
          self._n_post = n
        
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




    def generate_synapse_list(self, presynaptic_population, 
                              postsynaptic_population, delay_scale,
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

        if not 0 <= self._n_post <= n_post_atoms:
            raise exceptions.ConfigurationException(
                "Sample size has to be a number less than the size of the "
                "population but greater than zero")

        
        for pre_atom in range(n_pre_atoms):
            post_synaptic_neurons = random.sample(range(0, n_post_atoms),
                                                  self._n_post)
            
            
            if (not self._allow_self_connections and
                presynaptic_population == postsynaptic_population and
                pre_atom in post_synaptic_neurons):
                post_synaptic_neurons.remove(pre_atom) 

            n_present = len(post_synaptic_neurons)

            id_lists[pre_atom] = post_synaptic_neurons
            
            #~ print("Connections going from PRE neuron %s (%s)"%(pre_atom, n_present))
            #~ print(post_synaptic_neurons)
            #~ print("----------------------------------------------------")

            weight_lists[pre_atom] = (generate_parameter_array(
                      self._weights, n_present, post_synaptic_neurons) * weight_scale)

            delay_lists[pre_atom] = (generate_parameter_array(
                        self._delays, n_present, post_synaptic_neurons) * delay_scale)

            type_lists[pre_atom] = generate_parameter_array(
                                       synapse_type, n_present, post_synaptic_neurons)

        connection_list = [SynapseRowInfo(id_lists[i], weight_lists[i],
                                          delay_lists[i], type_lists[i])
                           for i in range(0, n_pre_atoms)]

        return SynapticList(connection_list)
