import numpy


def focal_to_spike_source_array(focal_spikes, img_width, img_height, 
                                num_spikes_per_ms=5):
  '''Convert spike trains generated with FoCal algorithm into a PyNN 
     SpikeSourceArray format
     
     num_spikes_per_ms -> how many of the read spikes will ocurr per
     milisecond
  '''
  pop_size = img_width*img_height
  num_focal_spikes = len(focal_spikes['spk'])
  max_spikes = int(num_focal_spikes*0.3)
  
  spike_times = [[], [], [], []]
  for i in xrange(pop_size):
    spike_times[0].append([])
    spike_times[1].append([])
    spike_times[2].append([])
    spike_times[3].append([])
    
  spike_per_ms_count = 0
  spike_count = 0
  ms = 1.0
  for spike_data in focal_spikes['spk']:
    if spike_count > max_spikes:
      break

    neuron_id = spike_data[0]
    layer = spike_data[2]
    if spike_per_ms_count == num_spikes_per_ms:
      spike_per_ms_count = 0
      ms += 1.0
    
    spike_times[layer][neuron_id].append(ms)
    
    spike_count += 1
    spike_per_ms_count += 1
  
  return spike_times
  

def repeat_after(spikes, period_length, num_periods):
  if num_periods <= 1:
    return 
    
  for layer in xrange(len(spikes)):
    for neuron in xrange(len(spikes[layer])):
      tmp_times = spikes[layer][neuron][:]
      for period in xrange(num_periods):
        for spike_time in tmp_times:
          spikes[layer][neuron].append(spike_time + period*period_length)
  
  #return spikes

def add_spikes(original, additional, silence=0, place="after"):
  assert place in ("before", "after"), \
         "add spikes -> can't add spikes in that place (%s)"%(place)
  if silence < 0:
    silence = 0
  
  o = numpy.array(original)
  max_time = o.max()[0]

  for neuron in xrange(len(original)):
    max_neuron_time = max(original[neuron]) if len(original[neuron]) > 0 else 0
    for add_time in additional[neuron]:
      time_diff = max_time - max_neuron_time
      original[neuron].append(add_time + silence + time_diff)
  
    
  
  
def move_spikes(spikes, img_width, img_height, direction, distance):

  #~ assert direction in ("up", "down", "left", "right"), \
         #~ "move spikes -> not a valid direction"
  assert direction in ("up", "down"), \
         "move spikes -> not a valid direction (%s)"%(direction)
  
  if direction == "up":
    delta = -img_width*distance
  elif direction == "down":
    delta = img_width*distance
  #~ elif direction == "left":
    #~ delta = -distance
  #~ elif direction == "right":
    #~ delta = distance
  #~ 
  num_neurons = img_width*img_height
  new_spikes = []
  for idx in xrange(num_neurons):
    new_spikes.append([])
    
  neuron_idx = 0
  for neuron in spikes:
    if len(neuron) > 0:
      new_idx = neuron_count + delta
      if new_idx < num_neurons and new_idx > 0:
        new_spikes[new_idx] = spikes[neuron_idx][:]
      
    neuron_idx += 1
  
  
