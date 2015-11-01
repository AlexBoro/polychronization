import numpy
from numpy import lexsort, array, hstack
from scipy.spatial.distance import cdist
import time

### common math funcs
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, exp, fabs, \
    floor, fmod, hypot, ldexp, log, log10, modf, power, sin, sinh, sqrt, tan, tanh, \
    maximum, minimum, e, pi
    


def dist_dep_conn_3d_3d(pop3d1, pop3d2, conn_prob = 1.0, 
                        min_delay = 1.0, max_delay = 20.0,
                        max_dist = 3., weight=1.0,
                        dist_rule = "1.0/d", new_format=False):
 
  d = cdist(pop3d1, pop3d2, 'euclidean')

  d_results = eval(dist_rule)

  max_dist = eval(dist_rule, globals(), {'d': max_dist})
  
  #~ print "min", d_results.min()
  #~ print "max", d_results.max()
  #~ print "max_dist", max_dist
  
  smaller_dists = d_results > max_dist # exponential!
    
  #~ print "sum of smaller ", numpy.sum(smaller_dists)
  #~ print "num conn ", 
  if numpy.sum(smaller_dists) == 0:
    print("dist dep conn, no connections!")
    return []

  numpy.random.seed(int(time.time()))
  
  d -= numpy.abs(numpy.min(d))
  d /= numpy.max(d)/(max_delay - min_delay)
  d += min_delay

  d *= smaller_dists
  
  #numpy.random.normal(loc=max_dist, size=d.shape)
  conn_list = []
  row_count = 0
  w = 0
  delay = 0
  for row in d:
    col_count = 0
    for dist in row:
      if (pop3d1 is pop3d2) and (col_count == row_count):
        col_count += 1
        continue
        
      if dist > 0 and numpy.random.random() <= conn_prob:
        w = weight*numpy.random.random()
        delay = maximum(0., dist + min_delay*(numpy.random.random() - 0.5))
        if new_format:
          conn_list.append((row_count, col_count, 
                            weight/dist, dist))
        else:
          conn_list.append([row_count, col_count, 
                            weight/dist, dist])

      col_count += 1
    row_count += 1
  #conn_list = numpy.array(conn_list)
  return conn_list

