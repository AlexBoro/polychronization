from generate_populations import *
from generate_connections import *
from spike_adjustments import *
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
from numpy import lexsort, array, hstack



def plot_conn_list(conn_list, pop1, pop2, idx, ax, c='grey', alpha=0.5):
  for conn in conn_list:
    if conn[0] == idx:
      #print conn 
      ax.plot([pop1[idx][0], pop2[conn[1]][0]],
              [pop1[idx][1], pop2[conn[1]][1]],
              [pop1[idx][2], pop2[conn[1]][2]],
              c=c, alpha=alpha)

new_conn_list = True

width  = 28
height = 28
depth  = 6
sp_depth = 20

num_2d = width*height
num_3d = width*height*depth*2

pop_2d = grid2d_coords(num_2d)
pop_3d = rand_pop_3d_coords(num_3d, 
                            space_width=width*1.1, 
                            space_height=height*1.1, 
                            space_depth=depth)

num_inh = int(num_3d*0.2)
num_exc = int(num_3d*0.8)
inh_3d = pop_3d[:num_inh]
exc_3d = pop_3d[num_inh:]

min_z = (pop_3d[:,2].min() - 0.1)*numpy.ones((pop_2d.shape[0], 1))
pop_2d_3d = hstack((pop_2d, min_z))

in2exc = dist_dep_conn_3d_3d(pop_2d_3d, exc_3d, dist_rule = "exp(-(d**2))",
                             max_dist=3., conn_prob = 0.5, 
                             new_format=new_conn_list)
#in2inh = dist_dep_conn_3d_3d(pop_2d_3d, inh_3d, dist_rule = "exp(-d)")

inh2exc = dist_dep_conn_3d_3d(inh_3d, exc_3d, dist_rule = "exp(-(d**2))", 
                              conn_prob = 0.5,
                              new_format=new_conn_list)
inh2inh = dist_dep_conn_3d_3d(inh_3d, inh_3d, dist_rule = "exp(-(d**2))",
                              conn_prob = 0.2)

exc2exc = dist_dep_conn_3d_3d(exc_3d, exc_3d, dist_rule = "exp(-sqrt(d*d))",
                              conn_prob = 0.1, max_dist = 10.0)
                              
exc2inh = dist_dep_conn_3d_3d(exc_3d, inh_3d, dist_rule = "exp(-sqrt(d*d))",
                              conn_prob = 0.2)



fig = pylab.figure()
ax = fig.add_subplot(111, projection='3d')

plot_conn_list(in2exc, pop_2d_3d, exc_3d, 0, ax, alpha=0.7, c='pink')
plot_conn_list(in2exc, pop_2d_3d, exc_3d, 1, ax, alpha=0.7, c='orange')
plot_conn_list(in2exc, pop_2d_3d, exc_3d, 2, ax, alpha=0.7, c='green')
#~ plot_conn_list(inh2exc, inh_3d, exc_3d, 0, ax, alpha=0.7, c='orange')
#~ plot_conn_list(exc2exc, exc_3d, exc_3d, 0, ax, alpha=0.7, c='green')

ax.scatter(inh_3d[:,0], inh_3d[:,1], inh_3d[:,2], c='r', marker='.')
ax.scatter(exc_3d[:,0], exc_3d[:,1], exc_3d[:,2], c='b', marker='.')

ax.scatter(pop_2d_3d[:,0], pop_2d_3d[:,1], pop_2d_3d[:,2],
           c='c', marker='.')



ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pylab.show()
