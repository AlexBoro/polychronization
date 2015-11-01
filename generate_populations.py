import numpy
from numpy import random, sqrt, arange, meshgrid, linspace, dstack
from pyNN.space import Grid2D, Grid3D, Cuboid, BaseStructure
import time


def rand_pop_3d_coords(num_neurons, 
                       space_width=1.0, space_height=1.0, space_depth=1.0,
                       x0=0.0, y0=0.0, z0=0.0,
                       rng=random):

  numpy.random.seed(int(time.time()))

  max_dist = numpy.array([space_width, space_height, space_depth])*0.5
  origin = numpy.array([x0, y0, z0])
  xyz = origin + rng.uniform(-1, 1, size=(num_neurons,3)) * max_dist

#  return numpy.array((xyz[:,0], xyz[:,1], xyz[:,2]))
  return xyz


def grid2d_coords(num_neurons, x0=0.0, y0=0.0):

  width = int(sqrt(num_neurons))
  
  if num_neurons%width != 0:
    print "grid 2d --- not all neurons will be used?"
  
  height = num_neurons/width
  
  x_range = linspace(-1, 1, width)*width*0.5
  y_range = linspace(-1, 1, height)*height*0.5
  
  x, y = meshgrid(x_range, y_range)
  
  x += x0
  y += y0
  
  xy = dstack((x, y))
  
  return xy.reshape((xy.shape[0]*xy.shape[1], xy.shape[2]))




class myCuboid(BaseStructure):
  """
  Represents a structure with neurons distributed on a 3D grid.
  """
  
  parameter_names = ("width", "height", "depth", "rng", "aspect_ratios", 
                     "dx", "dy", "dz", "x0", "y0", "z0", "fill_order")

  def __init__(self, width, height, depth, rng=numpy.random, 
                aspect_ratioXY=1.0, aspect_ratioXZ=1.0, 
                dx=1.0, dy=1.0, dz=1.0, x0=0.0, y0=0.0, z0=0,
                fill_order="sequential"):
    """
    If fill_order is 'sequential', the z-index will be filled first, then y then x, i.e.
    the first cell will be at (0,0,0) (given default values for the other arguments),
    the second at (0,0,1), etc.
    """
    self.aspect_ratios = (aspect_ratioXY, aspect_ratioXZ)
    assert fill_order in ('sequential', 'random')
    self.fill_order = fill_order
    self.dx = dx; self.dy = dy; self.dz = dz
    self.x0 = x0; self.y0 = y0; self.z0 = z0

    self.width = width
    self.height = height
    self.depth = depth
    self.rng = rng
      
  def calculate_size(self, n):
    #~ a,b = self.aspect_ratios
    sq_size = int(n/self.depth)
    nx = int(sqrt(sq_size))
    ny = int(sqrt(sq_size))
    #~ nx = int(round(power(n*a*b, 1/3.0)))
    #~ ny = int(round(nx/a))
    #~ nz = int(round(nx/b))
    #~ assert nx*ny*nz == n, str((nx, ny, nz, nx*ny*nz, n, a, b))
    #~ return nx, ny, nz
    return self.width, self.height, self.depth

  def generate_positions(self, n):
    nx, ny, nz = self.calculate_size(n)
    x,y,z = numpy.indices((nx,ny,nz), dtype=float)
    x = self.x0 + self.dx*x.flatten()
    y = self.y0 + self.dy*y.flatten()
    z = self.z0 + self.dz*z.flatten()
    if self.fill_order == 'sequential':
      return numpy.array((x,y,z))
    else:
      numpy.random.seed(int(time.time()))

      max_distance = numpy.array((self.width, self.height, self.depth))
      xyz = numpy.array((self.x0, self.y0, self.z0))
      xyz += 0.5*rng.uniform(-1, 1, size=(n,3)) * max_distance

      return numpy.array((xyz[:,0], xyz[:,1], xyz[:,2]))
