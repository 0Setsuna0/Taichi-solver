import taichi as ti
import numpy as np
from functools import reduce


@ti.data_oriented
class ParticleSystem:
    def __init__(self):
        #particle 
        self.num_particle = ti.field(ti.i32, shape=())
        self.particle_max_num = 2 ** 15
        self.particle_neighbor_max_num = 100
        self.particle_grid_max_num = 100
        
        #material enum
        self.fluid_m = 1
        self.boundary_m = 0

        #particle attribute
        self.density_p = ti.field(ti.f32)
        self.pressure = ti.field(ti.f32)
        self.color = ti.field(ti.i32)
        self.velocities_p = ti.Vector.field(2, ti.f32)
        self.positions_p = ti.Vector.field(2, ti.f32)
        self.material = ti.field(ti.i32)
        self.m_V = 0.8 * 0.1 ** 2
        
        #grid
        self.grid_particles_num = ti.field(ti.i32)
        self.grid_particles = ti.field(ti.i32)
        self.grid_num = [2560, 2560]
        self.grid_spacing = 0.2

        #neighbor
        self.particle_radius = 0.05
        self.particle_diameter = 2.0 * self.particle_radius
        self.neighbor_radius = 4.0 * self.particle_radius
        self.particle_neighbors = ti.field(ti.i32)
        self.particle_neighbors_num = ti.field(ti.i32)

        #particle root
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.positions_p, self.velocities_p, self.density_p, self.pressure, self.material, self.color)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_neighbor_max_num)
        self.particle_node.place(self.particle_neighbors)

        #grid root
        grid_node = ti.root.dense(ti.ij, self.grid_num)
        grid_node.place(self.grid_particles_num)
        cell_node = grid_node.dense(ti.k, self.particle_grid_max_num)
        cell_node.place(self.grid_particles)
    
    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.positions_p[p] = x
        self.velocities_p[p] = v
        self.density_p[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.ext_arr(),
                      new_particles_velocity: ti.ext_arr(),
                      new_particle_density: ti.ext_arr(),
                      new_particle_pressure: ti.ext_arr(),
                      new_particles_material: ti.ext_arr(),
                      new_particles_color: ti.ext_arr()):
        for p in range(self.num_particle[None], self.num_particle[None] + new_particles_num):
            v = ti.Vector([0, 0])
            x = ti.Vector.zero(float, 2)
            for d in ti.static(range(2)):
                v[d] = new_particles_velocity[p - self.num_particle[None], d]
                x[d] = new_particles_positions[p - self.num_particle[None], d]
            self.add_particle(p, x, v,
                              new_particle_density[p - self.num_particle[None]],
                              new_particle_pressure[p - self.num_particle[None]],
                              new_particles_material[p - self.num_particle[None]],
                              new_particles_color[p - self.num_particle[None]])
        self.num_particle[None] += new_particles_num
    
    
    
    @ti.kernel
    def reset_system(self):
        for i in self.num_particle[None]:
            self.velocities_p[i] = ti.Vector([0,0])

    #get the grid index of a given particle
    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_spacing).cast(int)
    
    #check wheather the related_grid_index is in the grid boundary
    @ti.func
    def r_pos_inboundary(self, r_index):
        inside = True
        for i in ti.static(range(2)):
            inside = inside and (0 <= r_index[i] < self.grid_num[i])
        return inside

    @ti.kernel
    def search_neighbor_particles(self):
        for i in range(self.num_particle[None]):
            if self.material[i] == 0:
                continue
            center_index_i = self.pos_to_index(self.positions_p[i])
            pn_num = 0
            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                r_grid_index = center_index_i + offset
                if not self.r_pos_inboundary(r_grid_index):
                    break
                for j in range(self.grid_particles_num[r_grid_index]):
                    #get the particle index(the particle in the related grid)
                    particle_j= self.grid_particles[r_grid_index, j] #r_grid_index对应grid的第j个粒子在粒子场中的索引为particle_j
                    #calculate the distance between the center particle and the neighbor grid's particle
                    dis = (self.positions_p[i] - self.positions_p[particle_j]).norm()
                    if i!= particle_j and dis < self.neighbor_radius:
                        self.particle_neighbors[i, pn_num] = particle_j #第i个粒子的第pn_num个neighbor的索引为particle_j
                        pn_num += 1
            self.particle_neighbors_num[i] = pn_num 

    @ti.kernel
    def put_particle_to_grid(self):
        for i in range(self.num_particle[None]):
            cell = self.pos_to_index(self.positions_p[i])
            offset = self.grid_particles_num[cell].atomic_add(1)
            self.grid_particles[cell, offset] = i
    
    @ti.kernel
    def to_render(self):
        np_pos = np.ndarray((self.num_particle[None], 2), dtype=np.float32)
        for i in range(self.num_particle[None]):
            for j in ti.static(range(2)):
                np_pos[i, j] = self.positions_p[i][j]
        return np_pos
    

    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.num_particle[None]):
            for j in ti.static(range(2)):
                np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def debug_pos(self):
        for i in range(self.num_particle[None] / 10):
            print(self.positions_p[i])

    def dump(self):
        np_x = np.ndarray((self.num_particle[None], 2), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.positions_p)
        return np_x
    

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(2):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)



    def reset_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.put_particle_to_grid()
        self.search_neighbor_particles()