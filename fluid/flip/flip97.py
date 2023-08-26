import taichi as ti
import numpy as np
import random
ti.init(ti.cuda)

#constent
Grid_Num_X = 256
Grid_Num_Y = 256

G = -9.8
EMPTY = 0
FLUID = 1
SOLID = 2

flip_coef = 0.97
rho = 1000
dt = 0.01

width = 10
height = 10
gridSpacing_x = width / Grid_Num_X
gridSpacing_y = height / Grid_Num_Y

#data defined on grid, grid resolution = Grid_Num_X x Grid_Num_Y
#data origin (dx, dy)
pressure_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y))

#u-data origin (0, dy/2), y-data origin (dx/2, 0)
u_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X + 1, Grid_Num_Y))
v_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y + 1))
u_temp_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X + 1, Grid_Num_Y))
v_temp_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y + 1))
u_last_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X + 1, Grid_Num_Y))
v_last_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y + 1))

#p2g
weight_u_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X + 1, Grid_Num_Y))
weight_v_g = ti.field(dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y + 1))

#grid material 
grid_label = ti.field(dtype=ti.i32, shape=(Grid_Num_X, Grid_Num_Y))
m, n = Grid_Num_X, Grid_Num_X
npar = 2
# particle x and v
particle_positions = ti.Vector.field(2, dtype=ti.f32, shape=(Grid_Num_Y, Grid_Num_X, npar, npar))
particle_velocities = ti.Vector.field(2, dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y, npar, npar))

# particle type
particle_type = ti.field(dtype=ti.f32, shape=(Grid_Num_X, Grid_Num_Y, npar, npar))
P_FLUID = 1
P_OTHER = 0

gui = ti.GUI("flip-pic-97", (720, 720))
result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                framerate=24,
                                automatic_build=False)

@ti.data_oriented
class CGPressureSolver:
    #reference:
    def __init__(self, m, n, u, v, max_iteration, label):
        self.m = m
        self.n = n
        self.u = u
        self.v = v
        self.max_iteration = max_iteration
        self.label = label
        self.shape = (m, n)

        #build up linear system to solve the pressure such that div field = 0
        #reference:Fluid Simulation for Compute Graphics
        self.b = ti.field(dtype = ti.f32, shape = (m, n))
        self.Adiag = ti.field(dtype = ti.f32, shape = (m, n))
        self.Ax = ti.field(dtype = ti.f32, shape = (m, n))
        self.Ay = ti.field(dtype = ti.f32, shape = (m, n))

        #helper cg attribute
        self.p = ti.field(dtype = ti.f32, shape = (m, n))
        self.r = ti.field(dtype = ti.f32, shape = (m, n))
        self.d = ti.field(dtype = ti.f32, shape = (m, n))
        self.q = ti.field(dtype = ti.f32, shape = (m, n))
        self.alpha = ti.field(dtype = ti.f32, shape = ())
        self.beta = ti.field(dtype = ti.f32, shape = ())
        self.sum = ti.field(dtype = ti.f32, shape = ())
        
    @ti.kernel
    def calc_divergence_b(self, dx:ti.f32):
        scale = 1 / dx
        #apply fuild region influence
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                self.b[i, j] = -1 * scale * (self.u[i + 1, j] - self.u[i, j] \
                                              + self.v[i, j + 1] - self.v[i, j])
    
    @ti.kernel
    def calc_solid_b(self, dx:ti.f32):
        scale = 1 / dx
        #modifying the right-hand side to account for solid velocities
        #in book, code uses u_solid to modify rhs, we simply make it to be zero here
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                if self.label[i - 1, j] == SOLID:
                    self.b[i, j] -= scale * (self.u[i, j] - 0)
                if self.label[i + 1, j] == SOLID:
                    self.b[i, j] += scale * (self.u[i + 1, j] - 0)

                if self.label[i, j - 1] == SOLID:
                    self.b[i, j] -= scale * (self.v[i, j] - 0)
                if self.label[i, j + 1] == SOLID:
                    self.b[i, j] += scale * (self.v[i, j + 1] - 0)

    @ti.kernel
    def calc_lhs_a(self, rho:ti.f32, dx:ti.f32, dt:ti.f32):
        scale = dt / (rho * dx * dx)
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                #handle negative x neighbor
                if self.label[i - 1, j] == FLUID:
                    self.Adiag[i, j] += scale
                #handle positive x neighbor
                if self.label[i + 1, j] == FLUID:
                    self.Adiag[i, j] += scale
                    self.Ax[i, j] = -scale
                elif self.label[i + 1, j] == EMPTY:
                    self.Adiag[i, j] += scale

                #handle negative y neighbor
                if self.label[i, j - 1] == FLUID:
                    self.Adiag[i, j] += scale
                #handle positive y neighbor
                if self.label[i, j + 1] == FLUID:
                    self.Adiag[i, j] += scale
                    self.Ay[i, j] = - scale
                elif self.label[i, j + 1] == EMPTY:
                    self.Adiag[i, j] += scale

    def init_solver(self, rho, dx, dt):
        self.b.fill(0.0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)

        #init lhs and rhs
        self.calc_divergence_b(dx)
        self.calc_solid_b(dx)
        self.calc_lhs_a(rho, dx, dt)
    
    @ti.kernel
    def reduction(self, p:ti.template(), q:ti.template())->ti.f32:
        sum = 0.0
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                sum += p[i, j] * q[i, j]
        return sum

    @ti.kernel
    def compute_q(self):
        # q = Ad
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                self.q[i, j] = self.Adiag[i, j] * self.d[i, j] + self.Ax[
                    i - 1, j] * self.d[i - 1, j] + self.Ax[i, j] * self.d[
                        i + 1, j] + self.Ay[i, j - 1] * self.d[
                            i, j - 1] + self.Ay[i, j] * self.d[i, j + 1]

        
    @ti.kernel
    def iterate_p(self):
        # p = p + alpha*d
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                self.p[i, j] = self.p[i, j] + self.alpha[None] * self.d[i, j]

    @ti.kernel
    def iterate_r(self):
        # r = r - alpha*q
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                self.r[i, j] = self.r[i, j] - self.alpha[None] * self.q[i, j]

    @ti.kernel
    def iterate_d(self):
        # d = r + bera * d
        for i, j in ti.ndrange(self.m, self.n):
            if self.label[i, j] == FLUID:
                self.d[i, j] = self.r[i, j] + self.beta[None] * self.d[i, j]

    def solve_pressure(self):
        self.p.fill(0)
        self.q.fill(0)
        self.d.fill(0)
        self.r.copy_from(self.b)

        tol = 1e-12
        rTr_0 = self.reduction(self.r, self.r) #rTr

        if rTr_0 < tol:
            print("input condition is already convergence")
        else:
            self.d.copy_from(self.r)
            new_rTr = rTr_0
            iteration_num = 0

            for i in range(self.max_iteration):
                #q = Ad
                self.compute_q() 
                
                #alpha = new_rTr / dTq
                dTq = self.reduction(self.d, self.q) 
                self.alpha[None] = new_rTr / dTq

                #p = p + alpha * d
                self.iterate_p()

                #r = r - alpha * q
                self.iterate_r()

                old_rTr = new_rTr
                rTr = self.reduction(self.r, self.r)             
                if rTr < rTr_0 * tol:
                    break
                
                new_rTr = rTr
                self.beta[None] = new_rTr / old_rTr

                #d = r + beta * d
                self.iterate_d()
                iteration_num += 1

cg_pressure_solver = CGPressureSolver(Grid_Num_X, Grid_Num_Y, u_g, v_g, 400, grid_label)

#clamp the input data to the range of (x,y)
@ti.func
def clamp(data, x, y):
    return max(x, min(data, y))

@ti.func
def lerp(data1, data2, s):
    return data1 * (1 - s) + data2 * s

@ti.func
def bilinear_lerp(data1, data2, data3, data4, s, t):
    return lerp(lerp(data1, data2, s), lerp(data3, data4, s), t)

@ti.func
def pos_to_idx(pos, origin_x, origin_y):
    gird_idx_x = clamp(pos[0], origin_x * gridSpacing_x, width - 1e-4 - origin_x * gridSpacing_x)
    grid_idx_y = clamp(pos[1], origin_y * gridSpacing_y, height - 1e-4 - origin_y * gridSpacing_y) 
    
    grid_pos = ti.Vector([gird_idx_x / gridSpacing_x, grid_idx_y / gridSpacing_y]) 
    return grid_pos

@ti.func
def sample_scaler(data, idx, origin_x, origin_y, grid_x, grid_y):
    #tricky is to find out the left-bottom sample point
    offset_x, offset_y = idx[0] - origin_x, idx[1] - origin_y
    i0 = int(clamp(ti.floor(offset_x), 0, grid_x - 1))
    j0 = int(clamp(ti.floor(offset_y), 0, grid_y - 1))
    i1 = int(clamp(i0 + 1, 0, grid_x - 1))
    j1 = int(clamp(j0 + 1, 0, grid_y - 1))
    s, t = offset_x - i0, offset_y - j0
    
    return bilinear_lerp(data[i0, j0], data[i1, j0], data[i0, j1], data[i1, j1], s, t)

@ti.func
def sample_vel(pos, u, v):
    #tansfer the sampledn point's position into grid position
    grid_pos_u = pos_to_idx(pos, 0, 0.5)
    grid_pos_v = pos_to_idx(pos, 0.5, 0)

    vel_x = sample_scaler(u, grid_pos_u, 0, 0.5, Grid_Num_X + 1, Grid_Num_Y)
    vel_y = sample_scaler(v, grid_pos_v, 0.5, 0, Grid_Num_X, Grid_Num_Y + 1)

    return ti.Vector([vel_x, vel_y])

#helper func
@ti.func 
def is_fluid(i, j):
    return FLUID == grid_label[i, j] and i >= 0 and i < Grid_Num_X and j >= 0 and j < Grid_Num_Y

@ti.func 
def is_empty(i, j):
    return EMPTY == grid_label[i, j] and i >= 0 and i < Grid_Num_X and j >= 0 and j < Grid_Num_Y

@ti.func 
def is_solid(i, j):
    return SOLID == grid_label[i, j] and i >= 0 and i < Grid_Num_X and j >= 0 and j < Grid_Num_Y

#simulation step
@ti.kernel
def apply_gravity(dt:ti.f32):
    for I in ti.grouped(v_g):
        v_g[I] += G * dt
 
@ti.kernel
def apply_boundary_condition():
    for i, j in grid_label:
        if is_solid(i, j):
            u_g[i, j] = 0.0
            u_g[i + 1, j] = 0.0
            v_g[i, j] = 0.0
            v_g[i, j + 1] = 0.0

@ti.kernel
def apply_pressure_force(dt:ti.f32):
    #reference: Fluid Simulation for Computer Graphics
    scale = dt / (rho * gridSpacing_x)
    for i, j in ti.ndrange(Grid_Num_X, Grid_Num_Y):
        if is_fluid(i - 1, j) or is_fluid(i, j):
            if is_solid(i - 1, j) or is_solid(i, j):
                #sample point is between fluid and solid, simply make the velocity to be zero
                u_g[i, j] = 0.0
            else:
                u_g[i, j] -= scale * (pressure_g[i, j] - pressure_g[i - 1, j])

    for i, j in ti.ndrange(Grid_Num_X, Grid_Num_Y):
        if is_fluid(i, j - 1) or is_fluid(i, j):
            if is_solid(i, j - 1)  or is_solid(i, j):
                v_g[i, j] = 0.0
            else:
                v_g[i, j] -= scale * (pressure_g[i, j] - pressure_g[i, j - 1])

@ti.pyfunc
def vec2(x, y):
    return ti.Vector([x, y])

@ti.func
def g2p(v_g, v_g_last, pos, origin_x, origin_y):
    #tricky is also to find out the left bottom sample point base
    xp = pos_to_idx(pos, origin_x, origin_y) - ti.Vector([origin_x, origin_y])
    base = int(xp)
    fx = xp - base
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

    pic = 0.0
    flip = 0.0

    for offset in ti.static(ti.ndrange(3,3)):
        weight = 1.0
        for i in ti.static(range(2)):
            weight *= w[offset[i]][i]
        pic += weight * v_g[base + offset] #pic is gaining the new velocity from grid to particle
        flip += weight * (v_g[base + offset] - v_g_last[base + offset])#flip is gaining the difference of new and old vel field from grid to particle

    return pic, flip

@ti.kernel
def grid_to_particle():
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pic_u, flip_u = g2p(u_g, u_last_g, particle_positions[p], 0, 0.5)
            pic_v, flip_v = g2p(v_g, v_last_g, particle_positions[p], 0.5, 0)
            pic = ti.Vector([pic_u, pic_v])
            flip = particle_velocities[p] + ti.Vector([flip_u, flip_v])

            particle_velocities[p] = flip_coef * flip + (1 - flip_coef) * pic

@ti.func
def p2g(v_g, w_g, pos, vel, origin_x, origin_y):
    inv_dx = vec2(1.0 / gridSpacing_x, 1.0 / gridSpacing_y).cast(ti.f32)
    base = (pos * inv_dx - (ti.Vector([origin_x, origin_y]) + 0.5)).cast(ti.i32)
    fx = pos * inv_dx - (base.cast(ti.f32) + ti.Vector([origin_x, origin_y]))

    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

    for offset in ti.static(ti.ndrange(3, 3)):
        weight = 1.0
        for i in ti.static(range(2)):
            weight *= w[offset[i]][i]
        
        v_g[base + offset] += weight * vel
        w_g[base + offset] += weight

@ti.kernel
def particle_to_grid():
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            p2g(u_g, weight_u_g, particle_positions[p], particle_velocities[p][0], 0.0, 0.5)
            p2g(v_g, weight_v_g, particle_positions[p], particle_velocities[p][1], 0.5, 0.0)
    
    for I in ti.grouped(u_g):
        if weight_u_g[I] > 0:
            u_g[I] = u_g[I] / weight_u_g[I]
    
    for I in ti.grouped(v_g):
        if weight_v_g[I] > 0:
            v_g[I] = v_g[I] / weight_v_g[I]

@ti.kernel
def advect_particles(dt: ti.f32):
    for p in ti.grouped(particle_positions):
        if particle_type[p] == P_FLUID:
            pos = particle_positions[p]
            pv = particle_velocities[p]

            pos += pv * dt

            if pos[0] <= gridSpacing_x:  # left boundary
                        pos[0] = gridSpacing_x + 1e-3
                        pv[0] = 0
            if pos[0] >= width - gridSpacing_x:  # right boundary
                        pos[0] = width - gridSpacing_x - 1e-3
                        pv[0] = 0
            if pos[1] <= gridSpacing_y:  # bottom boundary
                        pos[1] = gridSpacing_y + 1e-3
                        pv[1] = 0
            if pos[1] >= height - gridSpacing_y:  # top boundary
                        pos[1] = height - gridSpacing_y - 1e-3
                        pv[1] = 0

            particle_positions[p] = pos
            particle_velocities[p] = pv 

@ti.kernel
def update_label():
    for i, j in grid_label:
        if not is_solid(i, j):
            grid_label[i, j] = EMPTY

    for i, j, ix, jx in particle_positions:
        if particle_type[i, j, ix, jx] == P_FLUID:
            pos = particle_positions[i, j, ix, jx]
            idx = ti.cast(ti.floor(pos / vec2(gridSpacing_x, gridSpacing_y)), ti.i32)

            if not is_solid(idx[0], idx[1]):
                grid_label[idx] = FLUID

#tricky func
def compute_pressure():
    cg_pressure_solver.init_solver(rho, gridSpacing_x, dt)
    cg_pressure_solver.solve_pressure()

    pressure_g.copy_from(cg_pressure_solver.p)

def step():
    #update grid vel
    apply_gravity(dt)
    apply_boundary_condition()

    compute_pressure()
    apply_pressure_force(dt)
    apply_boundary_condition()

    #g2p
    grid_to_particle()
    advect_particles(dt)
    update_label()

    #reset grid attrib
    u_g.fill(0.0)
    v_g.fill(0.0)
    weight_u_g.fill(0.0)
    weight_v_g.fill(0.0)
    pressure_g.fill(0.0)
    
    #p2g
    particle_to_grid()

    u_last_g.copy_from(u_g)
    v_last_g.copy_from(v_g)

@ti.kernel
def init_dambreak(x: ti.f32, y: ti.f32):
    xn = int(x / gridSpacing_x)
    yn = int(y / gridSpacing_y)

    for i, j in grid_label:
        if i <= 0 or i == m - 1 or j == 0 or j == n - 1:
            grid_label[i, j] = SOLID  # boundary
        else:
            if (i <= xn / 2 and j <= yn) or (i >= m - xn / 2 and j <= yn):
                grid_label[i, j] = FLUID
            else:
                grid_label[i, j] = EMPTY

@ti.kernel
def init_spherefall(xc: ti.f32, yc: ti.f32, r: ti.f32):
    for i, j in grid_label:
        if i <= 0 or i == m - 1 or j == 0 or j == n - 1:
            grid_label[i, j] = SOLID  # boundary
        else:
            x = (i + 0.5) * gridSpacing_x
            y = (j + 0.5) * gridSpacing_x

            phi = (x - xc)**2 + (y - yc) ** 2 - r**2

            if phi <= 0 :
                grid_label[i, j] = FLUID
            else:
                grid_label[i, j] = EMPTY

#init simulation
@ti.kernel
def init_field():
    u_g.fill(0.0)
    v_g.fill(0.0)
    u_last_g.fill(0.0)
    v_last_g.fill(0.0)
    pressure_g.fill(0.0)


@ti.kernel
def init_particles():
    for i, j, ix, jx in particle_positions:
        if grid_label[i, j] == FLUID:
            particle_type[i, j, ix, jx] = P_FLUID
        else:
            particle_type[i, j, ix, jx] = 0

        px = i * gridSpacing_x + (ix) * gridSpacing_x / 2
        py = j * gridSpacing_x + (jx) * gridSpacing_x / 2

        particle_positions[i, j, ix, jx] = vec2(px, py)
        particle_velocities[i, j, ix, jx] = vec2(0.0, 0.0)

def init():
    init_dambreak(4, 4)
    init_field()
    init_particles()

def draw(frame):
    clear_color = 0x808080
    particle_color = 0x87ceeb
    particle_radius = 1.3

    pf = particle_type.to_numpy()
    np_type = pf.copy()
    np_type = np.reshape(np_type, -1)

    pos = particle_positions.to_numpy()
    np_pos = pos.copy()
    np_pos = np.reshape(pos, (-1, 2))
    np_pos = np_pos[np.where(np_type == P_FLUID)]

    for i in range(np_pos.shape[0]):
        np_pos[i][0] /= width
        np_pos[i][1] /= height

    gui.clear(clear_color)
    gui.circles(np_pos, radius=particle_radius, color=particle_color)
    
    filename = f'{frame:04d}.png'   # create filename with suffix png
    print(f'Frame {frame} is recorded in {filename}')
    gui.show()  # export and show in GUI


def simulate():
    max_frame = 200
    frame_count = 0
    
    while frame_count < max_frame:
        draw(frame_count)
        for i in range(4):
            step()
        frame_count += 1


def main():
    init()
    simulate()

    print("Exporting mp4 and gif")
    video_manager.make_video(gif=True, mp4=True)



if __name__ == "__main__":
    main()