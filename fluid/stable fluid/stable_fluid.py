import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

res = 512
dt = 0.03
iteration_times = 40
damping = 1
gravity_scale = 9.8
dye_decay = 1 - 1 / 120

veclocity = ti.Vector.field(2, dtype = ti.f32, shape = (res, res))
veclocity_new = ti.Vector.field(2, dtype = ti.f32, shape = (res, res))

color = ti.Vector.field(3, dtype = ti.f32, shape = (res, res))
color_new = ti.Vector.field(3, dtype = ti.f32, shape = (res, res))

veclocity_div = ti.field(dtype = ti.f32, shape = (res, res))

pressure = ti.field(dtype = ti.f32, shape = (res, res))
pressure_new = ti.field(dtype = ti.f32, shape = (res, res))

@ti.kernel
def change_values(old:ti.template(), new:ti.template()):
    for I in ti.grouped(old):
        old[I], new[I] = new[I], old[I]

@ti.func
def sample_value(value:ti.template(), x, y):
    I = ti.Vector([int(x), int(y)])
    I = max(0, min(res - 1, I))
    return value[I]

@ti.func
def lerp(value_left:ti.template(), value_right:ti.template(), fraction:ti.template()):
    return value_left + fraction * (value_right - value_left)

@ti.func
def bilinear_lerp(value:ti.template(), pos:ti.template()):
    x = pos.x - 0.5
    y = pos.y - 0.5
    xl = ti.floor(x)
    yl = ti.floor(y)
    fx = x - xl#grid length is 1,so this equation equals (x - xl)/length
    fy = y - yl

    v1 = sample_value(value, xl, yl)#left-bottom
    v2 = sample_value(value, xl + 1, yl)#right-bottom
    v3 = sample_value(value, xl, yl + 1)#left-top
    v4 = sample_value(value, xl + 1, yl + 1)#right-top

    return lerp(lerp(v1, v2, fx), lerp(v3, v4, fx), fy)#bilinear lerp

@ti.func
def traceback(velocity:ti.template(), p:ti.template(), dt:ti.template()):#in advection step, find the last time step's position
    v1 = bilinear_lerp(veclocity, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilinear_lerp(veclocity, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilinear_lerp(veclocity, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)#RK3
    return p 

@ti.kernel
def advection(vf:ti.template(), value:ti.template(), new_value:ti.template(), dt:ti.template()):
    for i,j in vf:
        p = ti.Vector([i, j])#we define velocity at the center of each grid in this program,actually its physical position(both x-axis and y-axis) must add 0.5
        p = traceback(vf, p, dt)
        new_value[i, j] = bilinear_lerp(value, p)

@ti.kernel
def reset_velocity():
    for i,j in veclocity:
        if color[i, j].z != 1:
            veclocity[i, j] == ti.Vector([0, 0])

@ti.kernel
def apply_gravity():
    for i,j in veclocity:
        a = color[i, j].norm()
        veclocity[i, j] += gravity_scale * ti.Vector([0, -1]) * dt * 300 * a / (1 + a)

@ti.kernel
def calculate_div(vf:ti.template()):
    for i,j in vf:
        #gain related veclocity of each sampling point
        vl = sample_value(vf, i - 1, j)
        vr = sample_value(vf, i + 1, j)
        vd = sample_value(vf, i, j - 1)
        vu = sample_value(vf, i, j + 1)
        v = sample_value(vf, i, j)
        #handle boundary condition
        if i == 0:
            vl.x = -v.x
        if i == res - 1:
            vr.x = -v.x
        if j == 0:
            vd.y = -v.y
        if j == res - 1:
            vu.y = -v.y
        #calculate divergence
        veclocity_div[i, j] = (vr.x - vl.x + vu.y - vd.y) / 2

@ti.kernel
def solve_pressure(pressure:ti.template(), new_pressure:ti.template()):
    for i,j in pressure:
        pl = sample_value(pressure, i - 1, j)
        pr = sample_value(pressure, i + 1, j)
        pd = sample_value(pressure, i, j - 1)
        pu = sample_value(pressure, i, j + 1)
        div = veclocity_div[i, j]
        new_pressure[i,j] = (pl + pr + pd + pu - div) * 0.25

def jacobian_iteration():
    for i in range(iteration_times):
        solve_pressure(pressure, pressure_new)
        change_values(pressure, pressure_new)

@ti.kernel
def update_veclocity(v:ti.template(),pressure:ti.template()):
    for i,j in v:
        pl = sample_value(pressure, i - 1, j)
        pr = sample_value(pressure, i + 1, j)
        pd = sample_value(pressure, i, j - 1)
        pu = sample_value(pressure, i, j + 1)
        v[i, j] -= 0.5 * ti.Vector([pr - pl, pu - pd])
        v[i, j] *= ti.exp(-dt * 0.8)



def substep():
    #advection
    advection(veclocity, veclocity, veclocity_new, dt)
    advection(veclocity, color, color_new, dt)
    change_values(veclocity, veclocity_new)
    change_values(color, color_new)
    reset_velocity()
    #impulse
    apply_gravity()
    #projection
    calculate_div(veclocity)
    jacobian_iteration()
    change_values(pressure, pressure_new)
    update_veclocity(veclocity, pressure)

@ti.kernel
def init_gridvalues():
    veclocity.fill(0)
    veclocity_new.fill(0)
    pressure.fill(0)
    pressure_new.fill(0)
    color.fill(0)
    color_new.fill(0)
    for i in range(res // 5, res // 3):
        for j in range(res // 5, res // 3):
            color[i, j] = ti.Vector([0, 0, 1])  

@ti.kernel
def add_fluid(u:ti.template(), v:ti.template()):
    x_range = (u, u + 50)
    y_range = (v, v + 50)
    color_add = ti.Vector([ti.random(), ti.random(), ti.random()])
    for I in ti.grouped(ti.ndrange(x_range, y_range)):
        color[I] = color_add
def main():
    init_gridvalues()

    gui = ti.GUI('my first fluid solver', (res, res))

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.LMB:
                x = ti.floor(e.pos[0] * res)
                y = ti.floor(e.pos[1] * res)
                add_fluid(x, y)
            if e.key == 'r':
                init_gridvalues()
        substep()
        gui.set_image(color)
        gui.show()

if __name__ == '__main__':
    main()