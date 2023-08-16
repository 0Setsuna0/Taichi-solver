import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from sphfunc import SPHSolver
ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)

if __name__ == "__main__":
    ps = ParticleSystem()
    
    ps.add_cube(lower_corner=[6, 2],
                cube_size=[3.0, 3.0],
                velocity=[-5.0, -10.0],
                density=1000.0,
                color=0x956333,
                material=1)
    
    #ps.debug_pos()
    sphsolver = SPHSolver(ps)

    gui = ti.GUI(background_color=0xFFFFFF)

    tag = False
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                vec = ti.Vector([e.pos[0], e.pos[1]]) * 10
                input_vec = [int(vec[0]), int(vec[1])]
                ps.add_cube(lower_corner=input_vec,
                cube_size=[1, 1],
                velocity=[0, -10.0],
                density=1000.0,
                color=0x956333,
                material=1)
            if e.key == ti.GUI.RMB:
                vec = ti.Vector([e.pos[0], e.pos[1]]) * 10
                input_vec = [int(vec[0]), int(vec[1])]
                sphsolver.add_collider(input_vec)
            if e.key == ti.GUI.SPACE:
                tag = not tag
        if tag==True:
            for i in range(5):
                sphsolver.step()
        pinfo = ps.positions_p.to_numpy()
        cinfo = sphsolver.dump()
        gui.circles(cinfo * 50 / 512, 
                    radius=50,
                    color=0xffaa77)
        gui.circles(pinfo * 50 / 512,
                    radius = ps.particle_radius / 1.5 * 50,
                    color = 0x33ccff)
        gui.show()
