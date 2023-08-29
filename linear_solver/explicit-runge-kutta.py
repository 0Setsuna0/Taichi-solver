import taichi as ti
ti.init(arch=ti.cpu)

max_num_particles=256

dt=1e-3#dt,时间间隔，用Python中的微小量表示

num_particles=ti.field(ti.i32,shape=())#使用空场，定义在taichi scope和Python scope中都可以使用的全局变量
spring_stiffness=ti.field(ti.f32,shape=())#劲度系数
paused=ti.field(ti.i32,shape=())
damping=ti.field(ti.f32,shape=())#速度衰减系数

particle_mass=1#弹簧质点质量
bottom_y=0.05


x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)#弹簧质点的位置的向量场
v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)#弹簧质点速度的向量场
new_v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)

rk_v1 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
rk_v2 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
rk_v3 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
rk_v4 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
rk_pos = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)

dv=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)

rest_length=ti.field(ti.f32,shape=(max_num_particles,max_num_particles))#原长度，如果为零，表示两个顶点没有连接
#注意到，每当我们要用到一个质点系统的某些变量时，都可以使用一个场来表示

connection_radius=0.15

gravity = [0, -9.8]


#rk method
@ti.kernel
def equalRK_XV():
    for i in range(num_particles[None]):
        rk_v2[i] = v[i]
        rk_v3[i] = v[i]
        rk_v4[i] = v[i]
        rk_pos[i] = x[i]
@ti.kernel 
def updateRK_v(particles_pos:ti.template(), rkv_old:ti.template(),rkv_new:ti.template(), t:ti.template()):
    for i in range (num_particles[None]):
        particles_pos[i] = x[i]
    for i in range(num_particles[None]):
        particles_pos[i] += rkv_old[i] * t
        force = ti.Vector(gravity) * particle_mass
        for j in range(num_particles[None]):
            r = rest_length[i, j]
            if r != 0:
                x_ij = particles_pos[i] - particles_pos[j]
                force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        rkv_new[i] += force / particle_mass * t
@ti.kernel
def updateRK_pos():
    for i in range(num_particles[None]):
        x[i] += (v[i] + 2 * rk_v2[i] + 2 * rk_v3[i] + rk_v4[i]) / 6 * dt
        v[i] = (v[i] + 2 * rk_v2[i] + 2 * rk_v3[i] + rk_v4[i]) / 6
        v[i] *= ti.exp(-dt*damping[None])
        if x[i].y<bottom_y:
            x[i].y=bottom_y
            v[i].y=0

def substepRK():
    equalRK_XV()
    updateRK_v(rk_pos, v, rk_v2, 0.5 * dt)
    updateRK_v(rk_pos, rk_v2, rk_v3, 0.5 * dt)
    updateRK_v(rk_pos, rk_v3, rk_v4, dt)
    updateRK_pos()

@ti.kernel
def new_particle(pos_x: ti.f32,pos_y:ti.f32):#用来新建一个点的
    new_particle_id=num_particles[None]#
    x[new_particle_id]=[pos_x,pos_y]
    v[new_particle_id]=[0,0]
    num_particles[None]+=1

    for i in range(new_particle_id):
        dist=(x[new_particle_id]-x[i]).norm()
        if dist<connection_radius:
            rest_length[i,new_particle_id]=0.1
            rest_length[new_particle_id,i]=0.1

gui=ti.GUI("Mass Spring System",res=(512,512),background_color=0xdddddd)

spring_stiffness[None]=10000
damping[None]=20

new_particle(0.3,0.3)
new_particle(0.3,0.4)
new_particle(0.4,0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE,ti.GUI.EXIT]:
            exit()
        elif e.key==gui.SPACE:
            paused[None]=not paused[None]
        elif e.key==ti.GUI.LMB:#left mouse button
            new_particle(e.pos[0], e.pos[1])
        elif e.key=="c":#格式化
            num_particles[None]=0
            rest_length.fill(0)
        elif e.key=='s':
            if gui.is_pressed('Shift'):
                spring_stiffness[None]/=1.1
            else:
                spring_stiffness[None]*=1.1
        elif e.key=='d':
            if gui.is_pressed('Shift'):
                damping[None]/=1.1
            else:
                damping[None]*=1.1
    if not paused[None]:
        for step in range(10):
            substepRK()
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)
    
    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.show()

