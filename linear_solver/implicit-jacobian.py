import taichi as ti
ti.init()

max_num_particles=256

dt=1e-3#dt

num_particles=ti.field(ti.i32,shape=())
spring_stiffness=ti.field(ti.f32,shape=())
paused=ti.field(ti.i32,shape=())
damping=ti.field(ti.f32,shape=())#velocity damping

particle_mass=1
bottom_y=0.05


x=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)#position field
v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)#velocity field
new_v=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)

dv=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)

M=ti.Matrix.field(2,2,dtype=ti.f32,shape=(max_num_particles,max_num_particles))#mass matrix
A=ti.Matrix.field(2,2,dtype=ti.f32,shape=(max_num_particles,max_num_particles))#A=M-dt^2*J
b=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)#b=dt*F
J=ti.Matrix.field(2,2,dtype=ti.f32,shape=(max_num_particles,max_num_particles))#partial-f/partial-x
F=ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)#force of each point

rest_length=ti.field(ti.f32,shape=(max_num_particles,max_num_particles))#original length between two points,if two points are not connected,rest_length[i,j]=0
#usually use field to describe physical quantitieso of a system

connection_radius=0.15

gravity = [0, -9.8]


@ti.kernel
def jacobian_iterate():#jacobian method
    n=num_particles[None]
    for i in range(n):
        r=b[i]
        for j in range(n):
            if i!=j:
                r-=A[i,j]@v[j]#@matrix multiply
        new_v[i] = A[i, i].inverse() @ r
    for i in range(n):
        v[i]=new_v[i]

@ti.kernel
def update_M_matrix():
    m=ti.Matrix([[particle_mass,0],
                [0,particle_mass]])
    for i in range(num_particles[None]):
        M[i,i]=m#dia(m,m,m,m...)
@ti.kernel
def update_J_matrix():#df/dx,F=-spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
    I=ti.Matrix([[1,0],
                [0,1]])
    for i, d in J:
        J[i, d] *= 0.0
        for j in range(num_particles[None]):
            r_length=rest_length[i,j]
            if(r_length!=0 and (d==i or d==j)):
                x_ij=x[i]-x[j]
                x_ijdis = x_ij.norm()
                x_ijdir=x_ij/x_ijdis
                xMxtrans=x_ijdir.outer_product(x_ijdir)
                if d == i:
                    J[i, d] += -spring_stiffness[None] * (I - r_length / x_ijdis * (I - xMxtrans))
                else:
                    J[i, d] += spring_stiffness[None] * (I - r_length / x_ijdis * (I - xMxtrans))       
@ti.kernel
def update_A_matrix(b:ti.f32):#use b to do different integration,b equals 1:implicit,b equals 0.5:semi-implicit,b equals 0:explicit
    for i,j in A:
        A[i,j]=M[i,j]-b*dt*dt*J[i,j]
@ti.kernel
def update_F_vector():
    for i in range(num_particles[None]):
        F[i]=ti.Vector(gravity)*particle_mass
        for j in range(num_particles[None]):
            r=rest_length[i,j]
            if r!=0:
               x_ij=x[i]-x[j]
               F[i]+= -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
@ti.kernel
def update_b_vector():
    for i in range(num_particles[None]):
        v[i]*=ti.exp(-dt*damping[None])
        b[i]=M[i,i]@v[i]+dt*F[i]

@ti.kernel
def update_mass_spring_system():
    for i in range(num_particles[None]):
        x[i]+=v[i]*dt
        if x[i].y<bottom_y:
            x[i].y=bottom_y
            v[i].y=0

def substepimplicit(b:ti.f32,times):#implicit integration
    update_M_matrix()
    update_J_matrix()
    update_A_matrix(b)
    update_F_vector()
    update_b_vector()
    for i in range(times):
        jacobian_iterate()
    update_mass_spring_system()

@ti.kernel
def substep():#semi-implicit eular
    n=num_particles[None]
    for i in range(n):子
        v[i]*=ti.exp(-dt*damping[None])
        total_force=ti.Vector(gravity)*particle_mass
        for j in range(n):
            if rest_length[i,j]!=0:
                x_ij=x[i]-x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        v[i]+=dt*total_force/particle_mass

    for i in range(n):
        if x[i].y<bottom_y:
            x[i].y=bottom_y
            v[i].y=0

    for i in range(num_particles[None]):
        x[i]+=v[i]*dt

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
            substepimplicit(1,10)
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)
    
    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.show()
