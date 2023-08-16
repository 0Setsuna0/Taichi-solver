import taichi as ti
ti.init(ti.cuda)

#constent
Grid_Num_X = 256
Grid_Num_Y = 256

G = ti.Vector([0, -9.8])
