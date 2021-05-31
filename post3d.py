import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from PySWE import dt, xgrid, ygrid, b,h,u,v, at_time


# xgrid=np.load('xgrid.npy')
# ygrid=np.load('ygrid.npy')

# q=np.load('q.npy')
# b=np.load('b.npy')
# h=np.load('h.npy')
# u=np.load('u.npy')
# v=np.load('v.npy')

def error(a,b):
	return np.sqrt(np.mean(np.square(a-b)))


tsteps=len(at_time)
#tsteps=hanal.shape[0]
nxgrid=len(xgrid)
nygrid=len(ygrid)

fig=plt.figure()
ax=plt.axes(projection="3d")

print(tsteps)

# for t in range(0,len(w[:,0])):
# 	for i in range(0, nxgrid):
# 		w[t,i]=exp(-200*(xgrid[i]-0.25-0.5*t*dt)**2)



#ax.text(0,-80,'at time= %f sec'%(0*dt), size=10)
x,y=np.meshgrid(xgrid,ygrid)
l=1.0
b=1.3
nx=101
ny=101
x1grid=np.linspace(0,l,nx,dtype=float)
y1grid=np.linspace(0,b,ny,dtype=float)
x1,z1=np.meshgrid(x1grid, y1grid)
k=x1*0+0.5
ax.plot_surface(x,y,h[0,1:-1,1:-1],color='b',alpha=1)
ax.plot_surface(x1,k,z1,color='r',alpha=0.5)

ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
ax.set_zlim([0, 1.3])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("h(x,y)")

ncolor=plt.Rectangle((0, 0), 1, 1, fc="b")
acolor=plt.Rectangle((0, 0), 1, 1, fc="r", alpha=0.8)
ax.legend([ncolor, acolor],["h initial", "xz plane", ],loc="upper right")

plt.show()
# ax.plot_surface(x,y,h[-10,1:-1,1:-1],cmap='winter', edgecolor='none')
# plt.show()
k=5
#print(h.shape[0],hanal.shape[0])
# for t in range(0,int(tsteps/k)):
# 	for i in range(h.shape[2]):
# 		for j in range(h.shape[1]):
# 			# if h[k*t,j,i]<0.001:
# 			# 	h[k*t,j,i]=-b[j,i]

# 			if hanal[k*t,j,i]<0:
#  				hanal[k*t,j,i]=0

hanal[hanal<0]=0

#print("here",np.mean(err), np.max(err), np.min(err))

# def update(t):
# 	ax.cla()
# 	# for txt in ax.texts:
# 	# 	txt.set_visible(False)
# 	#ax.plot_surface(x,y,u[k*t,1:-1,1:-1],color='r', edgecolor='none')
# 	#ax.plot_surface(x,y,b[1:-1,1:-1],color='g', edgecolor='none',alpha=0.8)
# 	ax.plot_wireframe(x,y,h[k*t,1:-1,1:-1], color='b')
# 	ax.plot_wireframe(x,y,hanal[k*t,1:-1,1:-1] ,color='g',alpha=0.6)
# 	# ax.plot(xgrid, u[k*t,1:-1] , 'r--')
# 	#ax.plot(xgrid, b[1:-1], 'g')
# 	ax.text(0,0.1,0.12,'at time= %0.4f s'%at_time[k*t], size=10)

# 	ncolor=plt.Rectangle((0, 0), 1, 1, fc="b")
# 	acolor=plt.Rectangle((0, 0), 1, 1, fc="g", alpha=0.6)
# 	ax.legend([ncolor, acolor],["h(t,x,y) Numerical", "h(t,x,y) analytical", ],loc="upper right")
# 	ax.set_xlabel("x")
# 	ax.set_ylabel("y")
# 	ax.set_zlabel("h(x,y)")
# 	ax.set_title("Solution of h for %d*%d grid and for %d time steps(adaptive) "%(nxgrid,nxgrid,tsteps)) #time step=%0.4f s and CV size= %0.4fm"%(dt,mesh_length/nxcvele))
# 	# ax.set_ylim([0, 1.1])
# 	# ax.set_xlim([0, 1.1])
# 	ax.set_zlim([0, 0.15])

# sim= animation.FuncAnimation(fig,update, frames=range(0,int(tsteps/k)), interval=100, repeat=False)
# sim.save('2d solution.gif', writer='imagemagick')
# for t in range(0,tsteps):
# 	for i in range(hanal.shape[1]):
# 		for j in range(hanal.shape[2]):
# 			if hanal[t,j,i]<banal[j,i]:
# 				hanal[t,j,i]=-banal[j,i]
#anal
# def update(t):
# 	ax.cla()
# 	for txt in ax.texts:
# 		txt.set_visible(False)
# 	#ax.plot_surface(x,y,u[k*t,1:-1,1:-1],color='r', edgecolor='none')
# 	ax.plot_surface(x,y,hanal[t,:,:] + banal[:,:], edgecolor='none')
# 	#ax.plot_surface(x,y,banal[:,:],color='g', edgecolor='none',alpha=0.9)
# 	# ax.plot(xgrid, u[k*t,1:-1] , 'r--')
# 	#ax.plot(xgrid, b[1:-1], 'g')
# 	ax.text(0.5,0.5,0.05,'at time= %f sec'%dt*t, size=10)

# 	#hcolor=plt.Rectangle((0, 0), 1, 1, fc="b")
# 	#profilecolor=plt.Rectangle((0, 0), 1, 1, fc="g", alpha=0.4)
# 	#ax.legend([hcolor, profilecolor],["h(x) numerical", "profile", ],loc="upper right")
# 	ax.set_xlabel("x")
# 	ax.set_ylabel("y")
# 	# ax.set_title("SWE") #time step=%0.4f s and CV size= %0.4fm"%(dt,mesh_length/nxcvele))
# 	ax.set_ylim([0, 4])
# 	ax.set_xlim([0, 4])
# 	ax.set_zlim([0, 0.15])

# sim= animation.FuncAnimation(fig,update, frames=range(0,int(tsteps)), interval=50, repeat=False)
# #sim.save('send.gif', writer='imagemagick')


plt.show()

