import argparse
import sys
from datetime import datetime
import time
import os

import numpy as np
import matplotlib.pyplot as plt

import gc
try:
	from manta import *
except ImportError:
	pass

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_jet_f600_256')

parser.add_argument("--num_param", type=int, default=1)
parser.add_argument("--path_format", type=str, default='%d.npz')
parser.add_argument("--p0", type=str, default='frames')

num_sim = 600
parser.add_argument("--src_inflow", type=float, default=15) # 5 for 64x128
parser.add_argument("--src_buoyancy", type=float, default=-10e-4)
parser.add_argument("--src_x_pos", type=float, default=0.1)
parser.add_argument("--src_y_pos", type=float, default=0.25)
parser.add_argument("--src_z_pos", type=float, default=0.5)
parser.add_argument("--src_radius", type=float, default=0.04)
parser.add_argument("--src_height", type=float, default=0.1)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_sim-1)
parser.add_argument("--num_frames", type=int, default=num_sim)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=512)
parser.add_argument("--resolution_y", type=int, default=256)
parser.add_argument("--resolution_z", type=int, default=1)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=str, default='XyY')
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument('--data_gen', type=str2bool, default=True)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()


def test():
	dist_path = os.path.join(args.log_dir, 'distance.npz')
	wl2_map = np.load(dist_path)['wl2']
	# l2_map = np.load(dist_path)['l2']
	# l1_map = np.load(dist_path)['l1']
	# wl1_map = np.load(dist_path)['wl1']

	# fig = plt.figure()
	# ax = fig.add_subplot(321)
	# ax.set_title('l2')
	# ax.matshow(l2_map)

	# ax = fig.add_subplot(322)
	# ax.set_title('l1')
	# cax = ax.matshow(l1_map)
	# fig.colorbar(cax)

	# ax = fig.add_subplot(323)
	# ax.set_title('wl2')
	# ax.matshow(wl2_map)

	# ax = fig.add_subplot(324)
	# ax.set_title('wl1')
	# ax.matshow(wl1_map)

	# ax = fig.add_subplot(313)
	# ax.set_title('wl2 plot')

	# for i in range(600):
	# 	ax.plot(range(600), wl2_map[i,:])

	# plt.show()

	# np.fill_diagonal(wl2_map, 1000)
	# print('smallest', np.amin(wl2_map), np.argmin(wl2_map))
	
	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z
	gs = vec3(res_x, res_y, res_z)
	
	if res_z > 1:
		s = Solver(name='main', gridSize=gs, dim=3)
	else:
		s = Solver(name='main', gridSize=gs, dim=2)
	s.frameLength = 1.0
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)

	# stream function
	# omega = s.create(VecGrid)
	# stream = s.create(VecGrid)
	# vel_out = s.create(MACGrid)

	# noise field, tweak a bit for smoke source
	noise = s.create(NoiseField, loadFromFile=True)
	noise.posScale = vec3(45)
	noise.clamp = True
	noise.clampNeg = 0
	noise.clampPos = 1
	noise.valOffset = 0.75
	noise.timeAnim = 0.2

	if res_z > 1:
		v_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
		# d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
		# s_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
		# p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
	else:
		v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
		# d_ = np.zeros([res_y,res_x], dtype=np.float32)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()

		screen_dir = os.path.join(args.log_dir, 'img')
		if not os.path.exists(screen_dir):
			os.mkdir(screen_dir)
		#gui.pause()


	start_time = time.time()
		
	density.clear()
	vel.clear()
	# stream.clear()

	flags.initDomain(boundaryWidth=args.bWidth)
	flags.fillGrid()
	setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)
	
	src_center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
	src_radius = args.resolution_y*args.src_radius
	src_z = gs*vec3(args.src_height,0,0)
	source = s.create(Cylinder, center=src_center, radius=src_radius, z=src_z)

	#############################
	# # normal loading
	# for t in range(args.num_frames):
	# 	file_path = os.path.join(args.log_dir, 'v', args.path_format % t)
	# 	print(file_path)
	# 	v = np.load(file_path)['x']
	# 	v = np.dstack((v,np.zeros([args.resolution_y, args.resolution_x, 1])))
	# 	copyArrayToGridMAC(target=vel, source=v)

	# 	densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5)
	# 	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
	# 						openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
	# 	resetOutflow(flags=flags, real=density)
	# 	s.step()
	# exit(1)
	#############################

	t = 0
	intp_intv = 20.0
	last_spoint = 569
	search_spoint = 150
	pivot = np.random.randint(low=last_spoint, high=600-intp_intv)
	d_last = wl2_map[pivot,search_spoint:pivot-30]
	loop_id = search_spoint + np.argmin(d_last)
	print('pivot', pivot, 'closest loop id', loop_id)

	from scipy import interpolate

	i = 0
	while True:
		file_path = os.path.join(args.log_dir, 'v', args.path_format % t)
		print(file_path)
		v = np.load(file_path)['x']
		v = np.dstack((v,np.zeros([args.resolution_y, args.resolution_x, 1])))

		if t > pivot and t < pivot+intp_intv:
			dt = (t-pivot)
			t2 = loop_id + dt
			w_blend = dt/intp_intv
			print(t, t2, dt, w_blend)
			
			file_path = os.path.join(args.log_dir, 'v', args.path_format % t2)
			print('v2', file_path)
			v2 = np.load(file_path)['x']
			v2 = np.dstack((v2,np.zeros([args.resolution_y, args.resolution_x, 1])))

			v = np.stack((v, v2), axis=0)
			x = np.arange(v.shape[0])
			f_intp = interpolate.interp1d(x, v, axis=0, kind='slinear')
			v = f_intp(w_blend)

			if t+1 == pivot+intp_intv:
				t = t2
				pivot = np.random.randint(low=last_spoint, high=600-intp_intv)
				d_last = wl2_map[pivot,search_spoint:pivot-30]
				loop_id = search_spoint + np.argmin(d_last)
				print('pivot', pivot, 'closest loop id', loop_id)

		copyArrayToGridMAC(target=vel, source=v)

		densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5)
		# source.applyToGrid(grid=density, value=1)
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
		resetOutflow(flags=flags, real=density)
		s.step()

		gui.screenshot(os.path.join(screen_dir,'%04d_%04d.png' % (i,t)))
		t = t + 1
		i = i + 1
		
		# if t == args.num_frames:
		# 	t = loop_id


def compute_distance():
	from glob import glob
	file_list = glob(os.path.join(args.log_dir, 'v/*.npz'))
	file_list = sorted(file_list, key=lambda x: int(os.path.basename(x).split('.')[0]))
	num_files = len(file_list)
	assert num_files == args.num_frames

	l2_map = np.zeros([num_files, num_files])
	l1_map = np.zeros([num_files, num_files])
	wl2_map = np.zeros([num_files, num_files])
	wl1_map = np.zeros([num_files, num_files])
	for i, fp1 in enumerate(file_list):
		v1 = np.load(fp1)['x'][::-1,...]

		dudx = v1[1:,1:,0] - v1[1:,:-1,0]
		dvdx = v1[1:,1:,1] - v1[1:,:-1,1]
		dudy = v1[:-1,:-1,0] - v1[1:,:-1,0] # horizontally flipped
		dvdy = v1[:-1,:-1,1] - v1[1:,:-1,1] # horizontally flipped
		dudx = dudx[2:-2,2:-2]
		dvdx = dvdx[2:-2,2:-2]
		dudy = dudy[2:-2,2:-2]
		dvdy = dvdy[2:-2,2:-2]
		w1 = dvdx - dudy
		# plt.imshow(w1, cmap=plt.cm.RdBu)
		# plt.show()
		print(i)
		for j, fp2 in enumerate(file_list):
			v2 = np.load(fp2)['x'][::-1,...]

			dudx = v2[1:,1:,0] - v2[1:,:-1,0]
			dvdx = v2[1:,1:,1] - v2[1:,:-1,1]
			dudy = v2[:-1,:-1,0] - v2[1:,:-1,0] # horizontally flipped
			dvdy = v2[:-1,:-1,1] - v2[1:,:-1,1] # horizontally flipped
			dudx = dudx[2:-2,2:-2]
			dvdx = dvdx[2:-2,2:-2]
			dudy = dudy[2:-2,2:-2]
			dvdy = dvdy[2:-2,2:-2]
			w2 = dvdx - dudy
			
			l2 = np.mean((v1-v2)**2)
			l1 = np.mean(abs(v1-v2))

			wl2 = np.mean((w1-w2)**2)
			wl1 = np.mean(abs(w1-w2))			
			
			l2_map[i,j] = l2
			l1_map[i,j] = l1
			wl2_map[i,j] = wl2
			wl1_map[i,j] = wl1
			# print(i,j, l2, l1)
		# break

	dist_path = os.path.join(args.log_dir, 'distance.npz')
	np.savez_compressed(dist_path,
		l2=l2_map, l1=l1_map,
		wl2=wl2_map, wl1=wl1_map)

	# fig = plt.figure()
	# ax = fig.add_subplot(321)
	# ax.set_title('l2')
	# ax.matshow(l2_map)

	# ax = fig.add_subplot(322)
	# ax.set_title('l1')
	# cax = ax.matshow(l1_map)
	# fig.colorbar(cax)

	# ax = fig.add_subplot(323)
	# ax.set_title('wl2')
	# ax.matshow(wl2_map)

	# ax = fig.add_subplot(324)
	# ax.set_title('wl1')
	# ax.matshow(wl1_map)

	# ax = fig.add_subplot(313)
	# ax.set_title('wl2 plot')

	# for i in range(num_files):
	# 	ax.plot(range(num_files), wl2_map[i,:])

	# plt.show()

def generate():
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v'] #, 'vdb'] #, 's'] #, 'p']
	for field in field_type:
		field_path = os.path.join(args.log_dir,field)
		if not os.path.exists(field_path):
			os.mkdir(field_path)

	args_file = os.path.join(args.log_dir, 'args.txt')
	with open(args_file, 'w') as f:
		print('%s: arguments' % datetime.now())
		for k, v in vars(args).items():
			print('  %s: %s' % (k, v))
			f.write('%s: %s\n' % (k, v))

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z
	gs = vec3(res_x, res_y, res_z)
	
	if res_z > 1:
		s = Solver(name='main', gridSize=gs, dim=3)
	else:
		s = Solver(name='main', gridSize=gs, dim=2)
	s.frameLength = 1.0
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)
	pressure = s.create(RealGrid)

	# stream function
	# omega = s.create(VecGrid)
	# stream = s.create(VecGrid)
	# vel_out = s.create(MACGrid)

	# noise field, tweak a bit for smoke source
	noise = s.create(NoiseField, loadFromFile=True)
	noise.posScale = vec3(45)
	noise.clamp = True
	noise.clampNeg = 0
	noise.clampPos = 1
	noise.valOffset = 0.75
	noise.timeAnim = 0.2

	if res_z > 1:
		v_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
		# d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
		# s_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
		# p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
	else:
		v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
		# d_ = np.zeros([res_y,res_x], dtype=np.float32)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		#gui.pause()

	print('start generation')
	sim_id = 0
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	start_time = time.time()
		
	density.clear()
	vel.clear()
	pressure.clear()
	# stream.clear()

	flags.initDomain(boundaryWidth=args.bWidth)
	flags.fillGrid()
	setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)
	
	src_center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
	src_radius = args.resolution_y*args.src_radius
	src_z = gs*vec3(args.src_height,0,0)
	source = s.create(Cylinder, center=src_center, radius=src_radius, z=src_z)

	inflow = vec3(args.src_inflow,0,0)
	buoyancy = vec3(0,args.src_buoyancy,0)

	for t in range(args.num_frames):	
		print('%s: simulating %d of %d' % (datetime.now(), t, args.num_frames))
		densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5)
		# source.applyToGrid(grid=density, value=1)
		source.applyToGrid(grid=vel, value=inflow)
			
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
							openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode, strength=1.0)
		resetOutflow(flags=flags, real=density)
		setWallBcs(flags=flags, vel=vel)
		addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
		solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
		
		# # get streamfunction
		# curl1(vel, omega)
		# solve_stream_function_pcg(flags, omega, stream)
		# curl2(stream, vel)

		# param_ = p.tolist() + [t]
		param_ = [t]

		copyGridToArrayMAC(target=v_, _Source=vel)
		# copyGridToArrayReal(target=d_, source=density)
		# copyGridToArrayVec3(target=s_, source=stream)
		# copyGridToArrayReal(target=p_, source=pressure)

		v_range = [np.minimum(v_range[0], v_.min()),
					np.maximum(v_range[1], v_.max())]
		# d_range = [np.minimum(d_range[0], d_.min()),
		# 		   np.maximum(d_range[1], d_.max())]
		# s_range = [np.minimum(s_range[0], s_.min()),
		# 		   np.maximum(s_range[1], s_.max())]
		# p_range = [np.minimum(p_range[0], p_.min()),
		# 		   np.maximum(p_range[1], p_.max())]

		
		# pit = tuple(pi.tolist() + [t])
		pit = t

		v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
		np.savez_compressed(v_file_path,
							x=v_[:,:,:2], # x=v_, # yxzd for 3d
							y=param_)

		# d_file_path = os.path.join(args.log_dir, 'd', args.path_format % pit)
		# np.savez_compressed(d_file_path, 
		# 					x=np.expand_dims(d_, axis=-1),
		# 					y=param_)

		# vdb_file_path = os.path.join(args.log_dir, 'vdb', '%.2e_%.2e_%d.vdb' % tuple(param_))
		# density.save(vdb_file_path)

		# s_file_path = os.path.join(args.log_dir, 's', args.path_format % pit)
		# np.savez_compressed(s_file_path, 
		# 					x=s_, # yxzd
		# 					y=param_)

		# p_file_path = os.path.join(args.log_dir, 'p', args.path_format % pit)
		# np.savez_compressed(p_file_path, 
		# 					x=np.expand_dims(p_, axis=-1),
		# 					y=param_)

		s.step()

	gc.collect()
	duration = time.time() - start_time
	print('%s: done (%.3f sec)' % (datetime.now(), duration))

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# drange_file = os.path.join(args.log_dir, 'd_range.txt')
	# with open(drange_file, 'w') as f:
	# 	print('%s: density min %.3f max %.3f' % (datetime.now(), d_range[0], d_range[1]))
	# 	f.write('%.3f\n' % d_range[0])
	# 	f.write('%.3f' % d_range[1])

	# srange_file = os.path.join(args.log_dir, 's_range.txt')
	# with open(srange_file, 'w') as f:
	# 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
	# 	f.write('%.3f\n' % s_range[0])
	# 	f.write('%.3f' % s_range[1])

	# prange_file = os.path.join(args.log_dir, 'p_range.txt')
	# with open(prange_file, 'w') as f:
	# 	print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
	# 	f.write('%.3f\n' % p_range[0])
	# 	f.write('%.3f' % p_range[1])

	print('Done')


if __name__ == '__main__':
	np.random.seed(args.seed)

	if args.data_gen:
		generate()
		compute_distance()
	else:
		test()