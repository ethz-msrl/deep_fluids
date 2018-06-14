import argparse
import os
from tqdm import trange

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

parser.add_argument("--log_dir", type=str, default='data/smoke3_res5_96_f150')

parser.add_argument("--num_param", type=int, default=2)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--p0", type=str, default='res')
parser.add_argument("--p1", type=str, default='frames')

parser.add_argument("--min_res", type=float, default=32)
parser.add_argument("--max_res", type=float, default=96)
parser.add_argument("--num_res", type=int, default=5)
parser.add_argument("--src_x_pos", type=float, default=0.5)
parser.add_argument("--src_z_pos", type=float, default=0.5)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--src_radius", type=float, default=0.15)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=149)
parser.add_argument("--num_frames", type=int, default=150)
parser.add_argument("--num_simulations", type=int, default=750)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=1.0)
parser.add_argument("--clamp_mode", type=int, default=2)
parser.add_argument("--strength", type=float, default=0.00)

parser.add_argument('--is_test', type=str2bool, default=False)
parser.add_argument('--vpath', type=str, default='')

args = parser.parse_args()


def test():
	filename = os.path.basename(args.vpath)[:-4]
	print(filename)

	title = 'd_' + filename
	vdb_dir = os.path.join(os.path.dirname(args.vpath), filename)

	# solver params
	p1 = float(filename)

	# advection in highest resolution
	res_x = int(args.max_res)
	res_y = int(res_x*1.5)
	res_z = res_x
	gs = vec3(res_x, res_y, res_z)

	s = Solver(name='main', gridSize=gs, dim=3)
	s.timestep = args.time_step

	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)

	# noise field, tweak a bit for smoke source
	noise = s.create(NoiseField, loadFromFile=True)
	noise.posScale = vec3(45)
	noise.clamp = True
	noise.clampNeg = 0
	noise.clampPos = 1
	noise.valOffset = 0.75
	noise.timeAnim = 0.2

	flags.initDomain(boundaryWidth=args.bWidth)
	flags.fillGrid()
	radius = gs.x*args.src_radius
	source = s.create(Sphere, center=gs*vec3(args.src_x_pos,args.src_y_pos,args.src_x_pos), radius=radius)

	with np.load(args.vpath) as data:
		x = data['v']

	for i in range(args.num_frames):
		v = x[i,...]
		copyArrayToGridMAC(target=vel, source=v)

		# source.applyToGrid(grid=density, value=1)
		densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5)
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
		
		vdb_file_path = os.path.join(vdb_dir, title+'_%d.vdb' % i)
		density.save(vdb_file_path)	
		s.step()

def main():
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v', 'vdb'] #, 's'] #, 'p']
	for field in field_type:
		field_path = os.path.join(args.log_dir,field)
		if not os.path.exists(field_path):
			os.mkdir(field_path)

	args_file = os.path.join(args.log_dir, 'args.txt')
	with open(args_file, 'w') as f:
		for k, v in vars(args).items():
			print('  %s: %s' % (k, v))
			f.write('%s: %s\n' % (k, v))

	p_list = np.linspace(args.min_res, 
						 args.max_res,
						 args.num_res)
	pi_list = range(args.num_res)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		#gui.pause()

	print('start generation')
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	for i in trange(len(p_list), desc='scenes'):
		p = p_list[i]
		pi = pi_list[i]

		# solver params	
		res_x = int(p)
		res_y = int(res_x*1.5)
		res_z = res_x
		scale_factor = args.max_res / res_x
		print(res_x, res_y, res_z, scale_factor)
		gs = vec3(res_x, res_y, res_z)
		buoyancy = vec3(0,args.buoyancy,0)

		s = Solver(name='main', gridSize=gs, dim=3)
		s.timestep = args.time_step
		
		flags = s.create(FlagGrid)
		vel = s.create(MACGrid)
		density = s.create(RealGrid)
		pressure = s.create(RealGrid)

		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		radius = gs.x*args.src_radius
		source = s.create(Sphere, center=gs*vec3(args.src_x_pos,args.src_y_pos,args.src_x_pos), radius=radius)
		
		for t in trange(args.num_frames, desc='sim'):
			source.applyToGrid(grid=density, value=1)
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
								openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
								openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			
			if args.strength > 0:
				vorticityConfinement(vel=vel, flags=flags, strength=args.strength)

			setWallBcs(flags=flags, vel=vel)
			addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
			solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=flags, vel=vel)
			
			v_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
			copyGridToArrayMAC(target=v_, _Source=vel)

			if scale_factor > 1:
				vmin, vmax = v_.min(), v_.max()
				v_ = np.clip(scipy.ndimage.zoom(v_, [scale_factor]*3+[1], order=3), vmin, vmax)
				# print(v_.shape)

			v_range = [np.minimum(v_range[0], v_.min()),
						np.maximum(v_range[1], v_.max())]
			
			pit = (pi, t)
			param_ = [p, t]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
			np.savez_compressed(v_file_path, 
								x=v_, # yxzd 
								y=param_)

			s.step()

		if 'vdb' not in field_type: continue

		# advection in highest resolution
		res_x = int(args.max_res)
		res_y = int(res_x*1.5)
		res_z = res_x
		gs = vec3(res_x, res_y, res_z)

		s = Solver(name='main', gridSize=gs, dim=3)
		s.timestep = args.time_step
		gc.collect()
		
		flags = s.create(FlagGrid)
		vel = s.create(MACGrid)
		density = s.create(RealGrid)

		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		radius = gs.x*args.src_radius
		source = s.create(Sphere, center=gs*vec3(args.src_x_pos,args.src_y_pos,args.src_x_pos), radius=radius)
		
		for t in trange(args.num_frames, desc='advect'):
			pit = (pi, t)
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
			with np.load(v_file_path) as data:
				v = data['x']

			copyArrayToGridMAC(target=vel, source=v)
			source.applyToGrid(grid=density, value=1)				
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
								openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			
			# d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
			# copyGridToArrayReal(target=d_, source=density)

			# d_range = [np.minimum(d_range[0], d_.min()),
			# 		   np.maximum(d_range[1], d_.max())]

			# param_ = [p, t]
			# d_file_path = os.path.join(args.log_dir, 'd', args.path_format % pit)
			# np.savez_compressed(d_file_path, 
			# 					x=np.expand_dims(d_, axis=-1),
			# 					y=param_)
			
			vdb_file_path = os.path.join(args.log_dir, 'vdb', '%d_%d.vdb' % pit)
			density.save(vdb_file_path)
			s.step()
		gc.collect()

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('velocity min %.3f max %.3f' % (v_range[0], v_range[1]))
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
    if args.is_test:
        test()
    else:
        main()