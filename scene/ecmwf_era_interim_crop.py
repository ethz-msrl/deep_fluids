import argparse
from datetime import datetime
import time
import os
from glob import glob
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='../data/ecmwf_era_interim')

parser.add_argument("--num_param", type=int, default=2)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--p0", type=str, default='day')
parser.add_argument("--p1", type=str, default='time')

parser.add_argument("--num_year", type=int, default=5)
parser.add_argument("--min_year", type=float, default=2013)
parser.add_argument("--max_year", type=float, default=2017)
parser.add_argument("--year", type=int, default=2017)
parser.add_argument("--min_day", type=float, default=1)
parser.add_argument("--max_day", type=float, default=365)
parser.add_argument("--num_day", type=int, default=365)
parser.add_argument("--min_time", type=float, default=0)
parser.add_argument("--max_time", type=float, default=24)
parser.add_argument("--num_time", type=int, default=9) # 5
parser.add_argument("--num_simulations", type=int, default=1825)

parser.add_argument("--resolution_x", type=int, default=192) # 480
parser.add_argument("--resolution_y", type=int, default=96) # 240
parser.add_argument("--sx", type=int, default=150) # 150
parser.add_argument("--sy", type=int, default=24) # 24

args = parser.parse_args()

def main():
	# os.chdir(os.path.dirname(__file__)) # debug..
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v']
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

	p1_ = np.linspace(args.min_year, 
					  args.max_year,
					  args.num_year)
	p2_ = np.linspace(args.min_day,
					  args.max_day,
					  args.num_day)
	p3_ = np.linspace(args.min_time,
					  args.max_time,
				  	  args.num_time)
	
	an_path = os.path.join(args.log_dir, 'uv-2013-2017-an.npy')
	an = np.load(an_path)
	fc_paths = sorted(glob("{}/{}/*".format(args.log_dir, 'fc')))

	an_idx = 0
	fc_idx = 0	
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	for i in trange(args.num_year, desc='year'):
		p1 = p1_[i]
		for j in trange(args.num_day, desc='day'):
			p2 = p2_[j]
			if int(p1) != args.year or int(p2) == 366:
				an_idx += 2
				fc_idx += 8
				continue
			
			d_ = []
			# 0
			if j == 0:
				v = an[an_idx,:-1,:,:]				

				# # debug
				# plt.subplot(211)
				# plt.imshow(v[..., 0])
				# plt.subplot(212)
				# plt.imshow(v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, 0])
				# plt.show()

				v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]

			# use previous one
			d_.append(v)
			# 3
			d = np.load(fc_paths[fc_idx+1])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# 6
			d = np.load(fc_paths[fc_idx+2])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# 9
			d = np.load(fc_paths[fc_idx+3])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# # 12
			# v = an[an_idx+1,:-1,:,:]
			# v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			# d_.append(v)
			d = np.load(fc_paths[fc_idx])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# 15
			d = np.load(fc_paths[fc_idx+5])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# 18
			d = np.load(fc_paths[fc_idx+6])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# 21
			d = np.load(fc_paths[fc_idx+7])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)
			# # 24
			# if j < args.num_day-1:
			# 	v = an[an_idx+2,:-1,:,:]
			# 	v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			# 	d_.append(v)
			# else:
			# 	d = np.load(fc_paths[fc_idx+4])
			# 	v = d[:-1,:,:]
			# 	v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			# 	d_.append(v)
			d = np.load(fc_paths[fc_idx+4])
			v = d[:-1,:,:]
			v = v[args.sy:args.sy+args.resolution_y,args.sx:args.sx+args.resolution_x, :]
			d_.append(v)

			for k, d in enumerate(d_):
				p3 = p3_[k]
				v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (j, k))
				v_range = [np.minimum(v_range[0], d.min()),
						   np.maximum(v_range[1], d.max())]
				np.savez_compressed(v_file_path, 
									x=d[::-1,:,:],
									y=[p2, p3])

			an_idx += 2
			fc_idx += 8

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	print('Done')

if __name__ == '__main__':
    main()