try:
	from manta import *
	import gc
except ImportError:
	pass

import argparse
import os
from glob import glob
import queue
import threading
import signal
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default='data/liquid3_d5_r10_f150/final/test/')
parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=48)
parser.add_argument("--resolution_z", type=int, default=96)
parser.add_argument("--num_worker", type=int, default=16)
args = parser.parse_args()

class Param(object):
	pass

def meshing(m):
	m.pp.load(m.pt_path)

	# create surface
	gridParticleIndex(parts=m.pp, flags=m.flags, indexSys=m.pindex, index=m.gpi)
	averagedParticleLevelset(m.pp, m.pindex, m.flags, m.gpi, m.phi, 1, 1, 1) 

	m.phi.setBound(1, boundaryWidth=1)
	m.phi.createMesh(m.mesh)

	for _ in range(5):
		smoothMesh(mesh=m.mesh, strength=1e-3, steps=10) 
		subdivideMesh(mesh=m.mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=False)
	
	dirname = os.path.join(os.path.dirname(m.pt_path), '..')
	basename = os.path.basename(m.pt_path)[:-3]
	obj_file_path = os.path.join(dirname, basename+'obj')
	m.mesh.save(obj_file_path)

def set_manta():
	m = Param()

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z
	gs = vec3(res_x, res_y, res_z)

	s = Solver(name='main', gridSize=gs, dim=3)

	m.flags = s.create(FlagGrid)
	m.phi = s.create(LevelsetGrid)
	m.pp = s.create(BasicParticleSystem) 
	m.mesh = s.create(Mesh)

	# acceleration data for particle nbs
	m.pindex = s.create(ParticleIndexSystem) 
	m.gpi = s.create(IntGrid)

	# scene setup
	m.flags.initDomain(boundaryWidth=1)
	return m

def worker(q):
	m = set_manta()
	while True:
		pt_path = q.get()
		if pt_path is None:
			break

		m.pt_path = pt_path
		meshing(m)
		q.task_done()
		print('done:', pt_path)
	print('close thread')

def main():
	pt_dir = os.path.join(args.log_dir, 'pt')
	pt_paths = os.path.join(pt_dir, '*.uni')
	pt_paths = glob(pt_paths)

	q = queue.Queue()
	threads = [threading.Thread(target=worker, 
								args=(q,)
								) for i in range(args.num_worker)]

	for t in threads:
		t.start()

	# define signal handler
	def signal_handler(signum, frame):
		print('%s: canceled by SIGINT' % datetime.now())
		for t in threads:
			q.put(None)

		for t in threads:
			t.join()
		sys.exit(1)
	signal.signal(signal.SIGINT, signal_handler)

	# feeding
	for pt_path in pt_paths:
		q.put(pt_path)
	
	# block until all tasks are done
	q.join()

	for t in threads:
		q.put(None)

	for t in threads:
		t.join()

if __name__ == '__main__':
    main()