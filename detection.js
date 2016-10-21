'use strict';

const opencl = require('./opencl');
const gaussian = require('./gaussian');

const detection = (side, queue, group, to_cl_buffer, descriptor, to_width, to_height) => {
	let width = group.width,
		height = group.height;

	for (let i = 1; i < group.dogs_cl.length - 1; i++) {
		let l = group.dogs_cl[i - 1],
			m = group.dogs_cl[i],
			r = group.dogs_cl[i + 1];

		let tick = opencl.tick(side),
			kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['detection']());

		kernel.setArg(0, l);
		kernel.setArg(1, m);
		kernel.setArg(2, r);
		kernel.setArg(3, to_cl_buffer);
		kernel.setArg(4, descriptor);
		kernel.setArg(5, width, 'uint');
		kernel.setArg(6, height, 'uint');
		kernel.setArg(7, to_width, 'uint');
		kernel.setArg(8, to_height, 'uint');
		kernel.setArg(9, (group.sigma_arr[i] + group.sigma_arr[i + 1]) / 2, 'float');

		queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size);
	}
};


module.exports = detection;