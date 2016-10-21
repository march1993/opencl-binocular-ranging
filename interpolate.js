'use strict';

const opencl = require('./opencl');

const interpolate = (side, queue, in_cl_buffer, out_cl_buffer, in_width, in_height) => {

	let tick = opencl.tick(side);
	let kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['interpolate']());

	kernel.setArg(0, in_cl_buffer);
	kernel.setArg(1, out_cl_buffer);
	kernel.setArg(2, in_width, 'uint');
	kernel.setArg(3, in_height, 'uint');
	kernel.setArg(4, in_width * 2, 'uint');
	kernel.setArg(5, in_height * 2, 'uint');

	queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(in_width * 2, in_height * 2), opencl.get_local_size);
};


module.exports = interpolate;