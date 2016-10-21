'use strict';

const opencl = require('./opencl');

const clear = (side, queue, cl_buffer, width, height) => {
	let tick = opencl.tick(side),
		kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['clear']());

	kernel.setArg(0, cl_buffer);
	kernel.setArg(1, width, 'uint');
	kernel.setArg(2, height, 'uint');

	queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size);

};


module.exports = clear;