'use strict';

const opencl = require('./opencl');

/**
 *	in_image should be 4x larger(means 2x larger both in width and height) than out_image
 */
const downsample = (side, queue, in_cl_buffer, out_cl_buffer, out_width, out_height) => {

	let tick = opencl.tick(side);
	let kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['downsample']());

	kernel.setArg(0, in_cl_buffer);
	kernel.setArg(1, out_cl_buffer);
	kernel.setArg(2, out_width * 2, 'uint');
	kernel.setArg(3, out_height * 2, 'uint');
	kernel.setArg(4, out_width, 'uint');
	kernel.setArg(5, out_height, 'uint');

	queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(out_width, out_height), opencl.get_local_size());
};


module.exports = downsample;