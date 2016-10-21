'use strict';

const opencl = require('./opencl');

const float_to_int = (side, queue, in_cl_buffer, out_buffer, out_cl_buffer, width, height) => {

	let tick = opencl.tick(side);
	let kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['float_to_int']());

	kernel.setArg(0, in_cl_buffer);
	kernel.setArg(1, out_cl_buffer);
	kernel.setArg(2, width, 'uint');
	kernel.setArg(3, height, 'uint');

	queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size());

	return queue.waitable(true).enqueueReadBuffer(out_cl_buffer, 0, out_buffer.length, out_buffer);
};


module.exports = float_to_int;