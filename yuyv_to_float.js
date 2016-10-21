'use strict';

const opencl = require('./opencl');

const yuyv_to_float = (side, queue, in_buffer, in_cl_buffer, out_cl_buffer, width, height) => {
	// queue.enqueueWriteBuffer(new CLBuffer(opencl.context, defs.CL_MEM_READ_ONLY, 614400), 0, in_buffer.length, in_buffer);
	queue.enqueueWriteBuffer(in_cl_buffer, 0, in_buffer.length, in_buffer);

	let tick = opencl.tick(side);
	let kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['yuyv_to_float']());

	kernel.setArg(0, in_cl_buffer);
	kernel.setArg(1, out_cl_buffer);
	kernel.setArg(2, width, 'uint');
	kernel.setArg(3, height, 'uint');

	queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size());
};


module.exports = yuyv_to_float;