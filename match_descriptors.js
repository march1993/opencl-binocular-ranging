'use strict'

const opencl = require('./opencl');

const match_descriptors = (queue, width, height, left_cl, left_descriptor, right_cl, right_descriptor, matches_cl) => {

	let tick = opencl.tick('both'),
	kernel = opencl.kernels['both'][tick] || (opencl.kernels['both'][tick] = opencl.kernel_makers['match_descriptors']());

	kernel.setArg(0, left_cl);
	kernel.setArg(1, left_descriptor);
	kernel.setArg(2, right_cl);
	kernel.setArg(3, right_descriptor);
	kernel.setArg(4, matches_cl);
	kernel.setArg(5, width, 'int');
	kernel.setArg(6, height, 'int');


	queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size);

};




module.exports = match_descriptors;