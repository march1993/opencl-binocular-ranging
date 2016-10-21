'use strict';
const opencl = require('./opencl');

const template_gaussian = (side, queue, group, SIGMA_IN_LAYER) => {

	let width = group.width,
		height = group.height;

	SIGMA_IN_LAYER.forEach((sigma, index) => {
		let tick = opencl.tick(side);
		let kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['template_gaussian_' + sigma.toString().replace('.', '_')]());

		kernel.setArg(0, group['layers_cl'][0]);
		kernel.setArg(1, group['layers_cl'][1 + index]);
		kernel.setArg(2, width, 'size_t');
		kernel.setArg(3, height, 'size_t');

		queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size());
	});

};




module.exports = template_gaussian;

