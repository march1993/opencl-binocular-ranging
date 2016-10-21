'use strict';

const opencl = require('./opencl');

const dog = (side, queue, group) => {
	let width = group.width,
		height = group.height;

	for (let i = 0; i < group.dogs_cl.length; i++) {
		let l1 = group.layers_cl[2 + i],
			l2 = group.layers_cl[2 + i + 1];
		let to = group.dogs_cl[i];

		let tick = opencl.tick(side),
			kernel = opencl.kernels[side][tick] || (opencl.kernels[side][tick] = opencl.kernel_makers['dog']());

		kernel.setArg(0, l1);
		kernel.setArg(1, l2);
		kernel.setArg(2, to);
		kernel.setArg(3, width, 'uint');
		kernel.setArg(4, height, 'uint');

		queue.enqueueNDRangeKernel(kernel, opencl.get_global_size(width, height), opencl.get_local_size);
	}

};


module.exports = dog;
