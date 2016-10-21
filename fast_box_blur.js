'use strict';

const PASSES = 3;

const opencl = require('./opencl');

const boxes_for_gauss = (sigma, n) => { // standard deviation, number of boxes
	let wIdeal = Math.sqrt((12 * sigma * sigma / n) + 1); // Ideal averaging filter width
	let wl = Math.floor(wIdeal); if(wl % 2 == 0) wl--;
	let wu = wl + 2;

	let mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
	let m = Math.round(mIdeal);
	// let sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );

	let sizes = [];  for(let i = 0; i < n; i++) sizes.push( i < m ? wl : wu);
	return sizes;
};


const fast_box_blur = (side, queue, group, SIGMA_IN_LAYER) => {
	let width = group.width,
		height = group.height;

	SIGMA_IN_LAYER.forEach((sigma, index) => {
		let bxs = boxes_for_gauss(sigma, PASSES);

		bxs.forEach(box_length => {
			let radius = (box_length - 1) / 2;

			// can be ignored when box_length == 1
			if (box_length > 1) {
				let tick_h = opencl.tick(side),
					kernel_h = opencl.kernels[side][tick_h] || (opencl.kernels[side][tick_h] = opencl.kernel_makers['fast_box_blur_h']());
				let tick_v = opencl.tick(side),
					kernel_v =  opencl.kernels[side][tick_v] || (opencl.kernels[side][tick_v] = opencl.kernel_makers['fast_box_blur_v']());

				kernel_h.setArg(0, group['layers_cl'][0]);
				kernel_h.setArg(1, group['layers_cl'][1]);
				kernel_h.setArg(2, width, 'uint');
				kernel_h.setArg(3, height, 'uint');
				kernel_h.setArg(4, box_length, 'uint');
				kernel_h.setArg(5, radius, 'uint');

				queue.enqueueNDRangeKernel(kernel_h, opencl.get_global_size(width, height), opencl.get_local_size());

				kernel_v.setArg(0, group['layers_cl'][1]);
				kernel_v.setArg(1, group['layers_cl'][2 + index]);
				kernel_v.setArg(2, width, 'uint');
				kernel_v.setArg(3, height, 'uint');
				kernel_v.setArg(4, box_length, 'uint');
				kernel_v.setArg(5, radius, 'uint');

				queue.enqueueNDRangeKernel(kernel_v, opencl.get_global_size(width, height), opencl.get_local_size());
			}
		});


	});

};


module.exports = fast_box_blur;
