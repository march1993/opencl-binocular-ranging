'use strict';

const fs = require('fs');

const opencl = require('./opencl');

const context = opencl.context;

const S = 3;
const N_LAYER = S + 3;
const SIGMA = 1.6;
const SIGMA_IN_LAYER = Array.apply(null, new Array(N_LAYER)).map((_, index) => SIGMA * Math.pow(2, index / S));


/**
 *	Build MATRIX gaussian template
 */
const calc_gaussian_matrix = (radius, width, sigma) => {
	let sigma2 = sigma * sigma;
	let ret = new Array(width).fill().map((_, y) =>
		new Array(width).fill().map((_, x) =>
			Math.exp(-((x - radius) * (x - radius) + (y - radius) * (y - radius)) / (2 * sigma2)) / (2 * Math.PI * sigma2)
		)
	);

	// Normalization
	let sum = ret.map(i => i.reduce((a, b) => a + b, 0)).reduce((a, b) => a + b, 0);
	ret = ret.map(line => line.map(i => i / sum));

	return ret;
};

/**
 *	MATRIX gaussian
 */
const build_gaussian_maxtrix_addition = (radius, width) =>
	new Array(width).fill().map((_, y) =>
		new Array(width).fill().map((_, x) =>
			`in[(y + (${y - radius})) * width + (x + (${x - radius}))] * GAUSSIAN[${y}][${x}]`
		)
	).reduce((a, b) => a.concat(b), []).filter(i => i != undefined).join('+ \\\n');

const template_gaussian_source = fs.readFileSync('./cl_templates/template_gaussian.cl', { encoding: 'utf-8' });
const init_gaussian = onResolved => {

	let ret = Promise.resolve();

	SIGMA_IN_LAYER.forEach(sigma => {
		let kernel_name = 'template_gaussian_' + sigma.toString().replace('.', '_');
		let RADIUS = Math.ceil(3 * sigma);
		let WIDTH = 2 * RADIUS + 1;
		let source
			= `#define KERNEL_NAME ${kernel_name}\n`
			+ `#define RADIUS ${RADIUS}\n`
			+ `#define WIDTH ${WIDTH}\n`

			/**
			 *	MATRIX gaussian
			 */

			// MATRIX mode
			+ `__constant float GAUSSIAN[WIDTH][WIDTH] = {\n`
			+ calc_gaussian_matrix(RADIUS, WIDTH, sigma).map(line => '{' + line.join(',\n') + '}').join(',\n')
			+ `};\n`
			+ `#define ADDITION ${build_gaussian_maxtrix_addition(RADIUS, WIDTH)} \n`

			+ template_gaussian_source;

		ret = ret.then(onResolved => {
			let program = context.createProgram(source);
			return program.build(opencl.CL_FLAG).then(onResolved => {
				let status = program.getBuildStatus(opencl.device);
				let log = program.getBuildLog(opencl.device);

				if (status < 0) {
					console.error(`Build kernel ${kernel_name} failed.`);
					console.log(source);
					return Promise.reject(log);
				}
				console.log(`Build kernel ${kernel_name} completed.`);

				opencl.kernel_makers[kernel_name] = _ => program.createKernel(kernel_name);
			});
		});
	});

	return ret;
};

exports.init_gaussian = init_gaussian;
exports.SIGMA_IN_LAYER = SIGMA_IN_LAYER;
exports.SIGMA = SIGMA;
exports.S = S;
exports.N_LAYER = N_LAYER;