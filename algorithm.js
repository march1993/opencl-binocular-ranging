'use strict';

const opencl = require('./opencl');
const context = opencl.context;
const cl_defs = opencl.defs;
const CLBuffer = opencl.CLBuffer;

const yuyv_to_float = require('./yuyv_to_float');
const float_to_int = require('./float_to_int');
const interpolate = require('./interpolate');
const downsample = require('./downsample');
const gaussian = require('./gaussian');
const template_gaussian = require('./template_gaussian');
const fast_box_blur = require('./fast_box_blur');
const dog = require('./dog');
const detection = require('./detection');
const clear = require('./clear');
const match_descriptors = require('./match_descriptors');

const ref = require('ref');
const double = ref.types.double;
const float = ref.types.float;

const SIGMA = gaussian.SIGMA;
const SIGMA_IN_LAYER = gaussian.SIGMA_IN_LAYER;
const N_LAYER = gaussian.N_LAYER;

const DATA = {
	left: [],
	right: []
};

const CL_BUFFERS = {
	left: {},
	right: {},
	both: {}
};

const OUTPUT_BUFFERS = {
	left: null,
	right: null,
	both: null
};

const QUEUE = opencl.make_queue();


const make_buffer_array = (width, height, count) =>
	new Array(count).fill()
	.map(i => Buffer.allocUnsafe(width * height * float.size));

const make_cl_buffer_array = (width, height, count) =>
	new Array(count).fill()
	.map(i => new CLBuffer(context, cl_defs.CL_MEM_READ_WRITE, width * height * ref.types.float.size));


const init = side => camera_config => {
	{
		let width = camera_config.width,
			height = camera_config.height;

		// zoom out to 2x size
		width = 2 * width;
		height = 2 * height;

		// 640 * 480
		// 320 * 240
		// 160 * 120
		// 80 * 60
		// 40 * 30
		// 20 * 15
		let counter = 0;
		do {
			let group = {
				width: width,
				height: height,
				// Additional one layer if for image without gaussian
				// and the another for tmp usage
				layers: make_buffer_array(width, height, N_LAYER + 2), // Gaussian
				layers_cl: make_cl_buffer_array(width, height, N_LAYER + 2),

				dogs: make_buffer_array(width, height, N_LAYER - 1), // Difference of Gaussian
				dogs_cl: make_cl_buffer_array(width, height, N_LAYER - 1),
				sigma_arr: SIGMA_IN_LAYER.map((_, i) => i + counter * gaussian.S)
			};

			DATA[side].push(group);

			width = width / 2;
			height = height / 2;
			counter += 1
		} while (width === parseInt(width) && height === parseInt(height) && Math.min(width, height) >= (6 * SIGMA + 1));
	}

	{
		let width = camera_config.width,
			height = camera_config.height;

		// initialize buffers
		OUTPUT_BUFFERS[side] = new Buffer(width * height * ref.types.uint8.size);

		// int buffer
		CL_BUFFERS[side]['yuyv_to_float'] = new CLBuffer(context, cl_defs.CL_MEM_READ_ONLY, 2 * width * height * ref.types.uint8.size);
		CL_BUFFERS[side]['float_to_int'] = new CLBuffer(context, cl_defs.CL_MEM_WRITE_ONLY, width * height * ref.types.uint8.size);

		CL_BUFFERS[side]['keypoint_descriptors'] = new CLBuffer(context, cl_defs.CL_MEM_READ_WRITE, width * height * 128 * ref.types.float.size);

	}
};
const init_both = camera_config => {
	let
		width = camera_config.width,
		height = camera_config.height;
	// matches result
	// matched_x, matched_y, delta_sigma
	OUTPUT_BUFFERS['matches'] = new Buffer(width * height * ref.types.float.size * 3);
	CL_BUFFERS['both']['matches'] = new CLBuffer(context, cl_defs.CL_MEM_WRITE_ONLY, width * height * ref.types.float.size * 3);
};
const init_left = init('left');
const init_right = init('right');

/**
 *	function after(buffer)
 *				buffer = width * height
 */
const calc = side => (source_buffer, after) => {
	opencl.clear_tick(side);


	let data = DATA[side];
	// source_buffer.copy(data[1].layers[0]);
	let queue = QUEUE;

	// int to float
	yuyv_to_float(side, queue, source_buffer, CL_BUFFERS[side]['yuyv_to_float'], DATA[side][1]['layers_cl'][0], DATA[side][1].width, DATA[side][1].height);

	// interpolate
	interpolate(side, queue, DATA[side][1]['layers_cl'][0], DATA[side][0]['layers_cl'][0], DATA[side][1].width, DATA[side][1].height);

	// downsample
	for (let i = 1; i <= DATA[side].length - 2; i++) {
		let out_width = DATA[side][i + 1].width,
			out_height = DATA[side][i + 1].height;
		let in_cl_buffer = DATA[side][i]['layers_cl'][0],
			out_cl_buffer = DATA[side][i + 1]['layers_cl'][0];

		downsample(side, queue, in_cl_buffer, out_cl_buffer, out_width, out_height);
	}

	// gaussian
	// DATA[side].forEach(group => template_gaussian(side, queue, group, SIGMA_IN_LAYER));
	DATA[side].forEach(group => fast_box_blur(side, queue, group, SIGMA_IN_LAYER));

	// difference of gaussian
	DATA[side].forEach(group => dog(side, queue, group));

	// detection
	let to_cl_buffer = DATA[side][1].layers_cl[1];
	let to_width = DATA[side][1].width,
		to_height = DATA[side][1].height;
	clear(side, queue, to_cl_buffer, to_width, to_height);
	DATA[side].forEach(group => detection(side, queue, group, to_cl_buffer, CL_BUFFERS[side]['keypoint_descriptors'], to_width, to_height));

	// float to int
	// queue.waitable(true);

	/*
	float_to_int(side, queue, DATA[side][1]['layers_cl'][N_LAYER], OUTPUT_BUFFERS[side], CL_BUFFERS[side]['float_to_int'], DATA[side][1].width, DATA[side][1].height)
	.promise.then(_ => {
		// let t_end = + new Date();
		// console.log(`Calc time: ${t_end - t_begin}ms`);

		after(OUTPUT_BUFFERS[side]);
	});
	*/

	// let ret = [];
	//queue.waitable(true).enqueueReadBuffer(DATA[2].layers_cl[1], 0, DATA[side][2].layers[1].length, DATA[side][2].layers[1]).promise.then(onResolved =>
	/*
	DATA[side].forEach(group => {
		queue.enqueueReadBuffer(group.layers_cl[1], 0, group.layers[1].length, group.layers[1]);
	});
	*/

	// let last = DATA[side][DATA[side].length - 1];
	/*
	let last = DATA[side][1];
	queue.waitable(true).enqueueReadBuffer(last.layers_cl[1], 0, last.layers[1].length, last.layers[1]).promise.then(onResolved => {
		//DATA[side].forEach(group => {
			let group = last;
			let width = group.width,
				height = group.height;
			let arr = new Float32Array(group.layers[1].buffer);
			for (let i = 0; i < arr.length; i++) {
				if (arr[i]) {
					ret.push ({
						x: i % width,
						y: parseInt(i / width),
						v: arr[i],
						w: width,
						h: height
					});
				}
			}
		//});

		console.log(`keypoints count: ${ret.length}`);
		after && after(ret);

	});
	*/

};
const calc_left = calc('left');
const calc_right = calc('right');


const match = (left_buffer, right_buffer, after) => {
	opencl.clear_tick('both');

	let t_begin = + new Date();
	console.log(`start calc both: ${t_begin / 1000}`);


	calc_left(left_buffer);
	calc_right(right_buffer);

	let
		width = DATA['left'][1].width,
		height = DATA['left'][1].height;

	let queue = QUEUE;

	match_descriptors(queue,
		width,
		height,
		DATA['left'][1].layers_cl[1], CL_BUFFERS['left']['keypoint_descriptors'],
		DATA['right'][1].layers_cl[1], CL_BUFFERS['right']['keypoint_descriptors'],
		CL_BUFFERS['both']['matches']
	);

	queue.waitable(true).enqueueReadBuffer(CL_BUFFERS['both']['matches'], 0, OUTPUT_BUFFERS['matches'].length, OUTPUT_BUFFERS['matches']).promise.then(onResolved => {

		let ret = [];

		let arr = new Float32Array(OUTPUT_BUFFERS['matches'].buffer);

		for (let h = 0; h < height; h++)
		for (let w = 0; w < width; w++) {
			let
				target_x = (arr[(h * width + w) * 3 + 0]),
				target_y = (arr[(h * width + w) * 3 + 1]),
				delta_sigma = (arr[(h * width + w) * 3 + 2]);

			if (target_x >= 0 && target_y >= 0) {
				ret.push({
					right_x: target_x,
					right_y: target_y,
					left_x: w,
					left_y: h,
					delta_sigma: delta_sigma
				});
			}
		}


		let t_end = + new Date();
		console.log(`Calc time: ${t_end - t_begin}ms`);

		console.log(`matches count: ${ret.length}`);
		after && after(ret);

	});

};


// const ready = opencl.ready.then(gaussian.init_gaussian);
const ready = opencl.ready; // .then(gaussian.init_gaussian);


exports.DATA = DATA;
exports.CL_BUFFERS = CL_BUFFERS;
exports.init_left = init_left;
exports.init_right = init_right;
exports.init_both = init_both;
exports.calc_left = calc_left;
exports.calc_right = calc_right;
exports.match = match;
exports.ready = ready;