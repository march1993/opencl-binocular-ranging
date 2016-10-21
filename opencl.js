'use strict';

const CL_FLAG = '-cl-fast-relaxed-math';

const nooocl = require('nooocl');
const CLHost = nooocl.CLHost;
const CLPlatform = nooocl.CLPlatform;
const CLDevice = nooocl.CLDevice;
const CLContext = nooocl.CLContext;
const CLBuffer = nooocl.CLBuffer;
const CLCommandQueue = nooocl.CLCommandQueue;
const CLUserEvent = nooocl.CLUserEvent;
const NDRange = nooocl.NDRange;
const CLProgram = nooocl.CLProgram;
const CLKernel = nooocl.CLKernel;
const CLImage2D = nooocl.CLImage2D;
const CLImage3D = nooocl.CLImage3D;
const CLSampler = nooocl.CLSampler;

const fs = require('fs');
const ceil = Math.ceil;


// host = CLHost.createV11(); // for OpenCL 1.1
// host = CLHost.createV12(); // for OpenCL 1.2
// host = new CLHost(1.1); // for OpenCL 1.1
const host = new CLHost(1.2); // for OpenCL 1.2
const count = host.platformsCount;
const platform = host.getPlatforms()[0]; // first platform
const device = platform.allDevices()[0]; // first device

const defs = platform.cl.defs;
const types = host.cl.types;
const ImageFormat = types.ImageFormat;
const context = new CLContext(platform, defs.CL_DEVICE_TYPE_ALL);
const make_queue = _ => new CLCommandQueue(context, device);

const kernel_makers = {};
const kernels = {
	left: [],
	right: [],
	both: []
};

let sources = fs.readdirSync('./cl');

/**
 *	Build all kernels
 */
{
	let ret = Promise.resolve();

	sources.forEach(source =>
		ret = ret.then(onResolved => {
			let kernel_name = source.slice(0, -3);
			let program = context.createProgram(fs.readFileSync('./cl/' + source, { encoding: 'utf8' }));

			return program.build(CL_FLAG).then(onResolved => {
				let status = program.getBuildStatus(device);
				let log = program.getBuildLog(device);

				if (status < 0) {
					console.error(`Build kernel ${kernel_name} failed.`);
					return Promise.reject(log);
				}

				console.log(`Build kernel ${kernel_name} completed.`);

				kernel_makers[kernel_name] = _ => program.createKernel(kernel_name);

			});

		})

	);

	exports.ready = ret;
}



// Ranges:
// Number of work items in each local work group
// const N = 128;
// const get_local_size = _ => new NDRange(N, N);
const get_local_size = _ => undefined;

// Number of total work items - localSize must be devisor
// const get_global_size = (width, height) => new NDRange(ceil(width / N) * N, ceil(height / N) * N);
const get_global_size = (width, height) => new NDRange(width, height);



/**
 *	tick and tuck
 */
{

	let _tick = {
		left: 0,
		right: 0,
		both: 0
	};

	exports.tick = side => _tick[side]++;
	exports.clear_tick = side => _tick[side] = 0;
}

exports.CL_FLAG = CL_FLAG;

exports.kernel_makers = kernel_makers;
exports.kernels = kernels;
exports.context = context;
exports.make_queue = make_queue;
exports.platform = platform;
exports.device = device;
exports.defs = defs;
exports.types = types;
exports.ImageFormat = ImageFormat;
exports.CLBuffer = CLBuffer;
exports.NDRange = NDRange;
exports.host = host;

exports.get_local_size = get_local_size;
exports.get_global_size = get_global_size;
exports.set_N = v => N = v;