'use strict';

const fs = require('fs');
const v4l2camera = require("v4l2camera");
const express = require('express');
const app = express();
const opencl = require('./opencl');
const algorithm = require('./algorithm');

let server;
let left_buffer, right_buffer;

/**
 *	静态文件
 */
app.use('/static/', express.static('static'));

app.get('/', (req, res) => {
	// res.send('<script type="text/javascript">window.location = "/static/index.html";</script>');
	res.redirect('./static/index.html');
});

app.get('/snapshot/left', (req, res) => {
	left_buffer = new Buffer(cam_left.frameRaw());

	res.send(left_buffer);
	/*
	algorithm.calc_left(buffer, result_buffer => {
		res.send(result_buffer);
	});
	*/
});

app.get('/snapshot/right', (req, res) => {
	right_buffer = new Buffer(cam_right.frameRaw());

	res.send(right_buffer);
	/*
	algorithm.calc_right(buffer, result_buffer => {
		res.send(result_buffer);
	});
	*/
});

app.get('/keypoints/left', (req, res) => {

	algorithm.calc_left(left_buffer, result => {
		res.send(result);
	});

});

app.get('/keypoints/right', (req, res) => {

	algorithm.calc_right(right_buffer, result => {
		res.send(result);
	});

});

app.get('/matches', (req, res) => {

	algorithm.match(left_buffer, right_buffer, result => {
		res.send(result);
	});

});

app.get('/config/left', (req, res) => {
	res.send(cam_left.configGet());
});

app.get('/config/right', (req, res) => {
	res.send(cam_right.configGet());
});

let running = true;

algorithm.ready.then(onResolved => {
	server = app.listen(8080, _ => {
		console.log('Listening on port 8080!');
	});
}, onRejected => {
	running = false;
	console.error('Initialization failed.', onRejected);
});

const cam_left = new v4l2camera.Camera("/dev/video1");
const cam_right = new v4l2camera.Camera("/dev/video2");

const make_camera_init = camera => {
	// 640px * 480px @ 30fps
	camera.configSet(camera.formats[9]);

	let BWT_auto_id = camera.controls.filter(i => i.name === 'White Balance Temperature, Auto')[0].id;
	camera.controlSet(BWT_auto_id, 0);

	camera.start();

	let loop = success => {
		if (running) {
			camera.capture(loop);
		} else {
			running = false;
			camera.stop(_ => {
				console.log(`Camera ${camera.device} stopped.`);
			});
		}
	};

	camera.capture(loop);
};

make_camera_init(cam_left);
make_camera_init(cam_right);
console.log('Camera started.');

algorithm.init_left(cam_left.configGet());
algorithm.init_right(cam_right.configGet());
algorithm.init_both(cam_left.configGet());

console.log('Algorithm initialized.');

process.on('SIGINT', e => {
	server && server.close(_ => console.log('Server closed.'));
	running = false;

	setTimeout(_ => {
		process.exit();
	}, 1000);
});