'use strict';

const LENGTH = 16;

let descriptor_weight = sigma => {

	let origin = new Array(LENGTH * LENGTH).fill(0).map((_, index) => {
		let
			_y = parseInt(index / LENGTH),
			_x = index % LENGTH,
			y = 0.5 - LENGTH / 2 + _y,
			x = 0.5 - LENGTH / 2 + _x;

		return Math.exp(- (x * x + y * y) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma);
	});

	let sum = origin.reduce((a, b) => a + b, 0);

	return origin.map(i => i / sum);
}


module.exports = descriptor_weight(LENGTH / 2);