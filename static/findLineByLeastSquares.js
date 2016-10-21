'use strict';

// const findLineByLeastSquares = (values_x, values_y) => {

const findLineByLeastSquares = (values_y) => {

	let values_x = new Array(values_y.length).fill(0).map((_, i) => i);

	var sum_x = 0;
	var sum_y = 0;
	var sum_xy = 0;
	var sum_xx = 0;
	var count = 0;

	/*
	 * We'll use those variables for faster read/write access.
	 */
	var x = 0;
	var y = 0;
	var values_length = values_x.length;

	if (values_length != values_y.length) {
		throw new Error('The parameters values_x and values_y need to have same size!');
	}

	/*
	 * Nothing to do.
	 */
	if (values_length === 0) {
		return [ [], [] ];
	}

	/*
	 * Calculate the sum for each of the parts necessary.
	 */
	for (var v = 0; v < values_length; v++) {
		x = values_x[v];
		y = values_y[v];
		sum_x += x;
		sum_y += y;
		sum_xx += x*x;
		sum_xy += x*y;
		count++;
	}

	/*
	 * Calculate m and b for the formular:
	 * y = x * m + b
	 */
	var m = (count*sum_xy - sum_x*sum_y) / (count*sum_xx - sum_x*sum_x);
	var b = (sum_y/count) - (m*sum_x)/count;

	/*
	 * We will make the x and y result line now
	 */
	var result_values_x = [];
	var result_values_y = [];

	for (var v = 0; v < values_length; v++) {
		x = values_x[v];
		y = x * m + b;
		result_values_x.push(x);
		result_values_y.push(y);
	}

	// return [result_values_x, result_values_y];
	return {
		m: m,
		b: b,
		mid: values_length / 2 * m + b
	};
};