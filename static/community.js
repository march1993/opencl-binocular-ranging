'use strict';

const
	THRESHOLD_DD = 4,
	THRESHOLD_DX = 30,
	THRESHOLD_DY = 20;

const most_label_of = neighbour => {
	let stat = {};
	neighbour.map(i => i.label).forEach(i => {
		stat[i] = stat[i] || 0;
		stat[i] += 1;
	});

	let most = undefined, max = -1;
	for (let i in stat) {
		if (stat[i] > max) {
			max = stat[i];
			most = i;
		}
	}

	return most;
};

const find_communities = matches => {
	matches.forEach(match => {
		match.delta_x = match.right_x - match.left_x;
		match.delta_y = match.right_y - match.left_y;
	});

	// initialize
	matches.forEach((match, index) => {
		match.neighbour = [];
		match.label = '' + index;
	});

	const M = matches.map(i => i.delta_y).reduce((a, b) => a + b, 0)
		/ matches.map(i => i.delta_x).reduce((a, b) => a + b, 0);

	console.log(`M = ${M}`);

	matches = matches.filter(match =>
		Math.abs(match.delta_x) + Math.abs(match.delta_y) <= 50 && Math.abs(match.delta_y) < 5
		|| Math.abs(M - match.delta_y / match.delta_x) / M <= 2);


	for (let i = 0; i < matches.length; i++)
	for (let j = (i + 1); j < matches.length; j++) {
		let
			match = matches[i],
			match2 = matches[j];

		let
			ddx = match.delta_x - match2.delta_x,
			ddy = match.delta_y - match2.delta_y;

		let
			dx = match.left_x - match2.left_x,
			dy = match.left_y - match2.left_y;

		let signal = match.left_x * match2.left_x >= 0
			&& match.left_y * match2.left_y >= 0;

		if (signal && Math.abs(ddx) + Math.abs(ddy) <= THRESHOLD_DD && Math.abs(dx) <= THRESHOLD_DX && Math.abs(dy) <= THRESHOLD_DY) {
			// neighbour found
			match.neighbour.push(match2);
			match2.neighbour.push(match);
		}
	}


	let tick;
	do {
		tick = false;

		matches.forEach(match => {
			if (match.neighbour.length > 0) {
				let
					label_old = match.label,
					label_new = most_label_of(match.neighbour);

					if (label_old != label_new) {
						tick = true;
						match.label = label_new;
					}
			}
		});

	} while (tick);


	// build communities
	let communities = {};
	matches.forEach(match => {
		let label = match.label;
		communities[label] = communities[label] || [];
		communities[label].push(match);
	});

	let shrink = {};
	for (let label in communities) {
		let community = communities[label];
		let least_dx = findLineByLeastSquares(community.map(i => i.delta_x));
		if (community.length > 11 && Math.abs(least_dx.m) < 0.2 && Math.abs(least_dx.mid) < 640 / 3) {
			community.avg_delta_x = least_dx.mid;
			shrink[label] = community;
		}
	}

	return shrink;
};
