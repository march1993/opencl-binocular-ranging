__constant float RATE[4][4] = {
	{ 1.0 , 0.0 , 0.0 , 0.0  },
	{ 0.5 , 0.5 , 0.0 , 0.0  },
	{ 0.5 , 0.0 , 0.5 , 0.0  },
	{ 0.25, 0.25, 0.25, 0.25 }
};

__kernel void interpolate (
	__global const float * in,
	__global float * out,
	const uint in_width,
	const uint in_height,
	const uint out_width,
	const uint out_height) {

	// Get our global thread ID
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

	// Make sure we do not go out of bounds
	if (x >= out_width) x = 0;
	if (y >= out_height) y = 0;

	size_t
		q_x = x / 2,
		r_x = x % 2,
		q_y = y / 2,
		r_y = y % 2;

	__constant float * rate = RATE[r_y * 2 + r_x];


	out[y * out_width + x]
		= in[ q_y      * in_width + q_x    ] * rate[0]
		+ in[ q_y      * in_width + q_x + 1] * rate[1]
		+ in[(q_y + 1) * in_width + q_x    ] * rate[2]
		+ in[(q_y + 1) * in_width + q_x + 1] * rate[3];
}

