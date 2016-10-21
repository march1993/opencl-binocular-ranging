__kernel void downsample (
	__global float * in,
	__global float * out,
	const uint in_width,
	const uint in_height,
	const uint out_width,
	const uint out_height) {

	// Get our global thread ID
	size_t
		x = get_global_id(0),
		y = get_global_id(1);

	// Make sure we do not go out of bounds
	if (x >= out_width) x = 0;
	if (y >= out_height) y = 0;

	out[y * out_width + x] = (
		  in[(y * 2    ) * in_width + x * 2    ]
		+ in[(y * 2    ) * in_width + x * 2 + 1]
		+ in[(y * 2 + 1) * in_width + x * 2    ]
		+ in[(y * 2 + 1) * in_width + x * 2 + 1]
	) / 4.0;
}
