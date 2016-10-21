__kernel void clear (
	__global float * target,
	const uint width,
	const uint height) {

	// Get our global thread ID
	size_t
		x = get_global_id(0),
		y = get_global_id(1);

	// Make sure we do not go out of bounds
	if (x >= width) x = 0;
	if (y >= height) y = 0;

	target[y * width + x] = 0;
}
