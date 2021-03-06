

__kernel void fast_box_blur_v (
	__global const float * in,
	__global float * out,
	const uint width,
	const uint height,
	const uint box_length,
	const uint radius) {

	// Get our global thread ID
	size_t
		x = get_global_id(0),
		y = get_global_id(1);

	// Make sure we do not go out of bounds
	if (x >= width) x = 0;
	if (y >= height) y = 0; // y should always be zero, namely useless

	float sum = 0;
	for (size_t i = y - radius; i <= y + radius; i++) {
		sum += in[i * width + x];
	}
	out[y * width + x] = sum / box_length;
}
