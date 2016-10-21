__kernel void float_to_int (
	__global const float * in,
	__global uchar * out,
	const uint width,
	const uint height) {

	// Get our global thread ID
	size_t
		x = get_global_id(0),
		y = get_global_id(1);

	// Make sure we do not go out of bounds
	if (x >= width) x = 0;
	if (y >= height) y = 0;

	float t = in[y * width + x];
	out[y * width + x] = t > 255 ? 0xff : t < 0 ? 0 : t;
}
