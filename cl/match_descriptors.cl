#define RADIUS_Y 5
#define DELTA_N_TH_BLUR 2

__kernel void match_descriptors (
	__global const float * left_cl,
	__global const float * left_descriptor,
	__global float * right_cl,
	__global const float * right_descriptor,
	__global float * matches_cl,
	const int width,
	const int height) {

	// Get our global thread ID
	size_t
		x = get_global_id(0),
		y = get_global_id(1);

	// Make sure we do not go out of bounds
	if (x >= width) x = 0;
	if (y >= height) y = 0;


	size_t offset_left = (y * width + x) * 128;

	float min_sum = 1e20, second_sum = 1e21; // magic 0.5: -1 * 0.8 < 0.5
	int
		min_x = -1,
		min_y = -1;

	float
		sigma_left = left_cl[y * width + x],
		min_sigma_right = -1;

	if (sigma_left)
	for (int right_y = max(0, ((int)y) - RADIUS_Y); right_y < min((int)height, ((int)y) + RADIUS_Y); right_y++)
	for (int right_x = 0; right_x <= x; right_x++) {

		float sigma_right = right_cl[right_y * width + right_x];

		if (sigma_right && fabs(sigma_right - sigma_left) <= DELTA_N_TH_BLUR) {
			float sum = 0;
			size_t offset_right = (right_y * width + right_x) * 128;

			for (size_t i = 0; i < 128; i++) {
				float delta = left_descriptor[offset_left + i] - right_descriptor[offset_right + i];
				sum += delta * delta;
			}

			if (sum < min_sum) {
				second_sum = min_sum;

				min_sum = sum;
				min_x = right_x;
				min_y = right_y;
				min_sigma_right = sigma_right;
			}
		}
	}

	size_t offset = (y * width + x) * 3;
	if (min_sum < second_sum * 0.8) {
		matches_cl[offset + 0] = min_x;
		matches_cl[offset + 1] = min_y;
		matches_cl[offset + 2] = sigma_left - min_sigma_right;
	} else {
		matches_cl[offset + 0] = -1;
		matches_cl[offset + 1] = -1;
		matches_cl[offset + 2] = -1;
	}
}
