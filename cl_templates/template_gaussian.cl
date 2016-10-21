/**
 *	SIGMA, RADIUS, WIDTH and GAUSSIAN
 *	should be defined out of cl, namely in node.js context
 */

/*
#defien KERNEL_NAME template_gaussian_1_6
#define SIGMA 1.6

#define RADIUS ((int) floor(3 * SIGMA))
#define WIDTH (2 * RADIUS +1)


__constant float GAUSSIAN[WIDTH][WIDTH] = {
	{1, 2, 3},
	{4, 5, 6},
	{7, 8, 9}
};

#define ADDITION \
	  in[(y * 2    ) * width + x * 2    ] \
	+ in[(y * 2    ) * width + x * 2 + 1] \
	+ in[(y * 2 + 1) * width + x * 2    ] \
	+ in[(y * 2 + 1) * width + x * 2 + 1] \
)

*/

__kernel void KERNEL_NAME (
	__global const float * in,
	__global float * out,
	const size_t width,
	const size_t height) {

	//Get our global thread ID
	size_t
		x = get_global_id(0),
		y = get_global_id(1);

	//Make sure we do not go out of bounds
	if (x >= width) x = 0;
	if (y >= height) y = 0;

	out[y * width + x] = ADDITION;
}
