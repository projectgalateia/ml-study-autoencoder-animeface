#pragma once

#ifdef IMAGE_LOAD
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifdef IMAGE_SAVE
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

namespace ml {
namespace image {

struct Image {
	const int W;
	const int H;
	const int C;

	float *h_data;
	float *d_data;

	Image(const int W, const int H, const int C)
		: W(W), H(H), C(C)
	{
		h_data = new float[W * H * C];
		cudaMalloc(&d_data, sizeof(float) * W * H * C);
	}

	~Image()
	{
		delete h_data;
		cudaFree(&d_data);
	}

	void upload(float *ret = NULL)
	{
		if (!ret) ret = d_data;

		cudaMemcpy(ret, h_data, sizeof(float) * W * H * C, cudaMemcpyHostToDevice);
	}

	void download(float *ret = NULL)
	{
		if (!ret) ret = h_data;

		cudaMemcpy(ret, d_data, sizeof(float) * W * H * C, cudaMemcpyDeviceToHost);
	}

	void copyDevice(float *data)
	{
		cudaMemcpy(d_data, data, sizeof(float) * W * H * C, cudaMemcpyDeviceToDevice);
	}

	void copyHost(float *data)
	{
		memcpy(h_data, data, sizeof(float) * W * H * C);
	}

	Image *resize(const int w, const int h)
	{
		Image *ret = new Image(w, h, C);

		for (int k = 0; k < C; ++k) {
			for (int i = 0; i < h; ++i) {
				for (int j = 0; j < w; ++j) {
					const int idx = (i * w + j) * C + k;
					int sz = 0;
					float val = 0.0f;

					for (int p = (i * (H-1) / h); p <= ((i+1) * (H-1) / h); ++p) {
						for (int q = (j * (W-1) / w); q <= ((j+1) * (W-1) / w); ++q) {
							const int sdx = (p * W + q) * C + k;

							val += h_data[sdx];
							sz++;
						}
					}

					ret->h_data[idx] = val / sz;
				}
			}
		}

		ret->upload();

		return ret;
	}

	Image *crop(const int x, const int y, const int w, const int h)
	{
		Image *ret = new Image(w, h, C);

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				for (int k = 0; k < C; ++k) {
					const int Y = y + i;
					const int X = x + j;

					const int idx = (i * w + j) * C + k;
					const int sdx = (Y * w + X) * C + k;

					ret->h_data[idx] = h_data[sdx];
				}
			}
		}

		ret->upload();

		return ret;
	}

#ifdef IMAGE_SAVE
	void write(const char *filename)
	{
		unsigned char *raw = new unsigned char[W*H*C];

		for (int i = 0; i < H; ++i) {
			for (int j = 0; j < W; ++j) {
				for (int k = 0; k < C; ++k) {
					const int idx = (i * W + j) * C + k;

					raw[idx] = h_data[idx] * 255;
				}
			}
		}

		stbi_write_bmp(filename, W, H, C, raw);

		delete raw;
	}
#endif
};

#ifdef IMAGE_LOAD
static Image *load_image(const char *filename, const int C = 3)
{
	Image *ret = NULL;

	int w, h, comp;
	unsigned char *data;

	data = stbi_load(filename, &w, &h, &comp, C);

	ret = new Image(w, h, C);

	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			for (int k = 0; k < C; ++k) {
				const int idx = (i * w + j) * C + k;

				ret->h_data[idx] = data[idx] / 255.0f;
			}
		}
	}

	free(data);

	ret->upload();

	return ret;
}
#endif

__global__ void grayscale_rgb(const float *img, float *ret, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		ret[n] = 0.2126f * img[n * 3 + 0] + 0.7152f * img[n * 3 + 1] + 0.0722f * img[n * 3 + 2];
	}
}

template<size_t A, size_t B>
Image *grayscale(const Image *img)
{
	Image *ret = NULL;

	if (img->C == 1) {
		ret = new Image(img->W, img->H, 1);

		memcpy(ret->h_data, img->h_data, sizeof(float) * img->W * img->H);
		ret->upload();
	} else if (img->C == 3) {
		ret = new Image(img->W, img->H, 1);

		grayscale_rgb<<<A, B>>>(img->d_data, ret->d_data, img->W * img->H);
		ret->download();
	}

	return ret;
}

__device__ float fget(const float *img, const int y, const int x, const int W, const int H, const int C = 1, const int c = 0)
{
	if (x < 0 || x >= W) return 0.0f;
	if (y < 0 || y >= H) return 0.0f;

	return img[(y * W + x) * C + c];
}

__device__ void fset(float *img, const int y, const int x, const int W, const int H, const float val, const int C = 1, const int c = 0)
{
	if (x < 0 || x >= W) return;
	if (y < 0 || y >= H) return;

	img[(y * W + x) * C + c] = val;
}

__global__ void gaussian(const float *img, float *ret, const int W, const int H)
{
	const int x_min = threadIdx.x * W / blockDim.x;
	const int y_min = blockIdx.x * H / gridDim.x;

	const int x_max = (threadIdx.x+1) * W / blockDim.x;
	const int y_max = (blockIdx.x+1) * H / gridDim.x;

	const float M[5][5] = {
		{2,  4,  5,  4, 2},
		{4,  9, 12,  9, 4},
		{5, 12, 15, 12, 5},
		{4,  9, 12,  9, 4},
		{2,  4,  5,  4, 2},
	};

	for (int i = y_min; i < y_max; ++i) {
		for (int j = x_min; j < x_max; ++j) {
			const int idx = i * W + j;

			float res = 0.0f;

			for (int p = -2; p <= 2; ++p) {
				for (int q = -2; q <= 2; ++q) {
					res += M[p+2][q+2] * fget(img, i + p, j + q, W, H);
				}
			}

			ret[idx] = res / 159.0f;
		}
	}
}

template<size_t A, size_t B>
Image *gaussian(const Image *img)
{
	Image *ret = NULL;

	if (img->C == 1) {
		ret = new Image(img->W, img->H, img->C);

		gaussian<<<A, B>>>(img->d_data, ret->d_data, img->W, img->H);

		ret->download();
	}

	return ret;
}

__global__ void canny_gradient(const float *img, float *ret, const int W, const int H)
{
	const int x_min = threadIdx.x * W / blockDim.x;
	const int y_min = blockIdx.x * H / gridDim.x;

	const int x_max = (threadIdx.x+1) * W / blockDim.x;
	const int y_max = (blockIdx.x+1) * H / gridDim.x;

	const float GX[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	};

	const float GY[3][3] = {
		{-1, -2, -1},
		{ 0,  0,  0},
		{ 1,  2,  1},
	};

	for (int i = y_min; i < y_max; ++i) {
		for (int j = x_min; j < x_max; ++j) {
			const int idx = (i * W + j) * 2;

			float gx = 0.0f;
			float gy = 0.0f;

			for (int p = -1; p <= 1; ++p) {
				for (int q = -1; q <= 1; ++q) {
					gx += GX[p+1][q+1] * fget(img, i + p, j + q, W, H);
					gy += GY[p+1][q+1] * fget(img, i + p, j + q, W, H);
				}
			}

			ret[idx + 0] = sqrt(gx * gx + gy * gy);
			ret[idx + 1] = atan2f(gy, gx) * 180.0f / 3.1415926535f;
		}
	}
}

__global__ void canny_detect(const float *data, float *ret, const int W, const int H, const float threshold)
{
	const int x_min = threadIdx.x * W / blockDim.x;
	const int y_min = blockIdx.x * H / gridDim.x;

	const int x_max = (threadIdx.x+1) * W / blockDim.x;
	const int y_max = (blockIdx.x+1) * H / gridDim.x;

	for (int i = y_min; i < y_max; ++i) {
		for (int j = x_min; j < x_max; ++j) {
			const float gradd = fget(data, i, j, W, H, 2, 0);
			const float angle = fget(data, i, j, W, H, 2, 1);

			fset(ret, i, j, W, H, 0.0f);

			const float gradv = gradd - threshold;

			if (angle > 135) {
				if (gradv > fget(data, i, j-1, W, H, 2, 0) && gradv > fget(data, i, j+1, W, H, 2, 0)) {
					fset(ret, i, j, W, H, 1.0f);
				}
			} else if (angle > 45) {
				if (gradv > fget(data, i-1, j, W, H, 2, 0) && gradv > fget(data, i+1, j, W, H, 2, 0)) {
					fset(ret, i, j, W, H, 1.0f);
				}
			} else if (angle > -45) {
				if (gradv > fget(data, i, j-1, W, H, 2, 0) && gradv > fget(data, i, j+1, W, H, 2, 0)) {
					fset(ret, i, j, W, H, 1.0f);
				}
			} else if (angle > -135) {
				if (gradv > fget(data, i-1, j, W, H, 2, 0) && gradv > fget(data, i+1, j, W, H, 2, 0)) {
					fset(ret, i, j, W, H, 1.0f);
				}
			} else {
				if (gradv > fget(data, i, j-1, W, H, 2, 0) && gradv > fget(data, i, j+1, W, H, 2, 0)) {
					fset(ret, i, j, W, H, 1.0f);
				}
			}
		}
	}
}

template<size_t A, size_t B>
Image *edge_detection(const Image *img, const float threshold = 0.01f)
{
	Image *ret = NULL;

	if (img->C == 1) {
		float *tmp;

		ret = new Image(img->W, img->H, img->C);

		cudaMalloc(&tmp, sizeof(float) * img->W * img->H * 2);

		gaussian<<<A, B>>>(img->d_data, ret->d_data, img->W, img->H);
		canny_gradient<<<A, B>>>(ret->d_data, tmp, img->W, img->H);

		canny_detect<<<A, B>>>(tmp, ret->d_data, img->W, img->H, threshold);
		ret->download();

		cudaFree(tmp);
	}

	return ret;
}

}
}

