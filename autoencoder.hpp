#pragma once

#include <cublas_v2.h>

namespace ml {
namespace autoencoder {

__device__ float step_function(float v)
{
	return 1.0f / (1.0f + exp(-v));
}


__global__ void apply_step_function(float *preact, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(preact[idx]);
	}
}

__global__ void grad_preact(float *d_preact, float *d_output, float *preact, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		const float g = step_function(preact[idx]);

		d_preact[idx] = d_output[idx] * g * (1 - g);
	}
}

__global__ void take_weight(float *weight, float *ret, const int H, const int I, const int index)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = I * pos / size; idx < I * (pos+1) / size; ++idx) {
		ret[idx] = weight[H * idx + index];
	}
}

struct Autoencoder {
	float *input;

	float *weight;

	float *bias_h;
	float *preact_h;
	float *output_h;

	float *bias_o;
	float *preact_o;
	float *output_o;

	float *d_preact_o;
	float *d_weight_o;

	float *d_output_h;
	float *d_preact_h;
	float *d_weight_h;

	const int I;
	const int H;

	cublasHandle_t handle;

	Autoencoder(const int I, const int H)
		: I(I), H(H)
	{
		float h_weight[I][H];

		for (int i = 0; i < I; ++i) {
			for (int j = 0; j < H; ++j) {
				h_weight[i][j] = 1.0f - 2.0f * float(rand()) / float(RAND_MAX);
			}
		}

		cublasCreate(&handle);

		cudaMalloc(&input, sizeof(float) * I);

		cudaMalloc(&weight, sizeof(float) * I * H);

		cudaMalloc(&bias_h,   sizeof(float) * H);
		cudaMalloc(&preact_h, sizeof(float) * H);
		cudaMalloc(&output_h, sizeof(float) * H);

		cudaMalloc(&bias_o,   sizeof(float) * I);
		cudaMalloc(&preact_o, sizeof(float) * I);
		cudaMalloc(&output_o, sizeof(float) * I);

		cudaMalloc(&d_preact_o, sizeof(float) * I);
		cudaMalloc(&d_weight_o, sizeof(float) * I * H);

		cudaMalloc(&d_output_h, sizeof(float) * H);
		cudaMalloc(&d_preact_h, sizeof(float) * H);
		cudaMalloc(&d_weight_h, sizeof(float) * H * I);

		cudaMemcpy(weight, h_weight, sizeof(float) * H * I, cudaMemcpyHostToDevice);
		cudaMemset(bias_h, 0x0, sizeof(float) * H);
		cudaMemset(bias_o, 0x0, sizeof(float) * I);
	}

	~Autoencoder()
	{
		cublasDestroy(handle);
		
		cudaFree(input);

		cudaFree(weight);

		cudaFree(bias_h);
		cudaFree(preact_h);
		cudaFree(output_h);

		cudaFree(bias_o);
		cudaFree(preact_o);
		cudaFree(output_o);

		cudaFree(d_preact_o);
		cudaFree(d_weight_o);

		cudaFree(d_output_h);
		cudaFree(d_preact_h);
		cudaFree(d_weight_h);
	}

	void propagate(float *data)
	{
		static const float alpha = 1.0f;
		static const float beta = 0.0f;
		static const float minus = -1.0f;

		cudaMemcpy(input, data, sizeof(float) * I, cudaMemcpyHostToDevice);

		cublasSgemv(handle, CUBLAS_OP_N, H, I, &alpha, weight, H, input, 1, &beta, preact_h, 1);
		cublasSaxpy(handle, H, &alpha, bias_h, 1, preact_h, 1);
		apply_step_function<<<50, 50>>>(preact_h, output_h, H);

		cublasSgemv(handle, CUBLAS_OP_T, H, I, &alpha, weight, H, output_h, 1, &beta, preact_o, 1);
		cublasSaxpy(handle, I, &alpha, bias_o, 1, preact_o, 1);
		apply_step_function<<<50, 50>>>(preact_o, output_o, I);

		cublasScopy(handle, I, input, 1, d_preact_o, 1);
		cublasSaxpy(handle, I, &minus, output_o, 1, d_preact_o, 1);
	}

	void backpropagate()
	{
		static const float alpha = 1.0f;
		static const float beta = 0.0f;
		static const float dt = 0.1f;

		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H, I, 1, &alpha, output_h, H, d_preact_o, I,
			&beta, d_weight_o, H);
		cublasSgemv(handle, CUBLAS_OP_N, H, I, &alpha, weight, H, d_preact_o, 1, &beta, d_output_h, 1);
		grad_preact<<<50, 50>>>(d_preact_h, d_output_h, preact_h, H);

		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H, I, 1, &alpha, d_preact_h, H, input, I,
			&beta, d_weight_h, H);
		
		cublasSaxpy(handle, I, &dt, d_preact_o, 1, bias_o, 1);
		cublasSaxpy(handle, H*I, &dt, d_weight_o, 1, weight, 1);
		cublasSaxpy(handle, H, &dt, d_preact_h, 1, bias_h, 1);
		// Makes Learning Process Slower... Why?
		//cublasSaxpy(handle, H*I, &dt, d_weight_h, 1, weight, 1);
	}

	float err()
	{
		float ret;

		cublasSnrm2(handle, I, d_preact_o, 1, &ret);

		return ret;
	}

	void visualize(int idx, float *ret)
	{
		float nrm;

		take_weight<<<50, 50>>>(weight, ret, H, I, idx);

		cublasSnrm2(handle, I, ret, 1, &nrm);
		
		nrm = 1.0f / nrm;

		cublasSscal(handle, I, &nrm, ret, 1);
	}
};

}
}

