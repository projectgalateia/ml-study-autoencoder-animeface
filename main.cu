#include <cuda.h>
#include <cublas_v2.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>

#define IMAGE_SAVE
#define IMAGE_LOAD
#include "image.hpp"
#include "autoencoder.hpp"

#define HIDDEN 10

#define PS 160
#define SZ (PS*PS)

int main()
{
	using namespace ml::image;
	using ml::autoencoder::Autoencoder;

	cuInit(0);

	std::vector<Image *> images;

	std::string filename;
	std::ifstream ifs("files.txt");

	while (std::getline(ifs, filename)) {
		printf("Loading %s\n", filename.c_str());

		Image *img = load_image(filename.c_str(), 1);

		Image *rmg = img->resize(PS, PS);

		Image *data = edge_detection<64, 64>(rmg);

		delete rmg;
		delete img;

		images.push_back(data);
	}

	Autoencoder encoder(SZ, HIDDEN);
	Image *res = new Image(PS, PS, 1);

	float err;

	do {
		err = 0.0f;

		for (int i = 0; i < images.size(); ++i) {
			Image *data = images[i];
			encoder.propagate(data->h_data);

			err += encoder.err() / images.size();

			/*res->copyDevice(encoder.output_o);*/
			/*res->download();*/

			/*float test[10];*/
			/*cudaMemcpy(test, encoder.output_h, sizeof(float) * 10, cudaMemcpyDeviceToHost);*/

			/*for (int i = 0; i < 10; ++i) {*/
			/*printf("%3.1f ", test[i]);*/
			/*}*/
			/*printf("\n");*/

			/*for (int i = 0; i < PS; ++i) {*/
			/*for (int j = 0; j < PS; ++j) {*/
			/*const int idx = i * PS + j;*/

			/*printf("%d", res->h_data[idx] > 0.1f);*/
			/*}*/
			/*printf("\n");*/
			/*}*/
			/*printf("\n");*/
			/*printf("err: %f\n", err);*/

			encoder.backpropagate();
		}

		printf("ERROR: %e\n", err);

		for (int i = 0; i < HIDDEN; ++i) {
			char filename[256];

			sprintf(filename, "res-%d.bmp", i);

			encoder.visualize(i, res->d_data);
			res->download();

			res->write(filename);
		}
	} while (err > 1E-01);

	return 0;
}

