#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessor.hpp>

class GpuProcessor : public ImageProcessor
{
	static constexpr int threadsPerBlock{16};

public:
	GpuProcessor(FileIO& fileIfc);
	~GpuProcessor();

private:
	cv::Mat process(float strength, InterpolationMethod method) override;
};
