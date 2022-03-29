#include <GPUImageProcess/GPUprocessor.hpp>

GpuProcessor::GpuProcessor(const DataLoader& ld) : loader(ld)
{
}

cv::Mat GpuProcessor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>(); !loader.getCube().is3D())
	{
		// cout << "Applying basic 1D LUT..." << endl;
		cout << "GPU-accelerated 1D LUTs are not implemented yet" << endl;
		//newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	}
	else if (loader.getVm().count("trilinear"))
	{
		cout << "Applying trilinear interpolation..." << endl;
		newImg = GpuTrilinear::applyTrilinearGpu(loader, opacity, threadsPerBlock);
	}
	else
	{
		cout << "Applying nearest-value interpolation..." << endl;
		newImg = GpuNearestVal::applyNearestGpu(loader, opacity, threadsPerBlock);
	}
	return newImg;
}

void GpuProcessor::save() const
{
	std::cout << "Saving...\n";
	const std::string name{loader.getVm()["output"].as<std::string>()};
	try
	{
		imwrite(name, newImg);
	}
	catch (cv::Exception& e)
	{
		cerr << e.what() << "\n"; // output exception message
	}
	CudaUtils::freeUnifiedPtr<unsigned char>(newImg.data);
}

void GpuProcessor::execute()
{
	if (!isCudaAvailable())
	{
		throw std::runtime_error("ERROR (CUDA): Unknown error\n");
	}
	process();
	save();
}

bool GpuProcessor::isCudaAvailable() const
{
	if (!CudaUtils::isCudaDriverAvailable())
	{
		throw std::runtime_error("ERROR (CUDA): CUDA driver was not detected\n");
	}
	if (!CudaUtils::isCudaDeviceAvailable())
	{
		throw std::runtime_error("ERROR (CUDA): No CUDA devices were detected\n");
	}
	return true;
}
