#pragma once
#ifndef CubeLUT_H
#define CubeLUT_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

using namespace std;

// Licensed under Creative Commons Attribution Non-Commercial 3.0 License
// Author: Adobe Inc. (with some slight modifications made by Jakub Ochnik)
// Source:
// "Cube LUT Specification 1.0"
// https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf

class CubeLUT
{
public:
	using tableRow = Eigen::Vector3f; // boost::numeric::ublas::vector<float>;
	using table1D = Eigen::Tensor<float, 2>;//vector<tableRow>;
	// using table2D = vector<table1D>;
	using table3D = Eigen::Tensor<float, 4>;//vector<table2D>;

	enum LUTState
	{
		OK = 0,
		NotInitialized = 1,
		ReadError = 10,
		WriteError,
		PrematureEndOfFile,
		LineError,
		UnknownOrRepeatedKeyword = 20,
		TitleMissingQuote,
		DomainBoundsReversed,
		LUTSizeOutOfRange,
		CouldNotParseTableData
	};

	LUTState status;
	string title;
	std::vector<float> domainMin{0, 0, 0};
	std::vector<float> domainMax{1, 1, 1};
	table1D LUT1D;
	table3D LUT3D;

	CubeLUT()
	{
		status = NotInitialized;
	}

	LUTState LoadCubeFile(ifstream& infile);

	bool is3D() const;
private:
	string ReadLine(ifstream& infile, char lineSeparator);
	void ParseTableRow(const string& lineOfText, const int r, const int g, const int b);
	void ParseTableRow(const string& lineOfText, const int i);
};

#endif