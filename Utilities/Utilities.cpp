#include "Utilities.h"

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <sstream>

::std::vector<::std::string> ReadLinesFromFile(const ::std::string& pathToFile)
{
	::std::vector< ::std::string> linesVector;

	::std::ifstream inputFile(pathToFile.c_str());
	
	::std::string tempLine;
	while(::std::getline(inputFile, tempLine))
		linesVector.push_back(tempLine);

	return linesVector;
}


::std::vector< double> DoubleVectorFromString(const ::std::string& inputString)
{
	::std::istringstream ss(inputString);

	::std::vector<double> result;
	while(!ss.eof())
	{
		double tmp;
		ss >> tmp;
		result.push_back(tmp);
	}

	return result;
}

double Norm2(const ::std::vector< double>& doubleVector)
{
	double tmp = 0;

	for (::std::vector<double>::const_iterator it = doubleVector.begin(); it != doubleVector.end(); ++ it)
		tmp += ::std::pow(*it, 2);

	return ::std::sqrt(tmp);
}

::Eigen::MatrixXd PseudoInverse(const ::Eigen::MatrixXd& matrix)
{
	::Eigen::MatrixXd quad = matrix * matrix.transpose();
	
	if (quad.determinant() == 0)
		throw("matrix is close to singularity!!");

	return matrix.transpose() * quad.inverse();
}

void PseudoInverse(const ::Eigen::MatrixXd& inputMatrix, ::Eigen::MatrixXd& outputMatrix, const ::Eigen::MatrixXd& weight)
{
	Eigen::MatrixXd tmp;
	if (weight.size() > 0)
		tmp = inputMatrix.transpose() * weight.transpose() * weight * inputMatrix;
	else
		tmp = inputMatrix * inputMatrix.transpose();
	tmp = tmp.inverse();
	outputMatrix =  inputMatrix.transpose() * tmp;

}

double NormSquared(const ::std::vector<double>& input)
{
	double sum = 0;
	for(::std::vector<double>::const_iterator it = input.begin(); it != input.end(); ++it)
		sum += ::std::pow(*it, 2);

	return sum;
}

::std::string GetDateString()
{
  time_t rawtime;
  struct tm * timeinfo;

  char buffer [80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime (buffer,80,"%Y-%m-%d-%H-%M-%S", timeinfo);

  return ::std::string(buffer);
}


void PrintCArray(const double* toPrint, size_t size, ::std::ostream& os)
{
	for (size_t i = 0; i < size; ++i)
		os << toPrint[i] << " ";

	os << ::std::endl;

}


::Eigen::Vector3d Vec3ToEigen(const Vec3& vec3)
{
	::Eigen::Vector3d vec3Eigen;
	for (int i = 0; i < 3; ++i)
		vec3Eigen(i) = vec3[i];

	return vec3Eigen;
}


void SO3ToEigen(const SO3& rot, ::Eigen::Matrix<double, 3, 3>& rotEigen)
{
	rotEigen.col(0) = Vec3ToEigen(rot.GetX());
	rotEigen.col(1) = Vec3ToEigen(rot.GetY());
	rotEigen.col(2) = Vec3ToEigen(rot.GetZ());
}

void PrintVec3(const Vec3& vecToPrint)
{
	for (int i = 0; i < 3; ++i)
		::std::cout << vecToPrint[i] << " ";
	::std::cout << ::std::endl;
}

std::wstring s2ws(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0); 
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}