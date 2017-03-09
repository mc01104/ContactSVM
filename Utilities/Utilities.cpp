#include "Utilities.h"

#ifndef LINUX
	#include <Windows.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>


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

#ifndef LINUX
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
#endif



bool is_not_digit(char c)
{
    return !std::isdigit(c);
}

bool numeric_string_compare(const std::string& s1, const std::string& s2)
{
    // handle empty strings...

	const std::string s1_ = s1.substr(0, s1.find_last_of("."));
	const std::string s2_ = s2.substr(0, s2.find_last_of("."));

    std::string::const_iterator it1 = s1_.begin(), it2 = s2_.begin();

    if (std::isdigit(s1_[0]) && std::isdigit(s2_[0])) {

		double n1 = 0;
		std::istringstream ss(s1_);
		ss >> n1;

		double n2 = 0;
		std::istringstream ss2(s2_);
		ss2 >> n2;

        if (n1 != n2) return n1 < n2;

        it1 = std::find_if(s1_.begin(), s1_.end(), is_not_digit);
        it2 = std::find_if(s2_.begin(), s2_.end(), is_not_digit);
    }

    return std::lexicographical_compare(it1, s1_.end(), it2, s2_.end());
}





char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

bool file_exists(const ::std::string& filename)
{
    std::ifstream infile(filename.c_str());
    return infile.good();
}