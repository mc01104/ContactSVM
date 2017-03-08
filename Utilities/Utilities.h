#pragma once
#include "time.h"
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "LieGroup.h"

//namespace utilities
//{
	::std::vector< ::std::string> ReadLinesFromFile(const ::std::string& pathToFile);


	::std::vector< double> DoubleVectorFromString(const ::std::string& inputString);


	template <class T>
	::std::vector<T> operator-(const ::std::vector<T>& lhs, const ::std::vector<T>& rhs)
	{
		
		if (lhs.size() != rhs.size())
			throw("this operation requires vector of the same size");

		::std::vector<T> result;

		for (size_t i = 0; i < lhs.size(); ++i)
			result.push_back(lhs[i] - rhs[i]);

		return result;
	};


	template <class T>
	::std::vector<T>& operator/=(::std::vector< T>& lhs, const ::std::vector< T>& rhs)
	{
		
		if (lhs.size() != rhs.size())
			throw("this operation requires vector of the same size");

		for (size_t i = 0; i < lhs.size(); ++i)
		{
			if (rhs[i] == 0.0)
				throw("cannot divide by zero");

			lhs[i] = lhs[i] / rhs[i];
		}

	};

	template <class T>
	::std::vector<T>& operator /= (::std::vector<T>& lhs, const double rhs)
	{
		for (size_t i = 0; i < lhs.size(); ++i)
		{
			if (rhs == 0.0)
				throw("cannot divide by zero");

			lhs[i] = lhs[i] / rhs;
		}
	}

	template <class T>
	::std::vector<T>& operator*=(::std::vector< T>& lhs, const ::std::vector< T>& rhs)
	{
		
		if (lhs.size() != rhs.size())
			throw("this operation requires vector of the same size");

		for (size_t i = 0; i < lhs.size(); ++i)
			lhs[i] = lhs[i] * rhs[i];

	};


	template <class T>
	::std::vector<T> operator/(const ::std::vector< T>& lhs, const ::std::vector< T>& rhs)
	{
		
		if (lhs.size() != rhs.size())
			throw("this operation requires vector of the same size");

		::std::vector<T> result;

		for (size_t i = 0; i < lhs.size(); ++i)
		{
			if (rhs[i] == 0.0)
				throw("cannot divide by zero");

			result.push_back(lhs[i] / rhs[i]);
		}

		return result;
	};


	template <class T>
	void PrintVector(const ::std::vector<T>& vectorToBePrinted)
	{
		for(::std::vector<T>::const_iterator it = vectorToBePrinted.begin(); it !=  vectorToBePrinted.end(); ++it)
			::std::cout << *it << " ";

		::std::cout << ::std::endl;
	};


	double Norm2(const ::std::vector< double>& doubleVector);


	::Eigen::MatrixXd PseudoInverse(const ::Eigen::MatrixXd& matrix);


	void PseudoInverse(const ::Eigen::MatrixXd& inputMatrix, ::Eigen::MatrixXd& outputMatrix, const ::Eigen::MatrixXd& weight = ::Eigen::MatrixXd());


	template <typename Derived>
	::std::ostream& operator<<(::std::ostream& os, const ::Eigen::EigenBase<Derived>& toPrint)
	{
		::Eigen::IOFormat OctaveFmt(::Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
		os << toPrint << ::std::endl;

		return os;
	}


	double NormSquared(const ::std::vector<double>& input);


	template <typename T>
	::std::ostream& operator<<(::std::ostream& os, const ::std::vector<T>& toPrint)
	{
		for (::std::vector<T>::const_iterator it = toPrint.begin(); it != toPrint.end(); ++it)
			os << *it << " ";
		os << ::std::endl;

		return os;
	};


	::std::string GetDateString();

	void PrintCArray(const double* toPrint, size_t size, ::std::ostream& os = ::std::cout);
//}

void SO3ToEigen(const SO3& rot, ::Eigen::Matrix<double, 3, 3>& rotEigen);

::Eigen::Vector3d Vec3ToEigen(const Vec3& vec3);

void PrintVec3(const Vec3& vecToPrint);

std::wstring s2ws(const std::string& s);



// Useful functions for sorting files by increasing number and not lexicographical order
// usage with a vector of strings: 
// std::sort(vector.begin(), vector.end(), numeric_string_compare);
bool is_not_digit(char c);
bool numeric_string_compare(const std::string& s1, const std::string& s2);



/*
Command line options parsing utility

Usage: 
	If the program i caled program.exe -h for instance, cmdOptionExists(argv, argv+argc, "-h") will return true

	For options with a value (for instance program.exe -i path_of_file), getCmdOption(argv, argv + argc, "-i") will return a char* with the value, that can be cast into the type needed later on

*/
char* getCmdOption(char ** begin, char ** end, const std::string & option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);

bool file_exists(const ::std::string& filename);