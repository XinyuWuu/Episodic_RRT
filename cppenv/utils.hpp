#pragma once
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <armadillo>
#include <Eigen/Dense>
#include <string>
#include <iterator>

inline Eigen::Quaterniond rpyToQuat(double roll, double pitch, double yaw)
{
    return Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
}

inline void diffSO2(const double *j1, const double *j2, double *res)
{
    *res = *j2 - *j1;
    if (*res > M_PI)
    {
        *res -= 2 * M_PI;
    }
    else if (*res < -M_PI)
    {
        *res += 2 * M_PI;
    }
}

inline void diffJoints(const double *j1, const double *j2, double *out)
{
    for (size_t i = 0; i < 6; i++)
    {
        diffSO2(j1 + i, j2 + i, out + i);
    }
}

inline void boundSO2(double *j)
{
    if (*j < -M_PI)
    {
        *j += 2 * M_PI;
    }
    else if (*j > M_PI)
    {
        *j -= 2 * M_PI;
    }
}

inline void boundJoints(double *joints)
{
    for (size_t i = 0; i < 6; i++)
    {
        boundSO2(joints + i);
    }
}

std::string readFileToString(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        return ""; // Or throw an exception
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

bool saveStringToFile(const std::string &text, const std::string &filename)
{
    std::ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    outputFile << text;

    outputFile.close();
    return true;
}

namespace YAML
{
    template <>
    struct convert<arma::Mat<double>>
    {
        static Node encode(const arma::Mat<double> &matrix)
        {
            Node node;
            node["rows"] = matrix.n_rows;
            node["cols"] = matrix.n_cols;
            Node data;
            for (int i = 0; i < matrix.n_rows; ++i)
            {
                for (int j = 0; j < matrix.n_cols; ++j)
                {
                    data.push_back(matrix(i, j));
                }
            }
            node["data"] = data;
            return node;
        }

        static bool decode(const Node &node, arma::Mat<double> &matrix)
        {
            if (!node["rows"] || !node["cols"] || !node["data"])
            {
                return false;
            }

            int rows = node["rows"].as<int>();
            int cols = node["cols"].as<int>();
            const Node &data = node["data"];

            if (data.size() != rows * cols)
            {
                return false;
            }

            matrix.set_size(rows, cols);
            int k = 0;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    matrix(i, j) = data[k].as<double>();
                    k++;
                }
            }
            return true;
        }
    };

    template <>
    struct convert<Eigen::Matrix3d>
    {
        static Node encode(const Eigen::Matrix3d &matrix)
        {
            Node node;
            node["rows"] = 3;
            node["cols"] = 3;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    node["data"].push_back(matrix(i, j));
                }
            }
            return node;
        }

        static bool decode(const Node &node, Eigen::Matrix3d &matrix)
        {
            if (!node.IsMap() || !node["rows"] || !node["cols"] || !node["data"])
            {
                return false;
            }

            int rows = node["rows"].as<int>();
            int cols = node["cols"].as<int>();
            if (rows != 3 || cols != 3)
                return false;

            const Node &data = node["data"];
            if (!data.IsSequence() || data.size() != 9)
                return false;

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    matrix(i, j) = data[i * 3 + j].as<double>();
                }
            }
            return true;
        }
    };

    template <>
    struct convert<Eigen::Quaterniond>
    {
        static Node encode(const Eigen::Quaterniond &rhs)
        {
            Node node;
            node.push_back(rhs.x());
            node.push_back(rhs.y());
            node.push_back(rhs.z());
            node.push_back(rhs.w());
            return node;
        }

        static bool decode(const Node &node, Eigen::Quaterniond &rhs)
        {
            if (!node.IsSequence() || node.size() != 4)
            {
                return false;
            }

            rhs.x() = node[0].as<double>();
            rhs.y() = node[1].as<double>();
            rhs.z() = node[2].as<double>();
            rhs.w() = node[3].as<double>();
            return true;
        }
    };
}
