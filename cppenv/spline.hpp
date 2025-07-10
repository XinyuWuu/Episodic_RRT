#ifndef __SPLINE__
#define __SPLINE__
#include <armadillo>
#include <tinysplinecxx.h>
#include <tinyspline.h>
#include "utils.hpp"

class Spline
{
private:
public:
    tinyspline::BSpline bs;
    tsBSpline bsc, d1c;
    tsStatus e_;
    // 200 is the default value of tinyspline::BSpline::chordLengths
    const int nSamples = 200;
    double *samples, *lengths;
    tinyspline::std_real_vector_out knots;
    tinyspline::std_real_vector_out traj;
    double *trajc = nullptr;
    tinyspline::std_real_vector_out diff;
    double *diffc = nullptr;
    int bsn = 0;
    double splen = 0;
    arma::mat x;
    arma::mat dx;
    arma::mat thetas;
    arma::mat input;
    tsDeBoorNet diffnet;

    Spline();
    ~Spline();
    void interpolate(const std::vector<double> *ps, const int dim, const double dense = 1000, std::vector<double> *first = nullptr, std::vector<double> *last = nullptr, const double alpha = 0.5);
    void interpolate(const double *ps, const int num, const int dim, const double dense = 1000, const double *first = nullptr, const double *last = nullptr, const double alpha = 0.5);
    // void interpolate(const std::vector<double> *ps, const double dense = 180.0 / M_PI, const std::vector<double> *first = nullptr);
    // void interpolate(const double *ps, const int num, const double dense = 180.0 / M_PI, const double *first = nullptr);
    void unboundSO6(double *ps, const int num);
    void boundSO6(double *ps, const int num);
    const double *derive(const int n);
    void derive_all();
    std::vector<double> lens;
    const double weights[6] = {1, 1, 1, 0.3, 0.3, 0.3};
    double distance(const double *j1, const double *j2);
};

#endif
