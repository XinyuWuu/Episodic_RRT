
#include "spline.hpp"

Spline::Spline()
{
    samples = new double[nSamples];
    lengths = new double[nSamples];
    bsc = ts_bspline_init();
    d1c = ts_bspline_init();
}

Spline::~Spline()
{
    delete[] samples;
    delete[] lengths;
}

void Spline::interpolate(const double *ps, const int num, const int dim, const double dense, const double *first, const double *last, const double alpha)
{
    ts_bspline_free(&bsc);
    // bsc = ts_bspline_init();
    if (ts_bspline_interpolate_catmull_rom(ps, num, dim, alpha, first, last, TS_POINT_EPSILON, &bsc, &e_))
    {
        throw std::runtime_error(e_.message);
    }
    ts_bspline_uniform_knot_seq(&bsc, nSamples, samples);
    if (ts_bspline_chord_lengths(&bsc, samples, nSamples, lengths, &e_))
    {
        throw std::runtime_error(e_.message);
    }
    splen = lengths[nSamples - 1];
    bsn = std::ceil(splen * dense) + 1;
    knots.resize(bsn);
    if (ts_chord_lengths_equidistant_knot_seq(samples, lengths, nSamples, bsn, knots.data(), &e_))
    {
        throw std::runtime_error(e_.message);
    }
    free(trajc);
    if (ts_bspline_eval_all(&bsc,
                            knots.data(),
                            knots.size(),
                            &trajc,
                            &e_))
    {
        throw std::runtime_error(e_.message);
    }
    x = arma::mat(trajc, dim, bsn, false, false);
    ts_bspline_free(&d1c);
    // d1c = ts_bspline_init();
    if (ts_bspline_derive(&bsc, 1, TS_POINT_EPSILON, &d1c, &e_))
        throw std::runtime_error(e_.message);
    // free(diffc);
    // if (ts_bspline_eval_all(&d1c,
    //                         knots.data(),
    //                         knots.size(),
    //                         &diffc,
    //                         &e_))
    // {
    //     throw std::runtime_error(e_.message);
    // }
    // dx = arma::mat(diffc, dim, bsn, false, false);
    // if (dim == 2)
    // {
    //     thetas = arma::atan2(dx.row(1), dx.row(0)).eval();
    // }
}

void Spline::derive_all()
{
    free(diffc);
    if (ts_bspline_eval_all(&d1c,
                            knots.data(),
                            knots.size(),
                            &diffc,
                            &e_))
    {
        throw std::runtime_error(e_.message);
    }
    dx = arma::mat(diffc, ts_bspline_dimension(&d1c), bsn, false, false);
    if (ts_bspline_dimension(&d1c) == 2)
    {
        thetas = arma::atan2(dx.row(1), dx.row(0)).eval();
    }
}

const double *Spline::derive(const int n)
{
    if (ts_bspline_eval(&d1c,
                        knots[n],
                        &diffnet,
                        &e_))
    {
        throw std::runtime_error(e_.message);
    }
    return ts_deboornet_result_ptr(&diffnet);
}

void Spline::interpolate(const std::vector<double> *ps, const int dim, const double dense, std::vector<double> *first, std::vector<double> *last, const double alpha)
{
    interpolate(ps->data(), ps->size() / dim, dim, dense, first ? first->data() : nullptr, last ? last->data() : nullptr, alpha);
    // bs = tinyspline::BSpline::interpolateCatmullRom(*ps, dim, alpha, first, last);
    // splen = bs.chordLengths().arcLength();
    // bsn = std::ceil(splen * dense) + 1;
    // knots = bs.equidistantKnotSeq(bsn);
    // traj = bs.evalAll(knots);
    // x = arma::mat(traj.data(), dim, bsn, false, true).eval();
    // diff = bs.derive(1).evalAll(knots);
    // dx = arma::mat(diff.data(), dim, bsn, false, true).eval();
    // if (dim == 2)
    // {
    //     thetas = arma::atan2(dx.row(1), dx.row(0)).eval();
    // }
}

double interpolate_SO2(double from, double to, double t)
{
    double res;
    double diff = to - from;
    if (std::abs(diff) <= M_PI)
        res = from + diff * t;
    else
    {
        if (diff > 0.0)
            diff = 2.0 * M_PI - diff;
        else
            diff = -2.0 * M_PI - diff;
        res = from - diff * t;
        // input states are within bounds, so the following check is sufficient
        if (res > M_PI)
            res -= 2.0 * M_PI;
        else if (res < -M_PI)
            res += 2.0 * M_PI;
    }
    return res;
}

// void Spline::interpolate(const double *ps, const int num, const double dense, const double *first)
// {
//     splen = 0;
//     lens.resize(num);
//     lens[0] = 0;
//     for (size_t i = 0; i < num - 1; i++)
//     {
//         splen += distance(ps + i * 6, ps + i * 6 + 6);
//         lens[i + 1] = splen;
//     }
//     bsn = std::ceil(splen * dense) + 1;
//     traj.resize(bsn * 6);
//     diff.resize(bsn * 6);
//     double lstep = splen / (bsn - 1);
//     x = arma::mat(traj.data(), 6, bsn, false, false);
//     dx = arma::mat(diff.data(), 6, bsn, false, false);
//     input = arma::mat(const_cast<double *>(ps), 6, num, false, false);
//     int idx = 0;
//     double ratio;
//     arma::mat diffNodes(6, 1);
//     diffJoints(input.col(0).colmem, input.col(1).colmem, diffNodes.memptr());
//     for (size_t i = 0; i < bsn; i++)
//     {
//         while (lens[idx + 1] + 1e-6 < i * lstep)
//         {
//             idx++;
//             diffJoints(input.col(idx).colmem, input.col(idx + 1).colmem, diffNodes.memptr());
//         }
//         if (lens[idx + 1] - lens[idx] < 1e-6)
//         {
//             ratio = 0;
//         }
//         else
//         {
//             ratio = ((i * lstep) - lens[idx]) / (lens[idx + 1] - lens[idx]);
//         }
//         // std::cout << idx << "/" << input.n_cols << "," << ratio << std::endl;
//         // std::cout << i * lstep << "/" << lens[idx] << "/" << lens[idx + 1] << std::endl;
//         // x.col(i) = (1 - ratio) * input.col(idx) + ratio * input.col(idx + 1);
//         x.col(i) = input.col(idx) + diffNodes * ratio;
//         boundJoints(x.colptr(i));
//         // for (size_t j = 0; j < 6; j++)
//         // {
//         //     x.col(i)[j] = interpolate_SO2(input.col(idx)[j], input.col(idx + 1)[j], ratio);
//         // }
//     }
//     for (size_t i = 1; i < bsn; i++)
//     {
//         // dx.col(i) = x.col(i) - x.col(i - 1);
//         diffJoints(x.col(i - 1).colmem, x.col(i).colmem, dx.colptr(i));
//     }
//     if (first)
//     {
//         // input = arma::mat(const_cast<double *>(first), 6, 1, true, true);
//         // dx.col(0) = dx.col(0) - input.col(0);
//         diffJoints(first, x.col(0).colmem, dx.colptr(0));
//     }
//     else
//     {
//         dx.col(0).fill(0);
//     }
// }

// void Spline::interpolate(const std::vector<double> *ps, const double dense, const std::vector<double> *first)
// {
//     interpolate(ps->data(), ps->size() / 6, dense, first ? first->data() : nullptr);
//     // splen = 0;
//     // lens.resize(ps->size() / 6);
//     // lens[0] = 0;
//     // for (size_t i = 0; i < ps->size() / 6 - 1; i++)
//     // {
//     //     splen += distance(ps->data() + i * 6, ps->data() + i * 6 + 6);
//     //     lens[i + 1] = splen;
//     // }
//     // bsn = std::ceil(splen * dense) + 1;
//     // traj.resize(bsn * 6);
//     // diff.resize(bsn * 6);
//     // double lstep = splen / (bsn - 1);
//     // int idx = 0;
//     // x = arma::mat(traj.data(), 6, bsn, false, true);
//     // dx = arma::mat(diff.data(), 6, bsn, false, true);
//     // input = arma::mat(const_cast<double *>(ps->data()), 6, ps->size() / 6, true, true);
//     // double ratio;
//     // for (size_t i = 0; i < bsn; i++)
//     // {
//     //     while (lens[idx + 1] + 1e-10 < i * lstep)
//     //     {
//     //         idx++;
//     //     }
//     //     ratio = ((i * lstep) - lens[idx]) / (lens[idx + 1] - lens[idx]);
//     //     // std::cout << idx + 1 << "/" << input.n_cols << "," << ratio << std::endl;
//     //     // std::cout << i * lstep << "/" << lens[idx] << "/" << lens[idx + 1] << std::endl;
//     //     // x.col(i) = (1 - ratio) * input.col(idx) + ratio * input.col(idx + 1);
//     //     for (size_t j = 0; j < 6; j++)
//     //     {
//     //         x.col(i)[j] = interpolate_SO2(input.col(idx)[j], input.col(idx + 1)[j], ratio);
//     //     }
//     // }
//     // for (size_t i = 1; i < bsn; i++)
//     // {
//     //     dx.col(i) = x.col(i) - x.col(i - 1);
//     // }
//     // if (first)
//     // {
//     //     input = arma::mat(const_cast<double *>(first->data()), 6, 1, true, true);
//     //     dx.col(0) = dx.col(0) - input.col(0);
//     // }
//     // else
//     // {
//     //     dx.col(0).fill(0);
//     // }
// }

double Spline::distance(const double *j1, const double *j2)
{
    static double sumlen, jlen;
    sumlen = 0;
    for (size_t i = 0; i < 6; i++)
    {
        jlen = std::abs(j1[i] - j2[i]);
        jlen = jlen > M_PI ? 2 * M_PI - jlen : jlen;
        sumlen += jlen * weights[i];
    }
    return sumlen;
}

void Spline::unboundSO6(double *ps, const int num)
{
    double diff;
    for (size_t i = 1; i < num; i++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            diffSO2(ps + (i - 1) * 6 + j, ps + i * 6 + j, &diff);
            *(ps + i * 6 + j) = *(ps + (i - 1) * 6 + j) + diff;
        }
    }
}

void Spline::boundSO6(double *ps, const int num)
{
    for (size_t i = 0; i < num; i++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            boundSO2(ps + i * 6 + j);
        }
    }
}
