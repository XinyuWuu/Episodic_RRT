#ifndef __UR5EIKFAST__
#define __UR5EIKFAST__
#define IKFAST_HAS_LIBRARY
#define IKFAST_NO_MAIN
#define IKFAST_CLIBRARY

#include <Eigen/Dense>
#include "ikfast.h"
#include "../utils.hpp"

namespace ikfast
{
    class UR5E
    {
    public:
        Eigen::Matrix3d rot;
        Eigen::Vector3d tran;
        IkSolutionList<IkReal> solutions = IkSolutionList<IkReal>();
        IkReal solvalues[6 * 8];
        bool success;
        size_t num_of_solutions;
        UR5E();
        ~UR5E();
        void forward(const IkReal *joints);
        bool inverse(const IkReal *tran, const IkReal *rot);
    };
}

#endif
