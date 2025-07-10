
#include "ur5e.hpp"
namespace ikfast
{
    UR5E::UR5E()
    {
    }
    UR5E::~UR5E() {}
    void UR5E::forward(const IkReal *joints)
    {
        ComputeFk(joints, tran.data(), rot.data());
        rot.transposeInPlace();
    }
    bool UR5E::inverse(const IkReal *tran_, const IkReal *rot_)
    {
        memcpy(rot.data(), rot_, 9 * sizeof(double));
        rot.transposeInPlace();
        solutions.Clear();
        success = ComputeIk(tran_, rot.data(), nullptr, solutions);

        if (!success)
        {
            return false;
        }
        num_of_solutions = solutions.GetNumSolutions();
        if (num_of_solutions > 8)
        {
            return false;
        }

        for (std::size_t i = 0; i < num_of_solutions; ++i)
        {
            solutions.GetSolution(i).GetSolution(solvalues + i * 6, nullptr);
            boundJoints(solvalues + i * 6);
        }
        return true;
    }
}
