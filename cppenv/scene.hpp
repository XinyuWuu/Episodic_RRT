#ifndef __SCENE__
#define __SCENE__

#include <vector>
#include <random>
#include <array>
#include <armadillo>
#include <fcl/fcl.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ur5e.hpp"
#include "ikfast/ur5e.hpp"

class Scene
{
private:
public:
    const double robot_r = 0.2;
    const double min_margin = 0.02;
    std::shared_ptr<fcl::Sphered> robot_geom = std::make_shared<fcl::Sphered>(robot_r + min_margin);
    // std::shared_ptr<fcl::Boxd> robot_geom = std::make_shared<fcl::Boxd>(2 * (robot_r + min_margin), 2 * (robot_r + min_margin), 2 * (robot_r + min_margin));
    fcl::CollisionObjectd robot = fcl::CollisionObjectd(robot_geom);
    // std::shared_ptr<fcl::Boxd> pillar_geom = std::make_shared<fcl::Boxd>(0.1, 0.1, 0.6);
    // fcl::CollisionObjectd pillar = fcl::CollisionObjectd(pillar_geom, Eigen::Matrix3d::Identity(), Eigen::Vector3d({0, 0.1, -0.3}));
    // fcl::CollisionRequest<double> colReq = fcl::CollisionRequest<double>();
    // fcl::CollisionResult<double> colRes = fcl::CollisionResult<double>();
    UR5E ur5e = UR5E();
    ikfast::UR5E ur5eIK = ikfast::UR5E();

    const int dim;
    const int seed;
    const double env_half_size = 30;
    const int npoints = 60;
    const double sep = 2 * env_half_size / npoints;
    const int max_obs = 30600;
    const double min_obs_r = 0.5;
    const double max_obs_r = 2;

    const double max_ph = 0.8;
    const double min_ph_oh = 0.1;
    const double maxd = 0.6;
    const double mind = 0.3;
    const double maxh = 0.6;
    const double minh = 0.2;

    std::vector<int> choices;
    std::mt19937 random_gen;
    std::uniform_real_distribution<double> randu = std::uniform_real_distribution<double>(0.0, 1.0);

    std::vector<std::pair<bool, int>> geomLoc;
    // center[2] radius[1]
    std::vector<std::array<double, 3>> circles;
    // center[2] radius[1] halfx[1] halfy[1] angle[1]
    std::vector<std::array<double, 6>> rectangles;
    // center[3] radius[1]
    std::vector<std::array<double, 4>> spheres;
    // center[3] radius[1] halfx[1] halfy[1] halfz[1] anglex[1] angley[1] anglez[1]
    std::vector<std::array<double, 10>> boxes;
    std::vector<Eigen::Quaterniond> quats;
    std::vector<Eigen::Matrix3d> rotMs;

    std::vector<std::shared_ptr<fcl::CollisionGeometryd>> colGs;
    std::vector<fcl::CollisionObjectd *> colOs;

    fcl::DynamicAABBTreeCollisionManagerd colM = fcl::DynamicAABBTreeCollisionManagerd();
    fcl::NaiveCollisionManagerd colMN = fcl::NaiveCollisionManagerd();
    fcl::DefaultCollisionData<double> colCB = fcl::DefaultCollisionData<double>();
    fcl::DefaultDistanceData<double> disCB = fcl::DefaultDistanceData<double>();

    Scene(const int seed, const int dim = 2);
    ~Scene();
    // circles, rectangles, geomLoc, colGs, colOs, colM
    void addOBS2d(const double min_obs_sep = 0.5);
    // spheres, boxes, quats, rotMs, geomLoc, colGs, colOs, colM
    void addOBS3d(const double min_obs_sep = 0.5);
    // spheres, boxes, quats, rotMs, geomLoc, colGs, colOs, colM, colMN, hardPose
    bool addPanel(const Eigen::Quaterniond &quat, const Eigen::Vector3d &center, const int panelCap = 4, const double min_oh = 0.1, const double max_oh = 0.4, const double panel_depth = 0.025);
    // spheres, boxes, quats, rotMs, geomLoc, colGs, colOs, colM, colMN, hardPose
    bool addOBS6d(const int obsCap = 4);
    std::vector<arma::mat> hardPose, hardPose_;
    arma::mat tightPose = arma::mat(6, 1);
    const int panel_add_seq[4] = {0, 2, 1, 3};

    inline void _setPos2d(const double *pos);
    inline void _setPos3d(const double *pos);
    inline void _setPos6d(const double *pos);
    inline void _setPos6dB(const double *pos);
    inline bool _colRobot();
    inline double _disRobot();
    bool colRobot2D(const double *pos);
    double disRobot2D(const double *pos);
    bool colRobot3D(const double *pos);
    double disRobot3D(const double *pos);
    bool colRobot6D(const double *pos);
    bool colRobot6DB(const double *pos);
    bool colRobot6DBL(const double *pos);
    double disRobot6D(const double *pos);
    bool selfCollide(const double *pos);
    bool selfCollideB(const double *pos);
    bool selfCollideBL(const double *pos);
};
#endif
