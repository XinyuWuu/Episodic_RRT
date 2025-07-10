#ifndef __UR5E__
#define __UR5E__

#include <Eigen/Dense>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <fcl/fcl.h>

class UR5E
{
private:
    // double j1 = 0;
    // Eigen::Quaterniond qj1 = Eigen::Quaterniond(Eigen::AngleAxisd(j1, Eigen::Vector3d::UnitZ()));
    // Eigen::Vector3d tj1 = Eigen::Vector3d(0, 0, 0.1625);
    // double j2 = 0;
    // Eigen::Quaterniond qj2 = qj1 * Eigen::AngleAxisd(j2, Eigen::Vector3d::UnitY());
    // Eigen::Vector3d tj2 = qj1 * Eigen::Vector3d(0, 0.138, 0) + tj1;
    // double j3 = 0;
    // Eigen::Quaterniond qj3 = qj2 * Eigen::AngleAxisd(j3, Eigen::Vector3d::UnitY());
    // Eigen::Vector3d tj3 = qj2 * Eigen::Vector3d(0, 0.007 - 0.138, 0.425) + tj2;
    // double j4 = 0;
    // Eigen::Quaterniond qj4 = qj3 * Eigen::AngleAxisd(j4, Eigen::Vector3d::UnitY());
    // Eigen::Vector3d tj4 = qj3 * Eigen::Vector3d(0, 0, 0.3922) + tj3;
    // double j5 = 0;
    // Eigen::Quaterniond qj5 = qj4 * Eigen::AngleAxisd(j5, Eigen::Vector3d::UnitZ());
    // Eigen::Vector3d tj5 = qj4 * Eigen::Vector3d(0, 0.134 - 0.007, 0) + tj4;
    // double j6 = 0;
    // Eigen::Quaterniond qj6 = qj5 * Eigen::AngleAxisd(j6, Eigen::Vector3d::UnitY());
    // Eigen::Vector3d tj6 = qj5 * Eigen::Vector3d(0, 0, 0.1) + tj5;

    std::string UR5E_parts_name[7] = {
        "base",
        "shoulder",
        "upperarm",
        "forearm",
        "wrist1",
        "wrist2",
        "wrist3"};
    Assimp::Importer importer;

public:
    double joints[6];
    Eigen::Quaterniond quaternions[6], endquat, endquat_;
    Eigen::Vector3d translations[6], translations_[6], axises[6], endpointOffset, endpoint;
    UR5E(/* args */);
    ~UR5E();
    void setJoints(const double *joints_);
    void loadSTL(const char *path);
    void setLinksPoseL(Eigen::Quaterniond *quats, const Eigen::Vector3d *trans);
    void setLinksPoseL();
    void setLinksPoseB(Eigen::Quaterniond *quats, const Eigen::Vector3d *trans);
    void setLinksPoseB();
    void calEndpoint();
    void calEndquat();
    bool selfCollide(fcl::CollisionObjectd **links);
    bool selfCollide(fcl::CollisionObjectd **links, int pair);
    bool collideL(fcl::CollisionObjectd *colO);
    bool collideL(fcl::NaiveCollisionManagerd *colM);
    bool collideB(fcl::CollisionObjectd *colO);
    bool collideB(fcl::NaiveCollisionManagerd *colM);
    bool collideL(fcl::CollisionObjectd *colO, int link);
    bool collideL(fcl::NaiveCollisionManagerd *colM, int link);
    bool collideB(fcl::CollisionObjectd *colO, int link);
    bool collideB(fcl::NaiveCollisionManagerd *colM, int link);
    double distance(fcl::NaiveCollisionManagerd &colM, fcl::CollisionObjectd *colO);

    std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> meshes[7];
    fcl::CollisionObjectd *links[7];
    // simplified version of links
    std::shared_ptr<fcl::CollisionGeometryd> colGs[7];
    fcl::CollisionObjectd *linksB[7];
    Eigen::Vector3d sizes[7], offsets[7];

    std::vector<std::pair<int, int>> selfColPairs;
    fcl::CollisionRequest<double> colReq = fcl::CollisionRequest<double>();
    fcl::CollisionResult<double> colRes = fcl::CollisionResult<double>();
    fcl::NaiveCollisionManagerd colML = fcl::NaiveCollisionManagerd();
    fcl::NaiveCollisionManagerd colMB = fcl::NaiveCollisionManagerd();
    fcl::DefaultCollisionData<double> colCB = fcl::DefaultCollisionData<double>();
    fcl::DefaultDistanceData<double> disCB = fcl::DefaultDistanceData<double>();
};

#endif
