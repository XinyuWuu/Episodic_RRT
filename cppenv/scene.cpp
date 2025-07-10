#include "scene.hpp"

Scene::Scene(const int seed, const int dim) : seed(seed), dim(dim)
{
    arma::arma_rng().set_seed(seed);
    random_gen.seed(seed);
    std::srand(seed);
    if (dim == 2 || dim == 3)
    {
        choices.resize(std::pow(npoints, dim) * 2);
        for (size_t i = 0; i < choices.size(); i++)
        {
            choices[i] = i;
        }
    }

    colCB.request.enable_contact = 0;
    colCB.request.num_max_contacts = 100000;
    // colReq.enable_contact = 0;
    // colReq.num_max_contacts = 100000;
}

Scene::~Scene()
{
    geomLoc.clear();
    colGs.clear();
    for (auto p : colOs)
    {
        delete p;
    }
    colOs.clear();
    circles.clear();
    rectangles.clear();
    spheres.clear();
    boxes.clear();
    quats.clear();
    rotMs.clear();
}

void Scene::addOBS2d(const double min_obs_sep)
{
    std::shuffle(choices.begin(), choices.end(), random_gen);
    auto center = arma::Mat<double>(dim, 1);
    bool shape;
    double radius, lx, ly, angle, ratio;
    std::shared_ptr<fcl::CollisionGeometryd> colG;
    fcl::CollisionObjectd *colO;
    Eigen::Quaterniond quat;
    Eigen::Matrix3d rotM;

    for (auto i : choices)
    {
        center[0] = (i / 2) / npoints;
        center[1] = (i / 2) % npoints;
        center = (center + randu(random_gen)) * sep - env_half_size;
        if (i % 2)
        {
            center = center + sep / 2;
        }
        shape = randu(random_gen) > 0.5;
        radius = randu(random_gen) * (max_obs_r - min_obs_r) + min_obs_r;
        radius = radius * (shape ? 1 : 1.2); // balance the number of two shapes

        if (!shape)
        {
            ratio = randu(random_gen) * (0.85 - 0.15) + 0.15;
            lx = std::sqrt(ratio * radius * radius);
            ly = std::sqrt((1 - ratio) * radius * radius);
            angle = (randu(random_gen) * 2 - 1) * M_PI;
        }

        if (shape)
        {
            // circle
            quat = quat.Identity();
            colG = std::make_shared<fcl::Sphered>(radius);
        }
        else
        {
            // rectangle
            quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
            colG = std::make_shared<fcl::Boxd>(2 * lx, 2 * ly, 1);
        }
        rotM = quat.toRotationMatrix();
        colO = new fcl::CollisionObjectd(colG, rotM, Eigen::Vector3d(center[0], center[1], 0));

        disCB.result.clear();
        colO->computeAABB();
        colM.distance(colO, &disCB, fcl::DefaultDistanceFunction);
        if (disCB.result.min_distance > min_obs_sep)
        {
            colGs.push_back(colG);
            colOs.push_back(colO);
            colM.registerObject(colO);
            colM.setup();
            if (shape)
            {
                circles.push_back(std::array<double, 3>({center[0], center[1], radius}));
            }
            else
            {
                rectangles.push_back(std::array<double, 6>({center[0], center[1], radius, lx, ly, angle}));
                quats.push_back(quat);
                rotMs.push_back(rotM);
            }
            geomLoc.push_back(std::pair<bool, int>({shape, shape ? circles.size() - 1 : rectangles.size() - 1}));
        }
        else
        {
            delete colO;
        }
        // enough
        if (colGs.size() >= max_obs)
        {
            break;
        }
    }
    // std::cout << circles.size() << "," << rectangles.size() << std::endl;
}

void Scene::addOBS3d(const double min_obs_sep)
{
    std::shuffle(choices.begin(), choices.end(), random_gen);
    auto center = arma::Mat<double>(dim, 1);
    bool shape;
    double radius, lx, ly, lz, anglex, angley, anglez, ratio1, ratio2;
    std::shared_ptr<fcl::CollisionGeometryd> colG;
    fcl::CollisionObjectd *colO;
    Eigen::Quaterniond quat;
    Eigen::Matrix3d rotM;

    for (auto i : choices)
    {
        center[0] = (i / 2) / (npoints * npoints);
        center[1] = (i / 2) % (npoints * npoints) / npoints;
        center[2] = (i / 2) % (npoints * npoints) % npoints;
        center = (center + randu(random_gen)) * sep - env_half_size;
        if (i % 2)
        {
            center = center + sep / 2;
        }
        shape = randu(random_gen) > 0.5;
        radius = randu(random_gen) * (max_obs_r - min_obs_r) + min_obs_r;
        radius = radius * (shape ? 1 : 1.3); // balance the number of two shapes

        if (!shape)
        {
            ratio1 = randu(random_gen) * (0.85 - 0.15) + 0.15;
            ratio2 = randu(random_gen) * (0.85 - 0.15) + 0.15;
            lx = std::sqrt(ratio1 * radius * radius * radius);
            ly = std::sqrt(ratio2 * (1 - ratio1) * radius * radius * radius);
            lz = std::sqrt((1 - ratio2) * (1 - ratio1) * radius * radius * radius);
            anglex = (randu(random_gen) * 2 - 1) * M_PI;
            angley = (randu(random_gen) * 2 - 1) * M_PI;
            anglez = (randu(random_gen) * 2 - 1) * M_PI;
        }

        if (shape)
        {
            // sphere
            quat = quat.Identity();
            colG = std::make_shared<fcl::Sphered>(radius);
        }
        else
        {
            // box
            quat = rpyToQuat(anglex, angley, anglez);
            colG = std::make_shared<fcl::Boxd>(2 * lx, 2 * ly, 2 * lz);
        }
        rotM = quat.toRotationMatrix();
        colO = new fcl::CollisionObjectd(colG, rotM, fcl::Vector3d(center[0], center[1], center[2]));

        colO->computeAABB();
        colCB.result.clear();
        colM.collide(colO, &colCB, fcl::DefaultCollisionFunction);
        if (!colCB.result.isCollision())
        {
            disCB.result.clear();
            colM.distance(colO, &disCB, fcl::DefaultDistanceFunction);
        }
        if (!colCB.result.isCollision() &&
            disCB.result.min_distance > min_obs_sep)
        {
            colGs.push_back(colG);
            colOs.push_back(colO);
            colM.registerObject(colO);
            colM.setup();
            if (shape)
            {
                spheres.push_back(std::array<double, 4>({center[0], center[1], center[2], radius}));
            }
            else
            {
                boxes.push_back(std::array<double, 10>({center[0], center[1], center[2], radius, lx, ly, lz, anglex, angley, anglez}));
                quats.push_back(quat);
                rotMs.push_back(rotM);
            }
            geomLoc.push_back(std::pair<bool, int>({shape, shape ? spheres.size() - 1 : boxes.size() - 1}));
        }
        else
        {
            delete colO;
        }
        // enough
        if (colGs.size() >= max_obs)
        {
            break;
        }
    }
    // std::cout << spheres.size() << "," << boxes.size() << std::endl;
}

bool Scene::addPanel(const Eigen::Quaterniond &quat, const Eigen::Vector3d &center, const int panelCap, const double min_oh, const double max_oh, const double panel_depth)
{
    // panel half size, half open size
    double phx, phy, ohx, ohy;

    ohx = randu(random_gen) * (max_oh / 1.5 - min_oh / 1.5) + min_oh / 1.5;
    ohy = randu(random_gen) * (max_oh - min_oh) + min_oh;
    const double min_ph = std::max(ohx, ohy) + min_ph_oh; // require oh + ph_oh < ph => min_oh + min_ph_oh < min_ph
    phx = randu(random_gen) * (max_ph - min_ph) + min_ph;
    phy = randu(random_gen) * (max_ph - min_ph) + min_ph;

    // row, pitch, yaw
    auto rpy = quat.toRotationMatrix().eulerAngles(0, 1, 2);

    Eigen::Vector3d centers[4], axises[4];
    double panel_depth_;
    Eigen::Vector3d center_;
    std::array<double, 10> panel[4];
    std::shared_ptr<fcl::CollisionGeometryd> pcolGs[4];
    fcl::CollisionObjectd *pcolOs[4];
    centers[0] = {0, (phy + ohy) / 2, 0};
    axises[0] = {phx, (phy - ohy) / 2, 0};
    centers[1] = {0, -(phy + ohy) / 2, 0};
    axises[1] = {phx, (phy - ohy) / 2, 0};
    centers[2] = {(phx + ohx) / 2, 0, 0};
    axises[2] = {(phx - ohx) / 2, ohy, 0};
    centers[3] = {-(phx + ohx) / 2, 0, 0};
    axises[3] = {(phx - ohx) / 2, ohy, 0};

    for (size_t i = 0; i < 4; i++)
    {
        panel_depth_ = (randu(random_gen) + 1) * panel_depth / 2;
        axises[i][2] = panel_depth_;
        centers[i][2] = panel_depth_;
        center_ = quat * centers[i] + center;
        panel[i] = std::array<double, 10>({center_.x(), center_.y(), center_.z(), axises[i].norm(),
                                           axises[i].x(), axises[i].y(), axises[i].z(), rpy.x(), rpy.y(), rpy.z()});
    }
    for (size_t i = 0; i < 4; i++)
    {
        pcolGs[i] = std::make_shared<fcl::Boxd>(2 * panel[i][4], 2 * panel[i][5], 2 * panel[i][6]);
        pcolOs[i] = new fcl::CollisionObjectd(pcolGs[i], quat.toRotationMatrix(), fcl::Vector3d(panel[i][0], panel[i][1], panel[i][2]));
        pcolOs[i]->computeAABB();
    }

    ur5e.setJoints(tightPose.mem);
    ur5e.setLinksPoseL();
    for (size_t i = 0; i < 4; i++)
    {
        if (ur5e.collideL(pcolOs[i]))
        {
            for (size_t j = 0; j < 4; j++)
            {
                delete pcolOs[j];
                pcolOs[j] = nullptr;
            }
            return false;
        }
    }

    // for (size_t i = 0; i < 4; i++)
    // {
    //     colRes.clear();
    //     fcl::collide(&pillar, pcolOs[i], colReq, colRes);
    //     if (colRes.isCollision())
    //     {
    //         for (size_t j = 0; j < 4; j++)
    //         {
    //             delete pcolOs[j];
    //             pcolOs[j] = nullptr;
    //         }
    //         return false;
    //     }
    // }
    int idx = 0;
    for (size_t i = 0; i < 4; i++)
    {
        idx = panel_add_seq[i];
        if (i + 1 > panelCap)
        {
            delete pcolOs[idx];
            pcolOs[idx] = nullptr;
            continue;
        }
        boxes.push_back(panel[idx]);
        quats.push_back(quat);
        rotMs.push_back(quat.toRotationMatrix());
        colGs.push_back(pcolGs[idx]);
        colOs.push_back(pcolOs[idx]);
        geomLoc.push_back(std::pair<bool, int>({false, boxes.size() - 1}));
        colM.registerObject(pcolOs[idx]);
        colMN.registerObject(pcolOs[idx]);
    }

    // double radius = (randu(random_gen) * 0.5 + 0.5) * 0.1;
    // int t = std::floor(randu(random_gen) * 4);
    // center_ = centers[t] + Eigen::Vector3d({0, 0, -axises[t].z() - radius * (1 + randu(random_gen))});
    // center_ = quat * center_ + center;
    // // center_ = {1, 1, 1};
    // spheres.push_back(std::array<double, 4>({center_[0], center_[1], center_[2], radius}));
    // colGs.push_back(std::make_shared<fcl::Sphered>(radius));
    // colOs.push_back(new fcl::CollisionObjectd(colGs.back(), Eigen::Quaterniond::Identity().toRotationMatrix(), fcl::Vector3d(center_[0], center_[1], center_[2])));
    // colOs.back()->computeAABB();
    // geomLoc.push_back(std::pair<bool, int>({true, spheres.size() - 1}));
    // colM.registerObject(colOs.back());
    // colMN.registerObject(colOs.back());
    // colM.setup();
    // colMN.setup();
    int seg = 0;
    double ratio = 0;
    for (size_t i = 0; i < panelCap; i++)
    {
        idx = panel_add_seq[i];
        seg = std::round(2 * std::max(axises[idx].x(), axises[idx].y()) / 0.1);
        seg = std::max(seg, 2);
        for (size_t s = 0; s < seg; s++)
        {
            ratio = (double(s) / seg - 0.5) * 2;
            center_ = centers[idx] + Eigen::Vector3d({axises[idx].x() * ratio, axises[idx].y() * ratio, -axises[idx].z() - 0.03});
            center_ = quat * center_ + center;
            if (ur5eIK.inverse(center_.data(), quat.toRotationMatrix().data()))
            {
                for (size_t j = 0; j < ur5eIK.num_of_solutions; j++)
                {
                    hardPose_.push_back(arma::mat(ur5eIK.solvalues + j * 6, 6, 1, true));
                }
            }
        }
    }
    return true;
}

bool Scene::addOBS6d(const int obsCap)
{
    double sep = 2 * M_PI / obsCap;
    double offset = sep * 2;
    double theta;
    double radius;
    Eigen::Vector3d center, center_;
    Eigen::Quaterniond quat;
    std::shared_ptr<fcl::CollisionGeometryd> colG;
    fcl::CollisionObjectd *colO;
    int added = 0;
    for (size_t i = 0; i < 20; i++)
    {
        if (added >= obsCap)
        {
            break;
        }
        if (i % obsCap == 0)
        {
            offset /= 2;
        }

        theta = offset + sep * i;
        center = {randu(random_gen) * 0.3 + 0.3,
                  0, randu(random_gen) * 0.3 + 0.3};
        center = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()) * center;
        radius = (randu(random_gen) * 0.5 + 0.5) * 0.1;
        colG = std::make_shared<fcl::Sphered>(radius);
        colO = new fcl::CollisionObjectd(colG, Eigen::Quaterniond::Identity().toRotationMatrix(), fcl::Vector3d(center[0], center[1], center[2]));
        colO->computeAABB();
        colCB.result.clear();
        colM.collide(colO, &colCB, fcl::DefaultCollisionFunction);
        ur5e.setJoints(tightPose.mem);
        ur5e.setLinksPoseL();
        if (colCB.result.isCollision() || ur5e.collideL(colO) || ur5e.distance(ur5e.colMB, colO) < 0.1)
        {
            delete colO;
        }
        else
        {
            colGs.push_back(colG);
            colOs.push_back(colO);
            spheres.push_back(std::array<double, 4>({center[0], center[1], center[2], radius}));
            geomLoc.push_back(std::pair<bool, int>({true, spheres.size() - 1}));
            colM.registerObject(colO);
            colMN.registerObject(colO);
            colM.setup();
            colMN.setup();

            // add hardpose
            for (size_t _ = 0; _ < 5; _++)
            {
                quat = Eigen::Quaterniond::UnitRandom();
                center_ = quat * (-Eigen::Vector3d::UnitZ()) * (radius + 0.03) + center;
                if (ur5eIK.inverse(center_.data(), quat.toRotationMatrix().data()))
                {
                    for (size_t j = 0; j < ur5eIK.num_of_solutions; j++)
                    {
                        hardPose_.push_back(arma::mat(ur5eIK.solvalues + j * 6, 6, 1, true));
                    }
                }
            }
            added++;
        }
    }
    return added > 0;
}

inline void Scene::_setPos2d(const double *pos)
{
    robot.setTranslation(fcl::Vector3d(pos[0], pos[1], 0));
    robot.computeAABB();
}

inline void Scene::_setPos3d(const double *pos)
{
    robot.setTranslation(fcl::Vector3d(pos[0], pos[1], pos[2]));
    robot.computeAABB();
}

inline void Scene::_setPos6d(const double *pos)
{
    ur5e.setJoints(pos);
    ur5e.setLinksPoseL();
}

inline void Scene::_setPos6dB(const double *pos)
{
    ur5e.setJoints(pos);
    ur5e.setLinksPoseB();
}

inline bool Scene::_colRobot()
{
    colCB.result.clear();
    colM.collide(&robot, &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

inline double Scene::_disRobot()
{
    disCB.result.clear();
    colM.distance(&robot, &disCB, fcl::DefaultDistanceFunction);
    return disCB.result.min_distance;
}

bool Scene::colRobot2D(const double *pos)
{
    _setPos2d(pos);
    return _colRobot();
}

double Scene::disRobot2D(const double *pos)
{
    _setPos2d(pos);
    return _disRobot();
}

bool Scene::colRobot3D(const double *pos)
{
    _setPos3d(pos);
    return _colRobot();
}

double Scene::disRobot3D(const double *pos)
{
    _setPos3d(pos);
    return _disRobot();
}

bool Scene::colRobot6D(const double *pos)
{
    _setPos6d(pos);
    if (ur5e.selfCollide(ur5e.links))
    {
        return true;
    }
    return ur5e.collideL(&colMN);
}

bool Scene::colRobot6DB(const double *pos)
{
    _setPos6dB(pos);
    if (ur5e.selfCollide(ur5e.linksB))
    {
        return true;
    }
    return ur5e.collideB(&colMN);
}

bool Scene::colRobot6DBL(const double *pos)
{
    ur5e.setJoints(pos);
    ur5e.setLinksPoseB();
    ur5e.setLinksPoseL();
    // if (ur5e.selfCollide(ur5e.linksB) && ur5e.selfCollide(ur5e.links))
    // {
    //     return true;
    // }
    // return ur5e.collideB(&colMN) && ur5e.collideL(&colMN);
    for (size_t i = 0; i < ur5e.selfColPairs.size(); i++)
    {
        if (ur5e.selfCollide(ur5e.linksB, i) && ur5e.selfCollide(ur5e.links, i))
        {
            return true;
        }
    }
    for (size_t i = 0; i < 7; i++)
    {
        if (ur5e.collideB(&colMN, i) && ur5e.collideL(&colMN, i))
        {
            return true;
        }
    }
    return false;
}

bool Scene::selfCollide(const double *pos)
{
    _setPos6d(pos);
    return ur5e.selfCollide(ur5e.links);
}

bool Scene::selfCollideB(const double *pos)
{
    _setPos6dB(pos);
    return ur5e.selfCollide(ur5e.linksB);
}

bool Scene::selfCollideBL(const double *pos)
{
    ur5e.setJoints(pos);
    ur5e.setLinksPoseB();
    ur5e.setLinksPoseL();
    for (size_t i = 0; i < ur5e.selfColPairs.size(); i++)
    {
        if (ur5e.selfCollide(ur5e.linksB, i) && ur5e.selfCollide(ur5e.links, i))
        {
            return true;
        }
    }
    return false;
}

double Scene::disRobot6D(const double *pos)
{
    _setPos6d(pos);
    if (ur5e.selfCollide(ur5e.links))
    {
        return -1;
    }
    double min_dis = 10000000;
    double dis;
    for (auto o : colOs)
    {
        dis = ur5e.distance(ur5e.colML, o);
        if (dis < min_dis)
        {
            min_dis = dis;
        }
    }
    return min_dis;
}
