#include "simulation.hpp"
#include <yaml-cpp/yaml.h>

void Simulation::genStartgoal2D(const double start_radius, const double safe_margin, const int mode)
{
    arma::mat pmax, pmin;
    // start_radius + random noise
    double radius, theta;
    radius = start_radius;
    radius = std::clamp(radius, -max_radius, max_radius);
    do
    {
        if (mode == 1)
        {
            for (size_t i = 0; i < dim; i++)
            {
                startp[i] = radius * (randu(random_gen) > 0.5 ? 1 : -1);
            }
            startp = startp + 2 * (arma::randu(dim, 1) * 2 - 1);
        }
        else
        {
            theta = randu(random_gen) * 2 * M_PI;
            startp[0] = radius * std::cos(theta);
            startp[1] = radius * std::sin(theta);
        }
        startp = startp.clamp(-max_radius, max_radius);
        goalp = (goalp.randu() * 2 - 1) - startp;
        goalp = goalp.clamp(-max_radius, max_radius);
        pmax = max_radius - arma::max(startp, goalp);
        pmin = -max_radius - arma::min(startp, goalp);
        // change pmax to random offset
        pmax = pmax.randu() % (pmax - pmin) + pmin;
        startp += pmax;
        goalp += pmax;
        if (disRobot2D(startp.mem) < safe_margin)
        {
            continue;
        }
        if (disRobot2D(goalp.mem) > safe_margin)
        {
            break;
        }
    } while (true);
}

void Simulation::genStartgoal3D(const double start_radius, const double safe_margin, const int mode)
{
    arma::mat pmax, pmin;
    // start_radius + random noise
    double radius, theta1, theta2;
    radius = start_radius;
    radius = std::clamp(radius, -max_radius, max_radius);
    do
    {
        if (mode == 1)
        {
            for (size_t i = 0; i < dim; i++)
            {
                startp[i] = radius * (randu(random_gen) > 0.5 ? 1 : -1);
            }
            startp = startp + 2 * (arma::randu(dim, 1) * 2 - 1);
        }
        else
        {
            theta1 = randu(random_gen) * 2 * M_PI;
            theta2 = randu(random_gen) * 2 * M_PI;
            startp[0] = radius * std::sin(theta1) * std::cos(theta2);
            startp[1] = radius * std::sin(theta1) * std::sin(theta2);
            startp[2] = radius * std::cos(theta1);
        }
        startp = startp.clamp(-max_radius, max_radius);
        goalp = (arma::randu(dim, 1) * 2 - 1) - startp;
        goalp = goalp.clamp(-max_radius, max_radius);
        pmax = max_radius - arma::max(startp, goalp);
        pmin = -max_radius - arma::min(startp, goalp);
        // change pmax to random offset
        pmax = arma::randu(dim, 1) % (pmax - pmin) + pmin;
        startp += pmax;
        goalp += pmax;
        if (disRobot3D(startp.mem) < safe_margin)
        {
            continue;
        }
        if (disRobot3D(goalp.mem) > safe_margin)
        {
            break;
        }
    } while (true);
}

bool Simulation::genStartgoal6D(const int mode, bool invert)
{
    invertsg = invert;
    double j0offset;
    int loop = 0;
    do
    {
        loop++;
        if (loop > 1e3)
        {
            return false;
        }
        startp = (arma::randu(dim, 1) - 0.5) * 2 * M_PI;
        j0offset = (randu(random_gen) * 0.75 + 0.25) * M_PI;
        j0offset = randu(random_gen) > 0.5 ? j0offset : -j0offset;
        startp[0] = tightPose[0] + j0offset;
        boundSO2(startp.memptr());
        startp[1] = startp[1] / 2.0;
        startp[2] = startp[2] / 2.0;
        if (!colRobot6DBL(startp.mem))
        {
            break;
        }
    } while (true);
    if (mode == 0) // random between tight and hard
    {
        if (randu(random_gen) > 0.5 || hardPose.size() == 0)
        {
            goalp = tightPose;
        }
        else
        {
            goalp = hardPose[rand() % hardPose.size()];
        }
    }
    else if (mode == 1) // tight
    {
        goalp = tightPose;
    }
    else // hard
    {
        if (hardPose.size() < 1)
        {
            return false;
        }
        goalp = hardPose[rand() % hardPose.size()];
    }
    if (invertsg)
    {
        auto tmp = goalp;
        goalp = startp;
        startp = tmp;
    }

    ur5e.setJoints(startp.mem);
    ur5e.calEndpoint();
    memcpy(endStartp.memptr(), ur5e.endpoint.data(), 3 * sizeof(double));
    ur5e.setJoints(goalp.mem);
    ur5e.calEndpoint();
    memcpy(endGoalp.memptr(), ur5e.endpoint.data(), 3 * sizeof(double));
    return true;
}

void Simulation::setTraj2D(const double *traj, const int num, const double *first, const double *last)
{
    traj1 = arma::mat(const_cast<double *>(traj), dim, num, false, true);
    traj2 = arma::mat(dim, num + 1);
    traj2.col(0) = pos;
    traj2.cols(1, num) = traj1.each_col() + pos;
}

void Simulation::step2D(const double *traj, const int num)
{
    setTraj2D(traj, num);
    spline.interpolate(traj2.mem, num + 1, dim, traj_dense, lastPos.mem);
    for (size_t i = 0; i < spline.bsn; i++)
    {
        collided = colRobot2D(spline.x.col(i).colmem);
        endT = (i == spline.bsn - 1) && !collided;
        dis2goal = arma::norm(spline.x.col(i) - goalp);
        reached = dis2goal < 0.1;
        valid_idx = i;
        if (endT or collided or reached)
        {
            if (i > 0)
            {
                lastPos = spline.x.col(i - 1);
                // lastVel = spline.dx.col(i - 1);
                memcpy(lastVel.memptr(), spline.derive(i - 1), dim * sizeof(double));
                pos = spline.x.col(i);
                // vel = spline.dx.col(i);
                memcpy(vel.memptr(), spline.derive(i), dim * sizeof(double));
            }
            break;
        }
    }
}

void Simulation::setTraj3D(const double *traj, const int num, const double *first, const double *last)
{
    traj1 = arma::mat(const_cast<double *>(traj), dim, num, false, true);
    traj2 = arma::mat(dim, num + 1);
    traj2.col(0) = pos;
    traj2.cols(1, num) = traj1.each_col() + pos;
}

void Simulation::step3D(const double *traj, const int num)
{
    setTraj3D(traj, num);
    spline.interpolate(traj2.mem, num + 1, dim, traj_dense, lastPos.mem);
    for (size_t i = 0; i < spline.bsn; i++)
    {
        collided = colRobot3D(spline.x.col(i).colmem);
        endT = (i == spline.bsn - 1) && !collided;
        dis2goal = arma::norm(spline.x.col(i) - goalp);
        reached = dis2goal < 0.1;
        valid_idx = i;
        if (endT or collided or reached)
        {
            if (i > 0)
            {
                lastPos = spline.x.col(i - 1);
                // lastVel = spline.dx.col(i - 1);
                memcpy(lastVel.memptr(), spline.derive(i - 1), dim * sizeof(double));
                pos = spline.x.col(i);
                // vel = spline.dx.col(i);
                memcpy(vel.memptr(), spline.derive(i), dim * sizeof(double));
            }
            break;
        }
    }
}

void Simulation::setTraj6D(const double *traj, const int num, const double *first)
{
    traj1 = arma::mat(const_cast<double *>(traj), dim, num, false, true);
    traj2 = arma::mat(dim, num + 1);
    traj2.col(0) = pos;
    // traj2.cols(1, num) = traj1;
    traj2.cols(1, num) = traj1.each_col() + pos;
}

void Simulation::step6D(const double *traj, const int num)
{
    setTraj6D(traj, num);
    spline.interpolate(traj2.mem, num + 1, dim, traj_dense, lastPos.mem);
    spline.boundSO6(spline.x.memptr(), spline.x.n_cols);
    for (size_t i = 0; i < spline.bsn; i++)
    {
        collided = colRobot6DBL(spline.x.col(i).colmem);
        endT = (i == spline.bsn - 1) && !collided;
        dis2goal = spline.distance(goalp.mem, spline.x.col(i).colmem);
        reached = dis2goal < 0.1;
        valid_idx = i;
        if (endT or collided or reached)
        {
            if (i > 0)
            {
                lastPos = spline.x.col(i - 1);
                // lastVel = spline.dx.col(i - 1);
                memcpy(lastVel.memptr(), spline.derive(i - 1), dim * sizeof(double));
                pos = spline.x.col(i);
                // vel = spline.dx.col(i);
                memcpy(vel.memptr(), spline.derive(i), dim * sizeof(double));
                // ur5e.setJoints(spline.x.col(i).colmem); // done in colRobot6DBL
                ur5e.calEndpoint();
                memcpy(endPos.memptr(), ur5e.endpoint.data(), 3 * sizeof(double));
                endDis2goal = arma::norm(endPos - endGoalp);
            }
            break;
        }
    }
}

Simulation::Simulation(const int seed, const int dim,
                       const int maxSensoredObs, const double maxSensorDis,
                       const char *asset_path)
    : Scene(seed, dim),
      maxSensoredObs(maxSensoredObs), maxSensorDis(maxSensorDis)
{
    // this->maxSensoredObs = maxSensoredObs;
    ur5e.loadSTL((std::string(asset_path) + "ur5e/collision/").c_str());
    state = new double[dim * 2];
    pos = arma::mat(state, dim, 1, false, true);
    vel = arma::mat(state + dim, dim, 1, false, true);
    lastState = new double[dim * 2];
    lastPos = arma::mat(lastState, dim, 1, false, true);
    lastVel = arma::mat(lastState + dim, dim, 1, false, true);
    traj_dense = dim != 6 ? 25.0 : 180.0 / M_PI / 6;
    endPos = arma::mat(3, 1);
    endStartp = arma::mat(3, 1);
    endGoalp = arma::mat(3, 1);

    startp = arma::mat(dim, 1);
    goalp = arma::mat(dim, 1);
    goaldir = arma::mat(dim, 1);
    mask = new double[maxSensoredObs * 2];
    mask1 = mask;
    mask2 = mask + maxSensoredObs;
    mask1Mat = arma::mat(mask1, maxSensoredObs, 1, false, true);
    mask2Mat = arma::mat(mask2, maxSensoredObs, 1, false, true);

    if (dim == 2)
    {
        outState = arma::mat(5, 1);
        // state[5]: goaldir[2], goaldis[1], vel[2]
        // circle[4]: cdir[2] cdis[1] radius[1]
        // rectangle[7]: cdir[2] cdis[1] axisx[2] axisy[2]
        obsbuf1 = new double[(4 + 5) * maxSensoredObs];
        obsbuf2 = new double[(7 + 5) * maxSensoredObs];
        obsbuf1Mat = arma::mat(obsbuf1, 4 + 5, maxSensoredObs, false, true);
        obsbuf2Mat = arma::mat(obsbuf2, 7 + 5, maxSensoredObs, false, true);
    }
    else if (dim == 3)
    {
        outState = arma::mat(7, 1);
        // state[7]: goaldir[3], goaldis[1], vel[3]
        // sphere[5]: cdir[3] cdis[1] radius[1]
        // box[13]: cdir[3] cdis[1] axisx[3] axisy[3] axisz[3]
        obsbuf1 = new double[(5 + 7) * maxSensoredObs];
        obsbuf2 = new double[(13 + 7) * maxSensoredObs];
        obsbuf1Mat = arma::mat(obsbuf1, 5 + 7, maxSensoredObs, false, true);
        obsbuf2Mat = arma::mat(obsbuf2, 13 + 7, maxSensoredObs, false, true);
    }
    else
    {
        outState = arma::mat(18, 1);
        // state[18]: joints[6], goal[6], diff[6]
        // sphere[5]: cdir[3] cdis[1] radius[1]
        // box[13]: cdir[3] cdis[1] axisx[3] axisy[3] axisz[3]
        obsbuf1 = new double[(5 + 18) * maxSensoredObs];
        obsbuf2 = new double[(13 + 18) * maxSensoredObs];
        obsbuf1Mat = arma::mat(obsbuf1, 5 + 18, maxSensoredObs, false, true);
        obsbuf2Mat = arma::mat(obsbuf2, 13 + 18, maxSensoredObs, false, true);
    }
    obsbuf1Mat.fill(0);
    obsbuf2Mat.fill(0);

    sensorCB.request.enable_contact = 0;
    sensorCB.request.num_max_contacts = 100000;
}

Simulation::~Simulation()
{
    delete[] state;
    delete[] lastState;
    delete[] obsbuf1;
    delete[] obsbuf2;
    delete[] obs1;
    delete[] obs2;
    delete[] mask;
}

void Simulation::clearScene()
{
    circles.clear();
    rectangles.clear();
    spheres.clear();
    boxes.clear();
    quats.clear();
    rotMs.clear();
    geomLoc.clear();
    colM.clear();
    colMN.clear();
    colGs.clear();
    for (auto p : colOs)
    {
        delete p;
    }
    colOs.clear();
    hardPose_.clear();
    hardPose.clear();
    obs_map.clear();
}

void Simulation::generate2D(const double min_obs_sep)
{
    clearScene();
    Scene::addOBS2d(min_obs_sep);
    generate2D_();
}

void Simulation::generate2D_()
{
    for (size_t i = 0; i < colGs.size(); i++)
    {
        obs_map.insert({colGs[i].get(), i});
    }
    delete[] obs1;
    delete[] obs2;
    obs1 = new double[4 * circles.size()];
    obs2 = new double[7 * rectangles.size()];
    obs1Mat = arma::mat(obs1, 4, circles.size(), false, false);
    comp1 = Comp(&obs1Mat, dim);
    obs2Mat = arma::mat(obs2, 7, rectangles.size(), false, false);
    comp2 = Comp(&obs2Mat, dim);
    for (auto loc : geomLoc)
    {
        if (loc.first)
        {
            obs1Mat.col(loc.second) = {circles[loc.second][0],
                                       circles[loc.second][1],
                                       0,
                                       circles[loc.second][2]};
        }
        else
        {
            obs2Mat.col(loc.second) = {rectangles[loc.second][0],
                                       rectangles[loc.second][1],
                                       0,
                                       rectangles[loc.second][3] * std::cos(rectangles[loc.second][5]),
                                       rectangles[loc.second][3] * std::sin(rectangles[loc.second][5]),
                                       -rectangles[loc.second][4] * std::sin(rectangles[loc.second][5]),
                                       rectangles[loc.second][4] * std::cos(rectangles[loc.second][5])};
        }
    }
}

void Simulation::generate3D(const double min_obs_sep)
{
    clearScene();
    Scene::addOBS3d(min_obs_sep);
    generate3D_();
}

void Simulation::generate3D_()
{
    for (size_t i = 0; i < colGs.size(); i++)
    {
        obs_map.insert({colGs[i].get(), i});
    }
    delete[] obs1;
    delete[] obs2;
    obs1 = new double[5 * spheres.size()];
    obs2 = new double[13 * boxes.size()];
    obs1Mat = arma::mat(obs1, 5, spheres.size(), false, false);
    comp1 = Comp(&obs1Mat, dim);
    obs2Mat = arma::mat(obs2, 13, boxes.size(), false, false);
    comp2 = Comp(&obs2Mat, dim);
    for (auto loc : geomLoc)
    {
        if (loc.first)
        {
            obs1Mat.col(loc.second) = {spheres[loc.second][0],
                                       spheres[loc.second][1],
                                       spheres[loc.second][2],
                                       0,
                                       spheres[loc.second][3]};
        }
        else
        {
            obs2Mat.col(loc.second).head(4) = {
                boxes[loc.second][0],
                boxes[loc.second][1],
                boxes[loc.second][2],
                0,
            };
            memcpy(obs2Mat.colptr(loc.second) + 4, (quats[loc.second] * Eigen::Vector3d::UnitX()).data(), 3 * sizeof(double));
            memcpy(obs2Mat.colptr(loc.second) + 7, (quats[loc.second] * Eigen::Vector3d::UnitY()).data(), 3 * sizeof(double));
            memcpy(obs2Mat.colptr(loc.second) + 10, (quats[loc.second] * Eigen::Vector3d::UnitZ()).data(), 3 * sizeof(double));
        }
    }
}

void Simulation::generate6D(const int panelCap, const int obsCap, const double min_oh, const double max_oh, const double deep)
{

    bool pvalid = false;
    double angleRow, anglePitch;
    Eigen::Quaterniond quat;
    Eigen::Vector3d center;
    double panel_depth = randu(random_gen) > 0.5 ? 0.2 : 0.02;
    // int loop = 0;
    while (!pvalid)
    {
        clearScene();
        // loop++;
        // if (loop > 1e3)
        // {
        //     std::cout << tightPose << std::endl;
        //     throw std::runtime_error("dead loop!");
        // }
        tightPose = (arma::randu(dim, 1) - 0.5) * 2 * M_PI;
        tightPose[1] = randu(random_gen) * M_PI / 6 - M_PI / 3;
        tightPose[2] = randu(random_gen) * M_PI / 4 + M_PI / 4;
        // tightPose[1] = -M_PI / 6;
        // tightPose[2] = M_PI / 2;
        // tightPose[1] = -M_PI / 3;
        // tightPose[2] = M_PI / 4;
        if (randu(random_gen) > 0.5)
        {
            tightPose[1] -= M_PI / 2;
        }
        if (selfCollideBL(tightPose.mem))
        {
            continue;
        }
        ur5e.setJoints(tightPose.mem);
        ur5e.setLinksPoseB();
        ur5e.setLinksPoseL();
        // two angle noise
        angleRow = (randu(random_gen) * 2 - 1) * M_PI / 12;   // X axis
        anglePitch = (randu(random_gen) * 2 - 1) * M_PI / 12; // Y axis
        // quat = rotation of 3rd coordinate with noise
        quat = ur5e.quaternions[2] * Eigen::AngleAxisd(angleRow, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(anglePitch, Eigen::Vector3d::UnitY());
        // offset in 3rd coordinate, approximately the end of forearm
        center = {0, 0.05, 0.4 - deep};
        // center of the panel in world coordinate
        center = ur5e.quaternions[2] * center + ur5e.translations[2];
        pvalid = Scene::addPanel(quat, center, panelCap, min_oh, max_oh, panel_depth);
        pvalid = pvalid && Scene::addOBS6d(obsCap);
    }
    std::copy_if(hardPose_.begin(), hardPose_.end(), std::back_inserter(hardPose),
                 [this](const arma::mat &pose)
                 { return !colRobot6DBL(pose.mem); });
    generate6D_();
}

void Simulation::generate6D_()
{
    for (size_t i = 0; i < colGs.size(); i++)
    {
        obs_map.insert({colGs[i].get(), i});
    }
    delete[] obs1;
    delete[] obs2;
    obs1 = new double[5 * spheres.size()];
    obs2 = new double[13 * boxes.size()];
    obs1Mat = arma::mat(obs1, 5, spheres.size(), false, false);
    obs2Mat = arma::mat(obs2, 13, boxes.size(), false, false);
    mask1Mat.fill(-INFINITY);
    mask2Mat.fill(-INFINITY);
    mask1Mat.head_rows(spheres.size()).fill(0);
    mask2Mat.head_rows(boxes.size()).fill(0);
    for (auto loc : geomLoc)
    {
        if (loc.first)
        {
            obs1Mat.col(loc.second) = {spheres[loc.second][0],
                                       spheres[loc.second][1],
                                       spheres[loc.second][2],
                                       0,
                                       spheres[loc.second][3]};
        }
        else
        {
            obs2Mat.col(loc.second).head(4) = {
                boxes[loc.second][0],
                boxes[loc.second][1],
                boxes[loc.second][2],
                0,
            };
            memcpy(obs2Mat.colptr(loc.second) + 4, (quats[loc.second] * Eigen::Vector3d::UnitX()).data(), 3 * sizeof(double));
            memcpy(obs2Mat.colptr(loc.second) + 7, (quats[loc.second] * Eigen::Vector3d::UnitY()).data(), 3 * sizeof(double));
            memcpy(obs2Mat.colptr(loc.second) + 10, (quats[loc.second] * Eigen::Vector3d::UnitZ()).data(), 3 * sizeof(double));
        }
    }
    obsbuf1Mat.submat(0, 0, 4, spheres.size() - 1) = obs1Mat;
    obsbuf2Mat.submat(0, 0, 12, boxes.size() - 1) = obs2Mat;
    // compute distance and normalise
    obsbuf1Mat.submat(3, 0, 3, spheres.size() - 1) = arma::vecnorm(obsbuf1Mat.submat(0, 0, 2, spheres.size() - 1));
    obsbuf1Mat.submat(0, 0, 2, spheres.size() - 1) = arma::normalise(obsbuf1Mat.submat(0, 0, 2, spheres.size() - 1));
    obsbuf2Mat.submat(3, 0, 3, boxes.size() - 1) = arma::vecnorm(obsbuf2Mat.submat(0, 0, 2, boxes.size() - 1));
    obsbuf2Mat.submat(0, 0, 2, boxes.size() - 1) = arma::normalise(obsbuf2Mat.submat(0, 0, 2, boxes.size() - 1));
}

void Simulation::genOBS2D(arma::mat &goalp_)
{
    // compute out state
    outState.rows(0, 1) = goalp_ - pos;
    outState[2] = arma::norm(outState.rows(0, 1));
    outState.rows(0, 1) = outState.rows(0, 1) / outState[2];
    outState[2] /= 10;
    outState.rows(3, 4) = vel / arma::norm(vel);
    // gather obs in sensor range
    sensorCB.result.clear();
    sensor.setTranslation({pos[0], pos[1], 0});
    sensor.computeAABB();
    colM.collide(&sensor, &sensorCB, fcl::DefaultCollisionFunction);
    obs1idx.clear();
    obs2idx.clear();
    int idx;
    for (size_t i = 0; i < sensorCB.result.numContacts(); i++)
    {
        idx = obs_map[sensorCB.result.getContact(i).o1];
        if (geomLoc[idx].first)
        {
            obs1idx.push_back(geomLoc[idx].second);
        }
        else
        {
            obs2idx.push_back(geomLoc[idx].second);
        }
    }
    // std::cout << obs1idx.size() << ", " << obs2idx.size() << std::endl;
    if (obs1idx.size() > 0)
    {
        // compute distance to center
        for (auto i : obs1idx)
        {
            obs1Mat.col(i)[2] = arma::norm(obs1Mat.col(i).head(2) - pos);
        }
        // sort and resize
        if (obs1idx.size() >= maxSensoredObs)
        {
            std::sort(obs1idx.begin(), obs1idx.end(), comp1);
            obs1idx.resize(maxSensoredObs);
        }
        // fill mask
        mask1Mat.tail_rows(maxSensoredObs - obs1idx.size()).fill(-INFINITY);
        mask1Mat.head_rows(obs1idx.size()).fill(0);
        // copy to buffer
        for (size_t i = 0; i < obs1idx.size(); i++)
        {
            obsbuf1Mat.col(i).head(4) = obs1Mat.col(obs1idx[i]);
        }
        // compute relative center pos
        obsbuf1Mat.submat(0, 0, 1, obs1idx.size() - 1).each_col() -= pos;
        // normalise
        obsbuf1Mat.submat(0, 0, 1, obs1idx.size() - 1) = arma::normalise(obsbuf1Mat.submat(0, 0, 1, obs1idx.size() - 1));
        // fill outstate
        obsbuf1Mat.submat(4, 0, 8, obs1idx.size() - 1).each_col() = outState;
    }
    else
    {
        // at least one default obs
        mask1Mat.fill(-INFINITY);
        mask1Mat[0] = 0;
        obsbuf1Mat.col(0).head(2) = -arma::normalise(outState.head_rows(2)); // goaldir
        obsbuf1Mat.col(0)[2] = maxSensorDis * 2;                             // out of sensor range
        obsbuf1Mat.col(0)[3] = 0;                                            // radius = 0
        obsbuf1Mat.col(0).tail(5) = outState;
    }
    if (obs2idx.size() > 0)
    {
        // compute distance to center
        for (auto i : obs2idx)
        {
            obs2Mat.col(i)[2] = arma::norm(obs2Mat.col(i).head(2) - pos);
        }
        // sort and resize
        if (obs2idx.size() >= maxSensoredObs)
        {
            std::sort(obs2idx.begin(), obs2idx.end(), comp2);
            obs2idx.resize(maxSensoredObs);
        }
        // fill mask
        mask2Mat.tail_rows(maxSensoredObs - obs2idx.size()).fill(-INFINITY);
        mask2Mat.head_rows(obs2idx.size()).fill(0);
        // copy to buffer
        for (size_t i = 0; i < obs2idx.size(); i++)
        {
            obsbuf2Mat.col(i).head(7) = obs2Mat.col(obs2idx[i]);
        }
        // compute relative center pos
        obsbuf2Mat.submat(0, 0, 1, obs2idx.size() - 1).each_col() -= pos;
        // normalise
        obsbuf2Mat.submat(0, 0, 1, obs2idx.size() - 1) = arma::normalise(obsbuf2Mat.submat(0, 0, 1, obs2idx.size() - 1));
        // fill outstate
        obsbuf2Mat.submat(7, 0, 11, obs2idx.size() - 1).each_col() = outState;
    }
    else
    {
        // at least one default obs
        mask2Mat.fill(-INFINITY);
        mask2Mat[0] = 0;
        obsbuf2Mat.col(0).fill(0);
        obsbuf2Mat.col(0).head(2) = -arma::normalise(outState.head_rows(2)); // goaldir
        obsbuf2Mat.col(0)[2] = maxSensorDis * 2;                             // out of sensor range
        // obsbuf2Mat.col(0)[3,4,5,6] = 0;                // halfx = 0, halfy = 0
        obsbuf2Mat.col(0).tail(5) = outState;
    }
}
void Simulation::genOBS2D()
{
    genOBS2D(goalp);
}
void Simulation::genOBS3D(arma::mat &goalp_)
{
    // compute out state
    outState.head_rows(3) = goalp_ - pos;
    outState[3] = arma::norm(outState.head_rows(3));
    outState.head_rows(3) = outState.head_rows(3) / outState[3];
    outState[3] /= 10;
    outState.tail_rows(3) = vel / arma::norm(vel);
    // gather obs in sensor range
    sensorCB.result.clear();
    sensor.setTranslation({pos[0], pos[1], pos[2]});
    sensor.computeAABB();
    colM.collide(&sensor, &sensorCB, fcl::DefaultCollisionFunction);
    obs1idx.clear();
    obs2idx.clear();
    int idx;
    for (size_t i = 0; i < sensorCB.result.numContacts(); i++)
    {
        idx = obs_map[sensorCB.result.getContact(i).o1];
        if (geomLoc[idx].first)
        {
            obs1idx.push_back(geomLoc[idx].second);
        }
        else
        {
            obs2idx.push_back(geomLoc[idx].second);
        }
    }
    // std::cout << obs1idx.size() << ", " << obs2idx.size() << std::endl;
    if (obs1idx.size() > 0)
    {
        // compute distance to center
        for (auto i : obs1idx)
        {
            obs1Mat.col(i)[3] = arma::norm(obs1Mat.col(i).head(3) - pos);
        }
        // sort and resize
        if (obs1idx.size() > maxSensoredObs)
        {
            std::sort(obs1idx.begin(), obs1idx.end(), comp1);
            obs1idx.resize(maxSensoredObs);
        }
        // fill mask
        mask1Mat.tail_rows(maxSensoredObs - obs1idx.size()).fill(-INFINITY);
        mask1Mat.head_rows(obs1idx.size()).fill(0);
        // copy to buffer
        for (size_t i = 0; i < obs1idx.size(); i++)
        {
            obsbuf1Mat.col(i).head(5) = obs1Mat.col(obs1idx[i]);
        }
        // compute relative center pos
        obsbuf1Mat.submat(0, 0, 2, obs1idx.size() - 1).each_col() -= pos;
        // normalise
        obsbuf1Mat.submat(0, 0, 2, obs1idx.size() - 1) = arma::normalise(obsbuf1Mat.submat(0, 0, 2, obs1idx.size() - 1));
        // fill outstate
        obsbuf1Mat.submat(5, 0, 11, obs1idx.size() - 1).each_col() = outState;
    }
    else
    {
        // at least one default obs
        mask1Mat.fill(-INFINITY);
        mask1Mat[0] = 0;
        obsbuf1Mat.col(0).head(3) = -arma::normalise(outState.head_rows(3)); // goaldir
        obsbuf1Mat.col(0)[3] = maxSensorDis * 2;                             // out of sensor range
        obsbuf1Mat.col(0)[4] = 0;                                            // radius = 0
        obsbuf1Mat.col(0).tail(7) = outState;
    }
    if (obs2idx.size() > 0)
    {
        // compute distance to center
        for (auto i : obs2idx)
        {
            obs2Mat.col(i)[3] = arma::norm(obs2Mat.col(i).head(3) - pos);
        }
        // sort and resize
        if (obs2idx.size() > maxSensoredObs)
        {
            std::sort(obs2idx.begin(), obs2idx.end(), comp2);
            obs2idx.resize(maxSensoredObs);
        }
        // fill mask
        mask2Mat.tail_rows(maxSensoredObs - obs2idx.size()).fill(-INFINITY);
        mask2Mat.head_rows(obs2idx.size()).fill(0);
        // copy to buffer
        for (size_t i = 0; i < obs2idx.size(); i++)
        {
            obsbuf2Mat.col(i).head(13) = obs2Mat.col(obs2idx[i]);
        }
        // compute relative center pos
        obsbuf2Mat.submat(0, 0, 2, obs2idx.size() - 1).each_col() -= pos;
        // normalise
        obsbuf2Mat.submat(0, 0, 2, obs2idx.size() - 1) = arma::normalise(obsbuf2Mat.submat(0, 0, 2, obs2idx.size() - 1));
        // fill outstate
        obsbuf2Mat.submat(13, 0, 19, obs2idx.size() - 1).each_col() = outState;
    }
    else
    {
        // at least one default obs
        mask2Mat.fill(-INFINITY);
        mask2Mat[0] = 0;
        obsbuf2Mat.col(0).fill(0);
        obsbuf2Mat.col(0).head(3) = -outState.head_rows(3); // goaldir
        obsbuf2Mat.col(0)[3] = maxSensorDis * 2;            // out of sensor range
        // obsbuf2Mat.col(0)[4,5,6,7,8,9,10,11,12] = 0;                // halfx = 0, halfy = 0
        obsbuf2Mat.col(0).tail(7) = outState;
    }
}
void Simulation::genOBS3D()
{
    genOBS3D(goalp);
}
void Simulation::genOBS6D(arma::mat &goalp_)
{
    // compute out state
    outState.head_rows(6) = pos;
    outState.rows(6, 11) = goalp_;
    outState.tail_rows(6) = vel / arma::norm(vel);
    // fill outstate
    obsbuf1Mat.tail_rows(18).cols(0, spheres.size() - 1).each_col() = outState;
    obsbuf2Mat.tail_rows(18).cols(0, boxes.size() - 1).each_col() = outState;
}
void Simulation::genOBS6D()
{
    genOBS6D(goalp);
}

std::string Simulation::toYAML()
{
    YAML::Node node;
    node["startp"] = startp;
    node["goalp"] = goalp;
    node["circles"] = circles;
    node["rectangles"] = rectangles;
    node["spheres"] = spheres;
    node["boxes"] = boxes;
    YAML::Emitter emitter;
    emitter << node;
    return emitter.c_str(); // copyed as a string, since emitter goes out of scope
}

void Simulation::fromYAML(const char *yaml)
{
    clearScene();
    auto node = YAML::Load(yaml);
    startp = node["startp"].as<arma::mat>();
    goalp = node["goalp"].as<arma::mat>();
    circles = node["circles"].as<std::vector<std::array<double, 3UL>>>();
    rectangles = node["rectangles"].as<std::vector<std::array<double, 6UL>>>();
    spheres = node["spheres"].as<std::vector<std::array<double, 4UL>>>();
    boxes = node["boxes"].as<std::vector<std::array<double, 10UL>>>();
    Eigen::Quaterniond quat;
    Eigen::Matrix3d rotM;
    if (dim == 2)
    {
        for (auto p : circles)
        {
            colGs.push_back(std::make_shared<fcl::Sphered>(p[2]));
            colOs.push_back(new fcl::CollisionObjectd(colGs.back(), Eigen::Matrix3d(), Eigen::Vector3d(p[0], p[1], 0)));
            geomLoc.push_back(std::pair<bool, int>({true, colGs.size() - 1}));
        }
        for (auto p : rectangles)
        {
            colGs.push_back(std::make_shared<fcl::Boxd>(2 * p[3], 2 * p[4], 1));
            quat = Eigen::Quaterniond(Eigen::AngleAxisd(p[5], Eigen::Vector3d::UnitZ()));
            rotM = quat.toRotationMatrix();
            quats.push_back(quat);
            rotMs.push_back(rotM);
            colOs.push_back(new fcl::CollisionObjectd(colGs.back(), rotM, Eigen::Vector3d(p[0], p[1], 0)));
            geomLoc.push_back(std::pair<bool, int>({false, colGs.size() - circles.size() - 1}));
        }
    }
    else
    {
        for (auto p : spheres)
        {
            colGs.push_back(std::make_shared<fcl::Sphered>(p[3]));
            colOs.push_back(new fcl::CollisionObjectd(colGs.back(), Eigen::Matrix3d(), Eigen::Vector3d(p[0], p[1], p[2])));
            geomLoc.push_back(std::pair<bool, int>({true, colGs.size() - 1}));
        }
        for (auto p : boxes)
        {
            colGs.push_back(std::make_shared<fcl::Boxd>(2 * p[4], 2 * p[5], 2 * p[6]));
            quat = rpyToQuat(p[7], p[8], p[9]);
            rotM = quat.toRotationMatrix();
            quats.push_back(quat);
            rotMs.push_back(rotM);
            colOs.push_back(new fcl::CollisionObjectd(colGs.back(), rotM, Eigen::Vector3d(p[0], p[1], p[2])));
            geomLoc.push_back(std::pair<bool, int>({false, colGs.size() - spheres.size() - 1}));
        }
    }
    for (auto o : colOs)
    {
        o->computeAABB();
        colM.registerObject(o);
        if (dim == 6)
        {
            colMN.registerObject(o);
        }
    }
    colM.setup();
    if (dim == 6)
    {
        colMN.setup();
    }
    if (dim == 2)
    {
        generate2D_();
    }
    else if (dim == 3)
    {
        generate3D_();
    }
    else
    {
        generate6D_();
    }
}
