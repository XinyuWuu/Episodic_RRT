#ifndef __SIMULATION__
#define __SIMULATION__

#include "scene.hpp"
#include "spline.hpp"
#include "utils.hpp"
#include <tsl/robin_map.h>
#include <string>

struct Comp
{
    arma::mat *mat;
    int dim;
    bool operator()(int a, int b)
    {
        return mat->col(a)[dim] < mat->col(b)[dim];
    }
};

class Simulation : public Scene
{
private:
public:
    Simulation(const int seed, const int dim, const int maxSensoredObs, const double maxSensorDis, const char *asset_path);
    ~Simulation();

    double *state = nullptr;
    arma::mat pos, vel; // share memory with state
    arma::mat endPos;
    double *lastState = nullptr;
    arma::mat lastPos, lastVel; // share memory with lastState
    arma::mat startp, goalp;
    arma::mat endStartp, endGoalp;
    arma::mat goaldir;
    double dis2goal, endDis2goal;
    bool collided, endT, reached;
    int valid_idx;
    arma::mat traj1, traj2;
    double traj_dense;
    Spline spline = Spline();

    void clearScene();
    void generate2D(const double min_obs_sep = 0.5);
    void generate2D_();
    void generate3D(const double min_obs_sep = 0.5);
    void generate3D_();
    void generate6D(const int panelCap = 4, const int obsCap = 4, const double min_oh = 0.1, const double max_oh = 0.4, const double deep = 0.15);
    void generate6D_();
    const double border_margin = 3;
    const double max_radius = env_half_size - border_margin;
    void genStartgoal2D(const double start_radius, const double safe_margin, const int mode = 0);
    void genStartgoal3D(const double start_radius, const double safe_margin, const int mode = 0);
    bool invertsg = false;
    bool genStartgoal6D(const int mode = 0, bool invert = false);

    void setTraj2D(const double *traj, const int num, const double *first = nullptr, const double *last = nullptr);
    void step2D(const double *traj, const int num);
    void setTraj3D(const double *traj, const int num, const double *first = nullptr, const double *last = nullptr);
    void step3D(const double *traj, const int num);
    void setTraj6D(const double *traj, const int num, const double *first = nullptr);
    void step6D(const double *traj, const int num);

    const int maxSensoredObs = 10;
    double *mask = nullptr, *mask1 = nullptr, *mask2 = nullptr;
    arma::mat mask1Mat, mask2Mat; // share memory with mask1, mask2
    double *obsbuf1 = nullptr, *obsbuf2 = nullptr;
    arma::mat obsbuf1Mat, obsbuf2Mat; // share memory with obsbuf1, obsbuf2
    double *obs1 = nullptr, *obs2 = nullptr;
    arma::mat obs1Mat, obs2Mat; // share memory with obs1, obs2
    const double maxSensorDis = 5;
    std::shared_ptr<fcl::Sphered> sensor_geom = std::make_shared<fcl::Sphered>(maxSensorDis);
    fcl::CollisionObjectd sensor = fcl::CollisionObjectd(sensor_geom);
    fcl::DefaultCollisionData<double> sensorCB = fcl::DefaultCollisionData<double>();
    tsl::robin_map<const fcl::CollisionGeometryd *, int> obs_map;
    std::vector<int> obs1idx, obs2idx;
    arma::mat outState;
    Comp comp1, comp2;
    void genOBS2D(arma::mat &goalp);
    void genOBS3D(arma::mat &goalp);
    void genOBS6D(arma::mat &goalp);
    void genOBS2D();
    void genOBS3D();
    void genOBS6D();
    std::string toYAML();
    void fromYAML(const char *yaml);
};

#endif
