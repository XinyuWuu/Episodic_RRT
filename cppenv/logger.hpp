#ifndef __LOGGER__
#define __LOGGER__
#include <rerun.hpp>
#include <armadillo>
#include <Eigen/Dense>
#include <string>
class Logger
{
private:
    std::string UR5E_parts_name[7] = {
        "base",
        "shoulder",
        "upperarm",
        "forearm",
        "wrist1",
        "wrist2",
        "wrist3"};
    rerun::Ellipsoids3D robot_shell = rerun::Ellipsoids3D::from_radii({0.2}).with_fill_mode(rerun::FillMode::Solid);
    rerun::Arrows3D robot_head = rerun::Arrows3D::from_vectors({{0.3, 0, 0}}).with_origins({{0, 0, 0.01}}).with_radii(0.05).with_colors(rerun::Color(255, 0, 0));
    rerun::Boxes3D ground = rerun::Boxes3D::from_half_sizes({{30, 30, 0.001}}).with_centers({{0, 0, -0.001}}).with_colors(rerun::Color(150, 150, 150)).with_fill_mode(rerun::FillMode::Solid);

public:
    rerun::RecordingStream *rec;
    Logger(std::string_view recording_id, std::string_view app_id = "SIM");
    ~Logger();
    void logGround(double env_half_size);
    void logRobot(double robot_r);
    void logUR5E(const char *path, const char *prefix = "", uint32_t rgba = 0xffffffff);
    void logGoal(const double *goal, int dim = 2);
    void logOBS(std::vector<std::array<double, 3>> circles, std::vector<std::array<double, 6>> rectangles);
    void logOBS(std::vector<std::array<double, 4>> spheres, std::vector<std::array<double, 10>> boxes, std::vector<Eigen::Quaterniond> quats);
    void logPos(const double *pos, const double theta, const double time);
    void logPos(const double *pos, const double *dir, const double time);
    void logPos(Eigen::Quaterniond *quats, Eigen::Vector3d *trans, double time, const char *prefix = "");
    void save(std::string_view path);
};

#endif
