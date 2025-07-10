#include "logger.hpp"

Logger::Logger(std::string_view recording_id, std::string_view app_id)
{
    rec = new rerun::RecordingStream(app_id, recording_id);
    rec->set_time_duration_secs("time", 0);
}

Logger::~Logger()
{
    delete rec;
}

void Logger::logGround(double env_half_size)
{
    rec->log_static("ground", rerun::Boxes3D::from_half_sizes({{float(env_half_size), float(env_half_size), 0.001}})
                                  .with_centers({{0, 0, -0.001}})
                                  .with_colors(rerun::Color(150, 150, 150))
                                  .with_fill_mode(rerun::FillMode::Solid));
}

void Logger::logRobot(double robot_r)
{
    rec->log_static("robot/shell", rerun::Ellipsoids3D::from_radii({float(robot_r)})
                                       .with_fill_mode(rerun::FillMode::Solid));
    rec->log_static("robot/head", rerun::Arrows3D::from_vectors({{float(robot_r + 0.1), 0, 0}})
                                      .with_origins({{0, 0, 0.01}})
                                      .with_radii(0.05)
                                      .with_colors(rerun::Color(255, 0, 0)));
    // rec->log_static("robot/cam", rerun::Pinhole::from_focal_length_and_resolution(200, rerun::Vec2D(800, 800))
    //                                  .with_camera_xyz(rerun::components::ViewCoordinates::FRU)
    //                                  .with_image_plane_distance(1.0f));
}

void Logger::logUR5E(const char *path, const char *prefix, uint32_t rgba)
{
    for (size_t i = 0; i < 7; i++)
    {
        rec->log_static(prefix + std::string("UR5E/") + UR5E_parts_name[i], rerun::Asset3D::from_file_path(path + UR5E_parts_name[i] + ".stl").value.with_albedo_factor(rgba));
    }
}

void Logger::logGoal(const double *goal, int dim)
{
    if (dim == 2)
    {
        rec->log_static("goal", rerun::Capsules3D::from_lengths_and_radii({9.0f}, {0.03f})
                                    .with_translations({{float(goal[0]), float(goal[1]), 0.0f}}));
    }
    else
    {
        rec->log_static("goal", rerun::Ellipsoids3D::from_radii({1.0f})
                                    .with_centers({{float(goal[0]), float(goal[1]), float(goal[2])}}));
    }
}

void Logger::save(std::string_view path)
{
    auto error = rec->save(path);
    if (!error.is_ok())
    {
        std::cout << error.description << std::endl;
    }
}

void Logger::logOBS(std::vector<std::array<double, 3>> circles, std::vector<std::array<double, 6>> rectangles)
{
    int i = 0;
    for (auto o : circles)
    {
        rec->log_static("obs/" + std::to_string(i),
                        rerun::Capsules3D::from_lengths_and_radii({float(1.0 + o.at(2))}, {float(o.at(2))})
                            .with_translations({{float(o.at(0)), float(o.at(1)), 0.0f}}));
        i++;
    }
    for (auto o : rectangles)
    {
        rec->log_static("obs/" + std::to_string(i),
                        rerun::Boxes3D::from_half_sizes({{float(o.at(3)), float(o.at(4)), float(1.0 + o.at(2))}})
                            .with_rotation_axis_angles({rerun::RotationAxisAngle({0.0f, 0.0f, 1.0f}, rerun::Angle::radians(o.at(5)))})
                            .with_centers({{float(o.at(0)), float(o.at(1)), 1.0f}})
                            .with_fill_mode(rerun::FillMode::Solid));
        i++;
    }
}

void Logger::logOBS(std::vector<std::array<double, 4>> spheres, std::vector<std::array<double, 10>> boxes, std::vector<Eigen::Quaterniond> quats)
{
    int j = 0;
    for (auto o : spheres)
    {
        rec->log_static("obs/" + std::to_string(j),
                        rerun::Ellipsoids3D::from_centers_and_radii({{float(o.at(0)), float(o.at(1)), float(o.at(2))}}, {float(o.at(3))})
                            .with_fill_mode(rerun::FillMode::Solid));
        j++;
    }
    for (size_t i = 0; i < boxes.size(); i++)
    {
        rec->log_static("obs/" + std::to_string(j),
                        rerun::Boxes3D::from_half_sizes({{float(boxes[i].at(4)), float(boxes[i].at(5)), float(boxes[i].at(6))}})
                            .with_quaternions({rerun::Quaternion().from_wxyz(quats[i].w(), quats[i].x(), quats[i].y(), quats[i].z())})
                            .with_centers({{float(boxes[i].at(0)), float(boxes[i].at(1)), float(boxes[i].at(2))}})
                            .with_fill_mode(rerun::FillMode::Solid));
        j++;
    }
}

void Logger::logPos(const double *pos, const double theta, const double time)
{
    rec->set_time_duration_secs("time", time);
    rec->log("robot/", rerun::archetypes::Transform3D(
                           rerun::RotationAxisAngle(
                               {0.0f, 0.0f, 1.0f},
                               rerun::Angle::radians(float(theta))))
                           .with_translation({float(pos[0]), float(pos[1]), 0.0f})
                           .with_axis_length(0));
}

void Logger::logPos(const double *pos, const double *dir, const double time)
{
    rec->set_time_duration_secs("time", time);
    rec->log("robot/", rerun::archetypes::Transform3D(
                           rerun::RotationAxisAngle(
                               {0.0f, 0.0f, 1.0f},
                               rerun::Angle::radians(float(0))))
                           .with_translation({float(pos[0]), float(pos[1]), float(pos[2])})
                           .with_axis_length(0));
}

void Logger::logPos(Eigen::Quaterniond *quats, Eigen::Vector3d *trans, double time, const char *prefix)
{
    rec->set_time_duration_secs("time", time);
    for (size_t i = 0; i < 6; i++)
    {
        rec->log(prefix + std::string("UR5E/") + UR5E_parts_name[i + 1],
                 rerun::Transform3D::from_translation_rotation(
                     rerun::Vec3D(trans[i].x(), trans[i].y(), trans[i].z()),
                     rerun::Quaternion::from_wxyz(quats[i].w(), quats[i].x(), quats[i].y(), quats[i].z()))
                     .with_axis_length(0.125));
    }
}
