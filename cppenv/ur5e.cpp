#include "ur5e.hpp"
#include <iostream>
// https://github.com/dartsim/dart/blob/d718a3efb1a72ad98403eb4c267497d24df17e14/dart/collision/fcl/FCLCollisionDetector.cpp#L532
template <class BV>
std::shared_ptr<fcl::BVHModel<BV>> createMesh(double _scaleX, double _scaleY, double _scaleZ, const aiScene *_mesh)
{
    // Create FCL mesh from Assimp mesh
    assert(_mesh);
    std::shared_ptr<fcl::BVHModel<BV>> model = std::make_shared<fcl::BVHModel<BV>>();
    model->beginModel();
    for (std::size_t i = 0; i < _mesh->mNumMeshes; i++)
    {
        for (std::size_t j = 0; j < _mesh->mMeshes[i]->mNumFaces; j++)
        {
            fcl::Vector3d vertices[3];
            for (std::size_t k = 0; k < 3; k++)
            {
                const aiVector3D &vertex = _mesh->mMeshes[i]
                                               ->mVertices[_mesh->mMeshes[i]->mFaces[j].mIndices[k]];
                vertices[k] = fcl::Vector3d(
                    vertex.x * _scaleX, vertex.y * _scaleY, vertex.z * _scaleZ);
            }
            model->addTriangle(vertices[0], vertices[1], vertices[2]);
        }
    }
    model->endModel();
    return model;
}

void UR5E::setJoints(const double *joints_)
{
    memcpy(joints, joints_, sizeof(double) * 6);
    joints[1] += M_PI / 2;
    joints[3] += M_PI / 2;
    quaternions[0] = Eigen::Quaterniond(Eigen::AngleAxisd(joints[0], axises[0]));
    //  translations[0] = translations_[0]; const value
    for (size_t i = 1; i < 6; i++)
    {
        quaternions[i] = quaternions[i - 1] * Eigen::Quaterniond(Eigen::AngleAxisd(joints[i], axises[i]));
        translations[i] = quaternions[i - 1] * translations_[i] + translations[i - 1];
    }
}

UR5E::UR5E(/* args */)
{

    translations_[0] = Eigen::Vector3d(0, 0, 0.1625);
    translations_[1] = Eigen::Vector3d(0, 0.138, 0);
    translations_[2] = Eigen::Vector3d(0, 0.007 - 0.138, 0.425);
    translations_[3] = Eigen::Vector3d(0, 0, 0.3922);
    translations_[4] = Eigen::Vector3d(0, 0.134 - 0.007, 0);
    translations_[5] = Eigen::Vector3d(0, 0, 0.1);

    axises[0] = Eigen::Vector3d::UnitZ();
    axises[1] = Eigen::Vector3d::UnitY();
    axises[2] = Eigen::Vector3d::UnitY();
    axises[3] = Eigen::Vector3d::UnitY();
    axises[4] = Eigen::Vector3d::UnitZ();
    axises[5] = Eigen::Vector3d::UnitY();

    translations[0] = translations_[0];

    endpointOffset = {0, 0.1, 0};
    endquat_ = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()));

    // selfColPairs.resize(10);
    // selfColPairs.push_back({0, 1}); <disable_collisions link1="base_link_inertia" link2="shoulder_link" reason="Adjacent" />
    // selfColPairs.push_back({0, 2}); <disable_collisions link1="base_link_inertia" link2="upper_arm_link" reason="Never" />
    selfColPairs.push_back({0, 3});
    // selfColPairs.push_back({0, 4}); <disable_collisions link1="base_link_inertia" link2="wrist_1_link" reason="Never" />
    selfColPairs.push_back({0, 5});
    selfColPairs.push_back({0, 6});
    // selfColPairs.push_back({1, 2}); <disable_collisions link1="shoulder_link" link2="upper_arm_link" reason="Adjacent" />
    selfColPairs.push_back({1, 3});
    selfColPairs.push_back({1, 4});
    selfColPairs.push_back({1, 5});
    selfColPairs.push_back({1, 6});
    // selfColPairs.push_back({2, 3}); <disable_collisions link1="forearm_link" link2="upper_arm_link" reason="Adjacent" />
    // selfColPairs.push_back({2, 4}); <disable_collisions link1="shoulder_link" link2="wrist_1_link" reason="Never" />
    // selfColPairs.push_back({2, 5}); <disable_collisions link1="shoulder_link" link2="wrist_2_link" reason="Never" />
    selfColPairs.push_back({2, 6});
    // selfColPairs.push_back({3, 4}); <disable_collisions link1="forearm_link" link2="wrist_1_link" reason="Adjacent" />
    selfColPairs.push_back({3, 5});
    selfColPairs.push_back({3, 6});
    // selfColPairs.push_back({4, 5}); <disable_collisions link1="wrist_1_link" link2="wrist_2_link" reason="Adjacent" />
    // selfColPairs.push_back({4, 6}); <disable_collisions link1="wrist_1_link" link2="wrist_3_link" reason="Never" />
    // selfColPairs.push_back({5, 6}); <disable_collisions link1="wrist_2_link" link2="wrist_3_link" reason="Adjacent" />

    colReq.enable_contact = 0;
    colReq.num_max_contacts = 100000;
    colCB.request.enable_contact = 0;
    colCB.request.num_max_contacts = 100000;

    sizes[0] = {0.151, 0.151, 0.1};
    offsets[0] = {0, 0, 0.05};
    sizes[1] = {0.121, 0.13, 0.142};
    offsets[1] = {0, 0.005, 0.0};
    sizes[2] = {0.121, 0.144, 0.54};
    offsets[2] = {0, 0.0, 0.214};
    sizes[3] = {0.12, 0.12, 0.489};
    offsets[3] = {0, 0.0, 0.1865};
    sizes[4] = {0.079, 0.112, 0.113};
    offsets[4] = {0, 0.11, 0.0};
    sizes[5] = {0.0787, 0.11, 0.087};
    offsets[5] = {0, -0.002, 0.0955};
    sizes[6] = {0.075, 0.0465, 0.082};
    offsets[6] = {0, 0.076, 0.0025};
    for (size_t i = 0; i < 7; i++)
    {
        colGs[i] = std::make_shared<fcl::Boxd>(sizes[i]);
        linksB[i] = new fcl::CollisionObjectd(colGs[i]);
        linksB[i]->computeAABB();
        colMB.registerObject(linksB[i]);
    }
    colMB.setup();
    linksB[0]->setTranslation(offsets[0]);
    linksB[0]->computeAABB();
    // colMB.update(linksB[0]);
}

UR5E::~UR5E()
{
    for (size_t i = 0; i < 7; i++)
    {
        delete links[i];
        delete linksB[i];
    }
    colML.clear();
    colMB.clear();
}

void UR5E::loadSTL(const char *path)
{
    for (size_t i = 0; i < 7; i++)
    {
        meshes[i] = createMesh<fcl::OBBRSSd>(1.0f, 1.0f, 1.0f, importer.ReadFile(path + UR5E_parts_name[i] + ".stl", aiProcess_Triangulate | aiProcess_GenNormals));
        links[i] = new fcl::CollisionObjectd(meshes[i]);
        links[i]->computeAABB();
        colML.registerObject(links[i]);
    }
    colML.setup();
}

void UR5E::setLinksPoseL(Eigen::Quaterniond *quats, const Eigen::Vector3d *trans)
{
    for (size_t i = 0; i < 6; i++)
    {
        links[i + 1]->setTransform(quats[i], trans[i]);
        links[i + 1]->computeAABB();
        // colML.update(links[i + 1]);
    }
}

void UR5E::setLinksPoseL()
{
    setLinksPoseL(quaternions, translations);
}

void UR5E::setLinksPoseB(Eigen::Quaterniond *quats, const Eigen::Vector3d *trans)
{
    for (size_t i = 0; i < 6; i++)
    {
        linksB[i + 1]->setTransform(quats[i], quats[i] * offsets[i + 1] + trans[i]);
        linksB[i + 1]->computeAABB();
        // colMB.update(linksB[i + 1]);
    }
}

void UR5E::setLinksPoseB()
{
    setLinksPoseB(quaternions, translations);
}

void UR5E::calEndpoint()
{
    endpoint = quaternions[5] * endpointOffset + translations[5];
}

void UR5E::calEndquat()
{
    endquat = quaternions[5] * endquat_;
}

bool UR5E::selfCollide(fcl::CollisionObjectd **links)
{
    for (auto p : selfColPairs)
    {
        colRes.clear();
        fcl::collide(links[p.first], links[p.second], colReq, colRes);
        if (colRes.isCollision())
        {
            return true;
        }
    }
    return false;
}

bool UR5E::selfCollide(fcl::CollisionObjectd **links, int pair)
{
    colRes.clear();
    fcl::collide(links[selfColPairs[pair].first], links[selfColPairs[pair].second], colReq, colRes);
    return colRes.isCollision();
}

bool UR5E::collideL(fcl::CollisionObjectd *colO)
{
    colCB.result.clear();
    colML.collide(colO, &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

bool UR5E::collideL(fcl::NaiveCollisionManagerd *colM)
{
    colCB.result.clear();
    colM->collide(&colML, &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

bool UR5E::collideB(fcl::CollisionObjectd *colO)
{
    colCB.result.clear();
    colMB.collide(colO, &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

bool UR5E::collideB(fcl::NaiveCollisionManagerd *colM)
{
    colCB.result.clear();
    colMB.collide(colM, &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

bool UR5E::collideL(fcl::CollisionObjectd *colO, int link)
{
    colRes.clear();
    fcl::collide(links[link], colO, colReq, colRes);
    return colRes.isCollision();
}

bool UR5E::collideL(fcl::NaiveCollisionManagerd *colM, int link)
{
    colCB.result.clear();
    colM->collide(links[link], &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

bool UR5E::collideB(fcl::CollisionObjectd *colO, int link)
{
    colRes.clear();
    fcl::collide(linksB[link], colO, colReq, colRes);
    return colRes.isCollision();
}

bool UR5E::collideB(fcl::NaiveCollisionManagerd *colM, int link)
{
    colCB.result.clear();
    colM->collide(linksB[link], &colCB, fcl::DefaultCollisionFunction);
    return colCB.result.isCollision();
}

double UR5E::distance(fcl::NaiveCollisionManagerd &colM, fcl::CollisionObjectd *colO)
{
    disCB.result.clear();
    colM.distance(colO, &disCB, fcl::DefaultDistanceFunction);
    return disCB.result.min_distance;
}
