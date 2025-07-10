//////////////// choose methods and dimension
#define NNsample
// #define STAR
#define BIDIRECTION
#define DIM 6

//////////////// ablation
// #define NOBISECTION
// #define DOWNSAMPLE
#define JUMP 1.0
#define MAXLplus 0

// #define NNsampleDebug 1
////////////// don't change
#if DIM == 2
#define MAXL 3
#define NUM 10
#define BOD 2.5
#define OBS 10
#define DIS 5.5
#define MODE 1
#elif DIM == 3
#define MAXL 1
#define NUM 7
#define BOD 1.5
#define OBS 10
#define DIS 3.0
#define MODE 1
#elif DIM == 6
#define MAXL 5
#define NUM 4
#define BOD 6.0
#define OBS 4
#define DIS 3.0
#define MODE 1
#define INVT false
#endif

#include <vector>
#include <armadillo>
#include <openvino/openvino.hpp>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <ompl/config.h>
#include <ompl/base/StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/DiscreteMotionValidator.h>
#include <ompl/util/Time.h>

#include "logger.hpp"
#include "spline.hpp"
#include "simulation.hpp"

namespace fs = std::filesystem;
namespace ob = ompl::base;
namespace og = ompl::geometric;

void assignJointSpaceState(ob::CompoundState *state, const double *joints)
{
    for (size_t i = 0; i < 6; i++)
    {
        state->components[i]->as<ob::SO2StateSpace::StateType>()->value = joints[i];
    }
}

void extractJointSpaceState(const ob::CompoundState *state, double *joints)
{
    for (size_t i = 0; i < 6; i++)
    {
        joints[i] = state->components[i]->as<ob::SO2StateSpace::StateType>()->value;
    }
}

class StateValidityCheckerWrapper : public ob::StateValidityChecker
{
public:
    StateValidityCheckerWrapper(const ob::SpaceInformationPtr &si, Simulation *SIM) : ob::StateValidityChecker(si)
    {
        this->SIM = SIM;
    }

    virtual bool isValid(const ob::State *state) const
    {
        totalChecks++;
#if DIM == 2
        valid = !SIM->colRobot2D(state->as<ob::RealVectorStateSpace::StateType>()->values);
#elif DIM == 3
        valid = !SIM->colRobot3D(state->as<ob::RealVectorStateSpace::StateType>()->values);
#else
        s = state->as<ob::CompoundStateSpace::StateType>();
        extractJointSpaceState(s, joints.data());
        valid = !SIM->colRobot6D(joints.data());
#endif
        if (valid)
        {
            validchecks++;
        }
        else
        {
            nonvalidchecks++;
        }
        return valid;
    }

    mutable int totalChecks = 0;
    mutable bool valid;
    mutable int validchecks = 0;
    mutable int nonvalidchecks = 0;

protected:
    mutable Simulation *SIM;
#if DIM == 6
    mutable const ompl::base::CompoundStateSpace::StateType *s;
    mutable std::array<double, 6> joints;
#endif
};

class NN
{
private:
public:
    Simulation *SIM;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor obs1, obs2, mask, mean, logstd;

    arma::Mat<float> obs1M, obs2M, meanM, logstdM;

    NN(Simulation *SIM) : SIM(SIM)
    {
        auto modelp = "cache/models/model" + std::to_string(DIM) + "D.xml";
        model = core.read_model(modelp.c_str());
        compiled_model = core.compile_model(model, "CPU");
        infer_request = compiled_model.create_infer_request();
        obs1 = infer_request.get_input_tensor(0);
        obs2 = infer_request.get_input_tensor(1);
        mask = infer_request.get_input_tensor(2);
        mean = infer_request.get_output_tensor(0);

        obs1.set_shape(ov::Shape({SIM->obsbuf1Mat.n_cols, SIM->obsbuf1Mat.n_rows}));
        obs2.set_shape(ov::Shape({SIM->obsbuf2Mat.n_cols, SIM->obsbuf2Mat.n_rows}));
        obs1M = arma::Mat<float>(obs1.data<float>(), SIM->obsbuf1Mat.n_cols, SIM->obsbuf1Mat.n_rows, false, true);
        obs2M = arma::Mat<float>(obs2.data<float>(), SIM->obsbuf2Mat.n_cols, SIM->obsbuf2Mat.n_rows, false, true);
        mask.set_shape(ov::Shape({ulong(SIM->maxSensoredObs * 2)}));
        mean.set_shape(ov::Shape({DIM * NUM}));
        meanM = arma::Mat<float>(mean.data<float>(), 1, DIM * NUM, false, true);
    }
    ~NN()
    {
    }
};

class NNSampler
{
private:
#ifdef NNsampleDebug
    Logger logger = Logger("sample_debug_test" + std::to_string(revert), "sample_debug");
#endif
    Simulation *SIM;
    double traj_dense;
    NN *nn;
    Spline spline = Spline();
    double boundSep;
    arma::Mat<double> meanMd, logstdMd, bound;
    std::normal_distribution<double> ndis;
    std::mt19937 random_gen;
    int idx = 10000000, idx0 = 10000000, idx1 = 10000000;
    arma::mat startp, goalp;
    const bool revert;
    double mindis = 1000;
    double maxstep;
#if DIM != 6
    double *result;
#else
    ob::CompoundState *result;
#endif
    tsl::robin_map<const ob::State *, int> obs_map;

    bool endEpisode = true;

public:
    std::size_t NNruns = 0, NNsamples = 0, L = 0;
    NNSampler(Simulation *SIM, NN *nn, bool revert = false) : SIM(SIM), nn(nn), revert(revert)
    {
        if (revert)
        {
            startp = SIM->goalp;
            goalp = SIM->startp;
        }
        else
        {
            startp = SIM->startp;
            goalp = SIM->goalp;
        }

        meanMd = arma::Mat<double>(1, DIM * NUM);
        bound = arma::Mat<double>(1, DIM * NUM);
        boundSep = DIM == 6 ? M_PI / BOD / NUM : BOD / NUM;
        for (size_t i = 0; i < NUM; i++)
        {
            std::fill(bound.memptr() + i * DIM,
                      bound.memptr() + i * DIM + DIM,
                      float(i + 1) * boundSep);
        }
#if DIM != 6
        traj_dense = SIM->traj_dense / 1.0;
#ifdef DOWNSAMPLE
        traj_dense /= 10;
#endif
        maxstep = JUMP * arma::norm(bound.cols(bound.n_cols - DIM - 1, bound.n_cols - 1));
#else
        traj_dense = SIM->traj_dense / 1.0;
#ifdef DOWNSAMPLE
        traj_dense /= 5;
#endif
        double tmpj[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        maxstep = JUMP * spline.distance(tmpj, bound.colptr(bound.n_cols - DIM - 1));
#endif
        random_gen.seed(SIM->seed);
#ifdef NNsampleDebug
#if DIM == 2
        logger.logGround(SIM->env_half_size);
        logger.logOBS(SIM->circles, SIM->rectangles);
        logger.logGoal(goalp.mem, DIM);
        logger.logRobot(SIM->robot_r);
#elif DIM == 3
        logger.logGround(SIM->env_half_size);
        // logger.logOBS(SIM.spheres, SIM.boxes, SIM.quats);
        logger.logGoal(goalp.mem, DIM);
        logger.logRobot(SIM->robot_r);
#else
        SIM->ur5e.setJoints(startp.mem);
        logger.logUR5E("assets/ur5e/collision/");
        logger.logOBS(SIM->spheres, SIM->boxes, SIM->quats);
        SIM->ur5e.setJoints(startp.mem);
        logger.logPos(SIM->ur5e.quaternions, SIM->ur5e.translations, 0);
        logger.logUR5E("assets/ur5e/collision/", "goal", 0xaaaaffaa);
        SIM->ur5e.setJoints(goalp.mem);
        logger.logPos(SIM->ur5e.quaternions, SIM->ur5e.translations, 0, "goal");
#endif
#endif
    }
    ~NNSampler()
    {
#ifdef NNsampleDebug
        logger.save("sample_debug" + std::to_string(revert) + ".rrd");
#endif
    }

    void setStart(ob::State *state, ob::State *state_out)
    {
#if DIM != 6
        result = state->as<ompl::base::RealVectorStateSpace::StateType>()->values;
        std::memcpy(SIM->pos.memptr(), result, DIM * sizeof(double));
#else
        result = state->as<ob::CompoundState>();
        extractJointSpaceState(result, SIM->pos.memptr());
#endif
        SIM->vel = goalp - startp;
        SIM->vel = SIM->vel / arma::norm(SIM->vel);
        SIM->lastPos = SIM->pos;
        SIM->lastVel = SIM->vel;
        int t = obs_map[state];
        if (t > 0)
        {
            this->sample(false, std::pow(2, t));
        }
        else
        {
            this->sample();
        }
        obs_map[state] = t + 1;
        mindis = 100000;
        L = 1;
        endEpisode = false;
        assign_state(state_out);
    }

    void sample(bool determinstic = true, double std_multiply = 1)
    {

#if DIM == 2
        SIM->genOBS2D(goalp);
#elif DIM == 3
        SIM->genOBS3D(goalp);
#else
        SIM->genOBS6D(goalp);
#endif
        std::copy(SIM->obsbuf1, SIM->obsbuf1 + SIM->obsbuf1Mat.n_cols * SIM->obsbuf1Mat.n_rows, nn->obs1.data<float>());
        std::copy(SIM->obsbuf2, SIM->obsbuf2 + SIM->obsbuf2Mat.n_cols * SIM->obsbuf2Mat.n_rows, nn->obs2.data<float>());
        std::copy(SIM->mask, SIM->mask + SIM->maxSensoredObs * 2, nn->mask.data<float>());

        nn->infer_request.infer();

        if (determinstic)
        {
            std::copy(nn->meanM.begin(), nn->meanM.end(), meanMd.begin());
        }
        else
        {
            for (size_t i = 0; i < nn->meanM.size(); i++)
            {
                ndis.param(std::normal_distribution<double>::param_type(nn->meanM[i], std_multiply * boundSep));
                meanMd[i] = ndis(random_gen);
            }
        }
        meanMd = arma::tanh(meanMd) % bound;
        NNruns++;
        L++;
#if DIM == 2
        SIM->setTraj2D(meanMd.mem, NUM);
        spline.interpolate(SIM->traj2.mem, NUM + 1, DIM, traj_dense, SIM->lastPos.mem);
#elif DIM == 3
        SIM->setTraj3D(meanMd.mem, NUM);
        spline.interpolate(SIM->traj2.mem, NUM + 1, DIM, traj_dense, SIM->lastPos.mem);
#else
        SIM->setTraj6D(meanMd.mem, NUM);
        spline.interpolate(SIM->traj2.mem, NUM + 1, DIM, traj_dense, SIM->lastPos.mem);
        spline.boundSO6(spline.x.memptr(), spline.x.n_cols);
#endif

#ifdef NNsampleDebug
        for (size_t i = 0; i < spline.x.n_cols; i++)
        {
            logger.logPos(spline.x.col(i).colmem, spline.thetas[i], NNsamples + i);
        }
#endif
        NNsamples += spline.x.n_cols;
#ifndef NOBISECTION
        idx = spline.x.n_cols - 1;
        idx0 = 0;
        idx1 = (spline.x.n_cols - 1) * 2;
#else
        idx = 1;
        idx0 = 0;
        idx1 = spline.x.n_cols;
#endif
    }

    inline void assign_state(ob::State *state)
    {
#if DIM != 6
        mindis = std::min(mindis, arma::norm(spline.x.col(idx) - goalp));
        result = state->as<ompl::base::RealVectorStateSpace::StateType>()->values;
        memcpy(result, spline.x.col(idx).colmem, DIM * sizeof(double));
#else
        mindis = std::min(mindis, spline.distance(spline.x.col(idx).colmem, goalp.mem));
        result = state->as<ob::CompoundState>();
        assignJointSpaceState(result, spline.x.col(idx).colmem);
#endif
    }

    bool getSample(ob::State *state, bool lastStateValid)
    {
        if (endEpisode)
        {
            return false;
        }
#ifndef NOBISECTION
        if (lastStateValid)
        {
            idx0 = idx;
            idx1 = spline.x.n_cols;
            idx = (idx0 + idx1) / 2;
        }
        else
        {
            idx1 = idx;
            idx = (idx0 + idx1) / 2;
        }
#else
        if (lastStateValid)
        {
            idx0 = idx;
            idx++;
        }
        else
        {
            idx1 = idx;
        }
#endif
        if (idx0 == spline.x.n_cols - 1)
        {
            if (L >= MAXL + MAXLplus)
            {
                goto SAMPLEGOAL;
            }
            SIM->pos = spline.x.col(idx0);
            memcpy(SIM->vel.memptr(), spline.derive(idx0), DIM * sizeof(double));
            SIM->lastPos = spline.x.col(idx0 - 1);
            memcpy(SIM->lastVel.memptr(), spline.derive(idx0 - 1), DIM * sizeof(double));
            this->sample();
        }

        if (idx == idx0 || idx == idx1)
        {
            goto SAMPLEGOAL;
        }
        assign_state(state);
        return true;

    SAMPLEGOAL:
        endEpisode = true;
        if (mindis > maxstep)
        {
            return false;
        }
#if DIM != 6
        result = state->as<ompl::base::RealVectorStateSpace::StateType>()->values;
        memcpy(result, goalp.mem, DIM * sizeof(double));
#else
        result = state->as<ob::CompoundState>();
        assignJointSpaceState(result, goalp.mem);
#endif
        return true;
    }
};

namespace ompl
{
    namespace geometric
    {
        class RLRRTstar : public ompl::geometric::RRTstar
        {
        public:
            Simulation *SIM;
            NN nn;
            NNSampler sampler;
            RLRRTstar(const base::SpaceInformationPtr &si, Simulation *SIM)
                : ompl::geometric::RRTstar(si), SIM(SIM), nn(SIM), sampler(SIM, &nn)
            {
                name_ = "RLRRTstar";
            }
            ~RLRRTstar()
            {
            }

            ompl::base::PlannerStatus
            solve(const base::PlannerTerminationCondition &ptc)
            {
                OMPL_DEBUG("%s solve function called.", name_.c_str());
                checkValidity();
                base::Goal *goal = pdef_->getGoal().get();
                auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

                bool symCost = opt_->isSymmetric();

                // Check if there are more starts
                if (pis_.haveMoreStartStates() == true)
                {
                    // There are, add them
                    while (const base::State *st = pis_.nextStart())
                    {
                        auto *motion = new Motion(si_);
                        si_->copyState(motion->state, st);
                        motion->cost = opt_->identityCost();
                        nn_->add(motion);
                        startMotions_.push_back(motion);
                    }

                    // And assure that, if we're using an informed sampler, it's reset
                    infSampler_.reset();
                }
                // No else

                if (nn_->size() == 0)
                {
                    OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
                    return base::PlannerStatus::INVALID_START;
                }

                // Allocate a sampler if necessary
                if (!sampler_ && !infSampler_)
                {
                    allocSampler();
                }

                OMPL_INFORM("%s: Started planning with %u states. Seeking a solution better than %.5f.", getName().c_str(), nn_->size(), opt_->getCostThreshold().value());

                if ((useTreePruning_ || useRejectionSampling_ || useInformedSampling_ || useNewStateRejection_) &&
                    !si_->getStateSpace()->isMetricSpace())
                    OMPL_WARN("%s: The state space (%s) is not metric and as a result the optimization objective may not satisfy "
                              "the triangle inequality. "
                              "You may need to disable pruning or rejection.",
                              getName().c_str(), si_->getStateSpace()->getName().c_str());

                const base::ReportIntermediateSolutionFn intermediateSolutionCallback = pdef_->getIntermediateSolutionCallback();

                Motion *approxGoalMotion = nullptr;
                double approxDist = std::numeric_limits<double>::infinity();

                auto *rmotion = new Motion(si_);
                base::State *rstate = rmotion->state;
                base::State *xstate = si_->allocState();

                std::vector<Motion *> nbh;

                std::vector<base::Cost> costs;
                std::vector<base::Cost> incCosts;
                std::vector<std::size_t> sortedCostIndices;

                std::vector<int> valid;
                unsigned int rewireTest = 0;
                unsigned int statesGenerated = 0;

                if (bestGoalMotion_)
                    OMPL_INFORM("%s: Starting planning with existing solution of cost %.5f", getName().c_str(),
                                bestCost_.value());

                if (useKNearest_)
                    OMPL_INFORM("%s: Initial k-nearest value of %u", getName().c_str(),
                                (unsigned int)std::ceil(k_rrt_ * log((double)(nn_->size() + 1u))));
                else
                    OMPL_INFORM(
                        "%s: Initial rewiring radius of %.2f", getName().c_str(),
                        std::min(maxDistance_, r_rrt_ * std::pow(log((double)(nn_->size() + 1u)) / ((double)(nn_->size() + 1u)),
                                                                 1 / (double)(si_->getStateDimension()))));

                // our functor for sorting nearest neighbors
                CostIndexCompare compareFn(costs, *opt_);

                bool lastStateValid = false;
                while (ptc == false)
                {
                    iterations_++;
                    // std::cout << iterations_ << "," << lastStateValid << std::endl;
                    if (!sampler.getSample(rstate, lastStateValid))
                    {
                        // sample random state (with goal biasing)
                        // Goal samples are only sampled until maxSampleCount() goals are in the tree, to prohibit duplicate goal
                        // states.
                        if (goal_s && goalMotions_.size() < goal_s->maxSampleCount() && rng_.uniform01() < goalBias_ &&
                            goal_s->canSample())
                            goal_s->sampleGoal(rstate);
                        else
                        {
                            // Attempt to generate a sample, if we fail (e.g., too many rejection attempts), skip the remainder of this
                            // loop and return to try again
                            if (!sampleUniform(rstate))
                                continue;
                        }
                        // ps: rstate is the random sample
                        // find closest state in the tree
                        Motion *nmotion = nn_->nearest(rmotion);

                        // ps: nmotion->state is the nearest state to rstate
                        sampler.setStart(nmotion->state, rstate);
                    }
                    // ps: rstate is the sample from NN
                    Motion *nmotion = nn_->nearest(rmotion);

                    if (intermediateSolutionCallback && si_->equalStates(nmotion->state, rstate))
                        continue;

                    // ps: dstate is the state to be added to the tree
                    base::State *dstate = rstate;
                    // find state to add to the tree
                    double d = si_->distance(nmotion->state, rstate);
                    if (d > maxDistance_)
                    {
                        si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
                        dstate = xstate;
                    }
                    // Check if the motion between the nearest state and the state to add is valid
                    lastStateValid = si_->checkMotion(nmotion->state, dstate);
                    if (lastStateValid)
                    {
                        // create a motion
                        auto *motion = new Motion(si_);
                        si_->copyState(motion->state, dstate);
                        motion->parent = nmotion;
                        motion->incCost = opt_->motionCost(nmotion->state, motion->state);
                        motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);

                        // Find nearby neighbors of the new motion
                        getNeighbors(motion, nbh);

                        rewireTest += nbh.size();
                        ++statesGenerated;

                        // cache for distance computations
                        //
                        // Our cost caches only increase in size, so they're only
                        // resized if they can't fit the current neighborhood
                        if (costs.size() < nbh.size())
                        {
                            costs.resize(nbh.size());
                            incCosts.resize(nbh.size());
                            sortedCostIndices.resize(nbh.size());
                        }

                        // cache for motion validity (only useful in a symmetric space)
                        //
                        // Our validity caches only increase in size, so they're
                        // only resized if they can't fit the current neighborhood
                        if (valid.size() < nbh.size())
                            valid.resize(nbh.size());
                        std::fill(valid.begin(), valid.begin() + nbh.size(), 0);

                        // Finding the nearest neighbor to connect to
                        // By default, neighborhood states are sorted by cost, and collision checking
                        // is performed in increasing order of cost
                        if (delayCC_)
                        {
                            // calculate all costs and distances
                            for (std::size_t i = 0; i < nbh.size(); ++i)
                            {
                                incCosts[i] = opt_->motionCost(nbh[i]->state, motion->state);
                                costs[i] = opt_->combineCosts(nbh[i]->cost, incCosts[i]);
                            }

                            // sort the nodes
                            //
                            // we're using index-value pairs so that we can get at
                            // original, unsorted indices
                            for (std::size_t i = 0; i < nbh.size(); ++i)
                                sortedCostIndices[i] = i;
                            std::sort(sortedCostIndices.begin(), sortedCostIndices.begin() + nbh.size(), compareFn);

                            // collision check until a valid motion is found
                            //
                            // ASYMMETRIC CASE: it's possible that none of these
                            // neighbors are valid. This is fine, because motion
                            // already has a connection to the tree through
                            // nmotion (with populated cost fields!).
                            for (std::vector<std::size_t>::const_iterator i = sortedCostIndices.begin();
                                 i != sortedCostIndices.begin() + nbh.size(); ++i)
                            {
                                if (nbh[*i] == nmotion ||
                                    ((!useKNearest_ || si_->distance(nbh[*i]->state, motion->state) < maxDistance_) &&
                                     si_->checkMotion(nbh[*i]->state, motion->state)))
                                {
                                    motion->incCost = incCosts[*i];
                                    motion->cost = costs[*i];
                                    motion->parent = nbh[*i];
                                    valid[*i] = 1;
                                    break;
                                }
                                else
                                    valid[*i] = -1;
                            }
                        }
                        else // if not delayCC
                        {
                            motion->incCost = opt_->motionCost(nmotion->state, motion->state);
                            motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);
                            // find which one we connect the new state to
                            for (std::size_t i = 0; i < nbh.size(); ++i)
                            {
                                if (nbh[i] != nmotion)
                                {
                                    incCosts[i] = opt_->motionCost(nbh[i]->state, motion->state);
                                    costs[i] = opt_->combineCosts(nbh[i]->cost, incCosts[i]);
                                    if (opt_->isCostBetterThan(costs[i], motion->cost))
                                    {
                                        if ((!useKNearest_ || si_->distance(nbh[i]->state, motion->state) < maxDistance_) &&
                                            si_->checkMotion(nbh[i]->state, motion->state))
                                        {
                                            motion->incCost = incCosts[i];
                                            motion->cost = costs[i];
                                            motion->parent = nbh[i];
                                            valid[i] = 1;
                                        }
                                        else
                                            valid[i] = -1;
                                    }
                                }
                                else
                                {
                                    incCosts[i] = motion->incCost;
                                    costs[i] = motion->cost;
                                    valid[i] = 1;
                                }
                            }
                        }

                        if (useNewStateRejection_)
                        {
                            if (opt_->isCostBetterThan(solutionHeuristic(motion), bestCost_))
                            {
                                nn_->add(motion);
                                motion->parent->children.push_back(motion);
                            }
                            else // If the new motion does not improve the best cost it is ignored.
                            {
                                si_->freeState(motion->state);
                                delete motion;
                                continue;
                            }
                        }
                        else
                        {
                            // add motion to the tree
                            nn_->add(motion);
                            motion->parent->children.push_back(motion);
                        }

                        bool checkForSolution = false;
                        for (std::size_t i = 0; i < nbh.size(); ++i)
                        {
                            if (nbh[i] != motion->parent)
                            {
                                base::Cost nbhIncCost;
                                if (symCost)
                                    nbhIncCost = incCosts[i];
                                else
                                    nbhIncCost = opt_->motionCost(motion->state, nbh[i]->state);
                                base::Cost nbhNewCost = opt_->combineCosts(motion->cost, nbhIncCost);
                                if (opt_->isCostBetterThan(nbhNewCost, nbh[i]->cost))
                                {
                                    bool motionValid;
                                    if (valid[i] == 0)
                                    {
                                        motionValid =
                                            (!useKNearest_ || si_->distance(nbh[i]->state, motion->state) < maxDistance_) &&
                                            si_->checkMotion(motion->state, nbh[i]->state);
                                    }
                                    else
                                    {
                                        motionValid = (valid[i] == 1);
                                    }

                                    if (motionValid)
                                    {
                                        // Remove this node from its parent list
                                        removeFromParent(nbh[i]);

                                        // Add this node to the new parent
                                        nbh[i]->parent = motion;
                                        nbh[i]->incCost = nbhIncCost;
                                        nbh[i]->cost = nbhNewCost;
                                        nbh[i]->parent->children.push_back(nbh[i]);

                                        // Update the costs of the node's children
                                        updateChildCosts(nbh[i]);

                                        checkForSolution = true;
                                    }
                                }
                            }
                        }

                        // Add the new motion to the goalMotion_ list, if it satisfies the goal
                        double distanceFromGoal;
                        if (goal->isSatisfied(motion->state, &distanceFromGoal))
                        {
                            motion->inGoal = true;
                            goalMotions_.push_back(motion);
                            checkForSolution = true;
                        }

                        // Checking for solution or iterative improvement
                        if (checkForSolution)
                        {
                            bool updatedSolution = false;
                            if (!bestGoalMotion_ && !goalMotions_.empty())
                            {
                                // We have found our first solution, store it as the best. We only add one
                                // vertex at a time, so there can only be one goal vertex at this moment.
                                bestGoalMotion_ = goalMotions_.front();
                                bestCost_ = bestGoalMotion_->cost;
                                updatedSolution = true;

                                OMPL_INFORM("%s: Found an initial solution with a cost of %.2f in %u iterations (%u "
                                            "vertices in the graph)",
                                            getName().c_str(), bestCost_.value(), iterations_, nn_->size());
                            }
                            else
                            {
                                // We already have a solution, iterate through the list of goal vertices
                                // and see if there's any improvement.
                                for (auto &goalMotion : goalMotions_)
                                {
                                    // Is this goal motion better than the (current) best?
                                    if (opt_->isCostBetterThan(goalMotion->cost, bestCost_))
                                    {
                                        bestGoalMotion_ = goalMotion;
                                        bestCost_ = bestGoalMotion_->cost;
                                        updatedSolution = true;

                                        // Check if it satisfies the optimization objective, if it does, break the for loop
                                        if (opt_->isSatisfied(bestCost_))
                                        {
                                            break;
                                        }
                                    }
                                }
                            }

                            if (updatedSolution)
                            {
                                if (useTreePruning_)
                                {
                                    pruneTree(bestCost_);
                                }

                                if (intermediateSolutionCallback)
                                {
                                    std::vector<const base::State *> spath;
                                    Motion *intermediate_solution =
                                        bestGoalMotion_->parent; // Do not include goal state to simplify code.

                                    // Push back until we find the start, but not the start itself
                                    while (intermediate_solution->parent != nullptr)
                                    {
                                        spath.push_back(intermediate_solution->state);
                                        intermediate_solution = intermediate_solution->parent;
                                    }

                                    intermediateSolutionCallback(this, spath, bestCost_);
                                }
                            }
                        }

                        // Checking for approximate solution (closest state found to the goal)
                        if (goalMotions_.size() == 0 && distanceFromGoal < approxDist)
                        {
                            approxGoalMotion = motion;
                            approxDist = distanceFromGoal;
                        }
                    }

                    // terminate if a sufficient solution is found
                    if (bestGoalMotion_ && opt_->isSatisfied(bestCost_))
                        break;
                }

                // Add our solution (if it exists)
                Motion *newSolution = nullptr;
                if (bestGoalMotion_)
                {
                    // We have an exact solution
                    newSolution = bestGoalMotion_;
                }
                else if (approxGoalMotion)
                {
                    // We don't have a solution, but we do have an approximate solution
                    newSolution = approxGoalMotion;
                }
                // No else, we have nothing

                // Add what we found
                if (newSolution)
                {
                    ptc.terminate();
                    // construct the solution path
                    std::vector<Motion *> mpath;
                    Motion *iterMotion = newSolution;
                    while (iterMotion != nullptr)
                    {
                        mpath.push_back(iterMotion);
                        iterMotion = iterMotion->parent;
                    }

                    // set the solution path
                    auto path(std::make_shared<PathGeometric>(si_));
                    for (int i = mpath.size() - 1; i >= 0; --i)
                        path->append(mpath[i]->state);

                    // Add the solution path.
                    base::PlannerSolution psol(path);
                    psol.setPlannerName(getName());

                    // If we don't have a goal motion, the solution is approximate
                    if (!bestGoalMotion_)
                        psol.setApproximate(approxDist);

                    // Does the solution satisfy the optimization objective?
                    psol.setOptimized(opt_, newSolution->cost, opt_->isSatisfied(bestCost_));
                    pdef_->addSolutionPath(psol);
                }
                // No else, we have nothing

                si_->freeState(xstate);
                if (rmotion->state)
                    si_->freeState(rmotion->state);
                delete rmotion;

                OMPL_INFORM("%s: Created %u new states. Checked %u rewire options. %u goal states in tree. Final solution cost "
                            "%.3f",
                            getName().c_str(), statesGenerated, rewireTest, goalMotions_.size(), bestCost_.value());

                // We've added a solution if newSolution == true, and it is an approximate solution if bestGoalMotion_ == false
                return {newSolution != nullptr, bestGoalMotion_ == nullptr};
            }
        };

        class RLRRTconnect : public ompl::geometric::RRTConnect
        {
        public:
            Simulation *SIM;
            NN nn;
            NNSampler samplers, samplerg;

            RLRRTconnect(const base::SpaceInformationPtr &si, Simulation *SIM)
                : ompl::geometric::RRTConnect(si), SIM(SIM), nn(SIM), samplers(SIM, &nn, false), samplerg(SIM, &nn, true)
            {
                name_ = "RLRRTconnect";
            }

            ompl::base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc)
            {
                OMPL_DEBUG("%s solve function called.", name_.c_str());
                checkValidity();
                auto *goal = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get());

                if (goal == nullptr)
                {
                    OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
                    return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
                }

                while (const base::State *st = pis_.nextStart())
                {
                    auto *motion = new Motion(si_);
                    si_->copyState(motion->state, st);
                    motion->root = motion->state;
                    tStart_->add(motion);
                }

                if (tStart_->size() == 0)
                {
                    OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
                    return base::PlannerStatus::INVALID_START;
                }

                if (!goal->couldSample())
                {
                    OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
                    return base::PlannerStatus::INVALID_GOAL;
                }

                if (!sampler_)
                    sampler_ = si_->allocStateSampler();

                OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(),
                            (int)(tStart_->size() + tGoal_->size()));

                TreeGrowingInfo tgi;
                tgi.xstate = si_->allocState();

                Motion *approxsol = nullptr;
                double approxdif = std::numeric_limits<double>::infinity();
                auto *rmotion = new Motion(si_);
                base::State *rstate = rmotion->state;
                bool solved = false;

                bool lastStateValids = false, lastStateValidg = false;
                Motion *nmotion;
                GrowState gs;
                int start_set;
                while (!ptc)
                {
                    TreeData &tree = startTree_ ? tStart_ : tGoal_;
                    tgi.start = startTree_;
                    startTree_ = !startTree_;
                    TreeData &otherTree = startTree_ ? tStart_ : tGoal_;

                    if (tGoal_->size() == 0 || pis_.getSampledGoalsCount() < tGoal_->size() / 2)
                    {
                        const base::State *st = tGoal_->size() == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
                        if (st != nullptr)
                        {
                            auto *motion = new Motion(si_);
                            si_->copyState(motion->state, st);
                            motion->root = motion->state;
                            tGoal_->add(motion);
                        }

                        if (tGoal_->size() == 0)
                        {
                            OMPL_ERROR("%s: Unable to sample any valid states for goal tree", getName().c_str());
                            break;
                        }
                    }
                    start_set = 0;
                    if (tgi.start)
                    {
                        while (true)
                        {
                            if (!samplers.getSample(rstate, lastStateValids))
                            {
                                if (start_set == 1)
                                {
                                    break;
                                }
                                /* sample random state */
                                sampler_->sampleUniform(rstate);
                                // ps: rstate is the random sample
                                // find closest state in the tree
                                nmotion = tree->nearest(rmotion);

                                // ps: nmotion->state is the nearest state to rstate
                                samplers.setStart(nmotion->state, rstate);
                                start_set++;
                            }
                            gs = growTree(tree, tgi, rmotion);
                            lastStateValids = gs == REACHED;
                        }
                    }
                    else
                    {
                        while (true)
                        {
                            if (!samplerg.getSample(rstate, lastStateValidg))
                            {
                                if (start_set == 1)
                                {
                                    break;
                                }
                                /* sample random state */
                                sampler_->sampleUniform(rstate);
                                // ps: rstate is the random sample
                                // find closest state in the tree
                                nmotion = tree->nearest(rmotion);

                                // ps: nmotion->state is the nearest state to rstate
                                samplerg.setStart(nmotion->state, rstate);
                                start_set++;
                            }
                            gs = growTree(tree, tgi, rmotion);
                            lastStateValidg = gs == REACHED;
                        }
                    }

                    if (gs != TRAPPED)
                    {
                        /* remember which motion was just added */
                        Motion *addedMotion = tgi.xmotion;

                        /* attempt to connect trees */

                        /* if reached, it means we used rstate directly, no need to copy again */
                        if (gs != REACHED)
                            si_->copyState(rstate, tgi.xstate);

                        tgi.start = startTree_;

                        /* if initial progress cannot be done from the otherTree, restore tgi.start */
                        GrowState gsc = growTree(otherTree, tgi, rmotion);
                        if (gsc == TRAPPED)
                            tgi.start = !tgi.start;

                        while (gsc == ADVANCED)
                            gsc = growTree(otherTree, tgi, rmotion);

                        /* update distance between trees */
                        const double newDist = tree->getDistanceFunction()(addedMotion, otherTree->nearest(addedMotion));
                        if (newDist < distanceBetweenTrees_)
                        {
                            distanceBetweenTrees_ = newDist;
                            // OMPL_INFORM("Estimated distance to go: %f", distanceBetweenTrees_);
                        }

                        Motion *startMotion = tgi.start ? tgi.xmotion : addedMotion;
                        Motion *goalMotion = tgi.start ? addedMotion : tgi.xmotion;

                        /* if we connected the trees in a valid way (start and goal pair is valid)*/
                        if (gsc == REACHED && goal->isStartGoalPairValid(startMotion->root, goalMotion->root))
                        {
                            // it must be the case that either the start tree or the goal tree has made some progress
                            // so one of the parents is not nullptr. We go one step 'back' to avoid having a duplicate state
                            // on the solution path
                            if (startMotion->parent != nullptr)
                                startMotion = startMotion->parent;
                            else
                                goalMotion = goalMotion->parent;

                            connectionPoint_ = std::make_pair(startMotion->state, goalMotion->state);

                            /* construct the solution path */
                            Motion *solution = startMotion;
                            std::vector<Motion *> mpath1;
                            while (solution != nullptr)
                            {
                                mpath1.push_back(solution);
                                solution = solution->parent;
                            }

                            solution = goalMotion;
                            std::vector<Motion *> mpath2;
                            while (solution != nullptr)
                            {
                                mpath2.push_back(solution);
                                solution = solution->parent;
                            }

                            auto path(std::make_shared<PathGeometric>(si_));
                            path->getStates().reserve(mpath1.size() + mpath2.size());
                            for (int i = mpath1.size() - 1; i >= 0; --i)
                                path->append(mpath1[i]->state);
                            for (auto &i : mpath2)
                                path->append(i->state);

                            pdef_->addSolutionPath(path, false, 0.0, getName());
                            solved = true;
                            break;
                        }
                        else
                        {
                            // We didn't reach the goal, but if we were extending the start
                            // tree, then we can mark/improve the approximate path so far.
                            if (tgi.start)
                            {
                                // We were working from the startTree.
                                double dist = 0.0;
                                goal->isSatisfied(tgi.xmotion->state, &dist);
                                if (dist < approxdif)
                                {
                                    approxdif = dist;
                                    approxsol = tgi.xmotion;
                                }
                            }
                        }
                    }
                }

                si_->freeState(tgi.xstate);
                si_->freeState(rstate);
                delete rmotion;

                OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(), tStart_->size() + tGoal_->size(),
                            tStart_->size(), tGoal_->size());

                if (approxsol && !solved)
                {
                    /* construct the solution path */
                    std::vector<Motion *> mpath;
                    while (approxsol != nullptr)
                    {
                        mpath.push_back(approxsol);
                        approxsol = approxsol->parent;
                    }

                    auto path(std::make_shared<PathGeometric>(si_));
                    for (int i = mpath.size() - 1; i >= 0; --i)
                        path->append(mpath[i]->state);
                    pdef_->addSolutionPath(path, true, approxdif, getName());
                    return base::PlannerStatus::APPROXIMATE_SOLUTION;
                }

                return solved ? base::PlannerStatus::EXACT_SOLUTION : base::PlannerStatus::TIMEOUT;
            }
        };

        class RLRRT : public ompl::geometric::RRT
        {
        public:
            Simulation *SIM;
            NN nn;
            NNSampler sampler;
            RLRRT(const base::SpaceInformationPtr &si, Simulation *SIM)
                : ompl::geometric::RRT(si), SIM(SIM), nn(SIM), sampler(SIM, &nn)
            {
                name_ = "RLRRT";
            }
            ~RLRRT()
            {
            }

            ompl::base::PlannerStatus
            solve(const base::PlannerTerminationCondition &ptc)
            {
                checkValidity();
                base::Goal *goal = pdef_->getGoal().get();
                auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

                while (const base::State *st = pis_.nextStart())
                {
                    auto *motion = new Motion(si_);
                    si_->copyState(motion->state, st);
                    nn_->add(motion);
                }

                if (nn_->size() == 0)
                {
                    OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
                    return base::PlannerStatus::INVALID_START;
                }

                if (!sampler_)
                    sampler_ = si_->allocStateSampler();

                OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());

                Motion *solution = nullptr;
                Motion *approxsol = nullptr;
                double approxdif = std::numeric_limits<double>::infinity();
                auto *rmotion = new Motion(si_);
                base::State *rstate = rmotion->state;
                base::State *xstate = si_->allocState();

                bool lastStateValid = false;
                while (ptc == false)
                {
                    if (!sampler.getSample(rstate, lastStateValid))
                    {
                        /* sample random state (with goal biasing) */
                        if ((goal_s != nullptr) && rng_.uniform01() < goalBias_ && goal_s->canSample())
                            goal_s->sampleGoal(rstate);
                        else
                            sampler_->sampleUniform(rstate);
                        // ps: rstate is the random sample
                        // find closest state in the tree
                        Motion *nmotion = nn_->nearest(rmotion);

                        // ps: nmotion->state is the nearest state to rstate
                        sampler.setStart(nmotion->state, rstate);
                    }
                    /* find closest state in the tree */
                    Motion *nmotion = nn_->nearest(rmotion);
                    base::State *dstate = rstate;

                    /* find state to add */
                    double d = si_->distance(nmotion->state, rstate);
                    if (d > maxDistance_)
                    {
                        si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
                        dstate = xstate;
                    }
                    // Check if the motion between the nearest state and the state to add is valid
                    lastStateValid = si_->checkMotion(nmotion->state, dstate);
                    if (lastStateValid)
                    {
                        if (addIntermediateStates_)
                        {
                            std::vector<base::State *> states;
                            const unsigned int count = si_->getStateSpace()->validSegmentCount(nmotion->state, dstate);

                            if (si_->getMotionStates(nmotion->state, dstate, states, count, true, true))
                                si_->freeState(states[0]);

                            for (std::size_t i = 1; i < states.size(); ++i)
                            {
                                auto *motion = new Motion;
                                motion->state = states[i];
                                motion->parent = nmotion;
                                nn_->add(motion);

                                nmotion = motion;
                            }
                        }
                        else
                        {
                            auto *motion = new Motion(si_);
                            si_->copyState(motion->state, dstate);
                            motion->parent = nmotion;
                            nn_->add(motion);

                            nmotion = motion;
                        }

                        double dist = 0.0;
                        bool sat = goal->isSatisfied(nmotion->state, &dist);
                        if (sat)
                        {
                            approxdif = dist;
                            solution = nmotion;
                            break;
                        }
                        if (dist < approxdif)
                        {
                            approxdif = dist;
                            approxsol = nmotion;
                        }
                    }
                }

                bool solved = false;
                bool approximate = false;
                if (solution == nullptr)
                {
                    solution = approxsol;
                    approximate = true;
                }

                if (solution != nullptr)
                {
                    lastGoalMotion_ = solution;

                    /* construct the solution path */
                    std::vector<Motion *> mpath;
                    while (solution != nullptr)
                    {
                        mpath.push_back(solution);
                        solution = solution->parent;
                    }

                    /* set the solution path */
                    auto path(std::make_shared<PathGeometric>(si_));
                    for (int i = mpath.size() - 1; i >= 0; --i)
                        path->append(mpath[i]->state);
                    pdef_->addSolutionPath(path, approximate, approxdif, getName());
                    solved = true;
                }

                si_->freeState(xstate);
                if (rmotion->state != nullptr)
                    si_->freeState(rmotion->state);
                delete rmotion;

                OMPL_INFORM("%s: Created %u states", getName().c_str(), nn_->size());

                return {solved, approximate};
            }
        };
    }
}

int main(int argc, char const *argv[])
{
    const int dim = DIM;
    int seed = 0;
    if (argc > 1)
    {
        seed = std::atoi(argv[1]);
    }
    double cost2stop = 1000000;
    if (argc > 2)
    {
        cost2stop = std::atof(argv[2]);
    }
    double timemax = 1;
    if (argc > 3)
    {
        timemax = std::atof(argv[3]);
    }
    int uselogger = 1;
    if (argc > 4)
    {
        uselogger = std::atoi(argv[4]);
    }
    int loggerspawn = 1;
    if (argc > 5)
    {
        loggerspawn = std::atoi(argv[5]);
    }
    auto SIM = Simulation(seed, dim,
                          OBS, DIS, "assets/");

    const double colResolution = SIM.traj_dense;
    auto spline = Spline();
    auto startp = arma::mat(dim, 1);
    auto endp = arma::mat(dim, 1);

    fs::path dirPath = "cache/envs/" + std::to_string(DIM) + "D";
    if (!fs::exists(dirPath))
    {
        fs::create_directories(dirPath);
    }
    fs::path filePath = "cache/envs/" + std::to_string(DIM) + "D/" + std::to_string(seed) + ".yaml";
    std::string yaml;
    if (!fs::exists(filePath))
    {
#if DIM == 2
        SIM.generate2D(0.35);
        SIM.genStartgoal2D(SIM.max_radius, 0.4, MODE);
#elif DIM == 3
        SIM.generate3D(0.25);
        SIM.genStartgoal3D(SIM.max_radius, 0.4, MODE);
#else
        bool valid = false;
        while (!valid)
        {
            SIM.generate6D(4, 4, 0.15, 0.3, 0.15);
            valid = SIM.genStartgoal6D(MODE, INVT);
        }
#endif
        yaml = SIM.toYAML();
        saveStringToFile(yaml, filePath.c_str());
    }
    else
    {
        yaml = readFileToString(filePath.c_str());
        SIM.fromYAML(yaml.c_str());
#ifdef INVT
#if INVT
        auto tmp = SIM.startp;
        SIM.startp = SIM.goalp;
        SIM.goalp = tmp;
#endif
#endif
    }
    startp = SIM.startp;
    endp = SIM.goalp;

    SIM.pos = SIM.startp;
    SIM.vel = SIM.goalp - SIM.startp;
    SIM.vel = SIM.vel / arma::norm(SIM.vel);

    auto logger = Logger("test" + std::to_string(DIM) + "D", "SIM" + std::to_string(DIM) + "D");
    int time = 0;
    if (uselogger)
    {
        if (loggerspawn)
        {
            auto e = logger.rec->connect_grpc("rerun+http://127.0.0.1:9876/proxy");
            if (e.is_err())
            {
                throw std::runtime_error(e.description);
            }
            // logger.rec->spawn().exit_on_failure();
        }

#if DIM == 2
        logger.logGround(SIM.env_half_size);
        logger.logOBS(SIM.circles, SIM.rectangles);
        logger.logGoal(endp.mem, dim);
        logger.logRobot(SIM.robot_r);
#elif DIM == 3
        logger.logGround(SIM.env_half_size);
        // logger.logOBS(SIM.spheres, SIM.boxes, SIM.quats);
        logger.logGoal(endp.mem, dim);
        logger.logRobot(SIM.robot_r);
#else
        logger.logOBS(SIM.spheres, SIM.boxes, SIM.quats);
        logger.logUR5E("assets/ur5e/collision/");
        SIM.ur5e.setJoints(startp.mem);
        logger.logPos(SIM.ur5e.quaternions, SIM.ur5e.translations, 0);
        logger.logUR5E("assets/ur5e/collision/", "goal", 0xaaaaffaa);
        SIM.ur5e.setJoints(endp.mem);
        logger.logPos(SIM.ur5e.quaternions, SIM.ur5e.translations, 0, "goal");
#endif
    }

#if DIM == 6
    auto space = std::make_shared<ob::CompoundStateSpace>();
    for (size_t i = 0; i < 6; i++)
    {
        space->addSubspace(std::make_shared<ob::SO2StateSpace>(), 1);
    }
#else
    auto space = std::make_shared<ob::RealVectorStateSpace>(dim);
    ob::RealVectorBounds bounds(dim);
    bounds.setLow(-SIM.env_half_size);
    bounds.setHigh(SIM.env_half_size);
    space->setBounds(bounds);
    space->setup();
#endif

    auto si = std::make_shared<ob::SpaceInformation>(space);
    auto StateValidityChecker = std::make_shared<StateValidityCheckerWrapper>(si, &SIM);
    si->setStateValidityChecker(StateValidityChecker);
    si->setStateValidityCheckingResolution(1 / colResolution / si->getStateSpace()->getMaximumExtent());
    si->setup();

#if DIM != 6
    auto start = space->allocState()->as<ompl::base::RealVectorStateSpace::StateType>();
    auto goal = space->allocState()->as<ompl::base::RealVectorStateSpace::StateType>();
    start->values[0] = startp[0];
    start->values[1] = startp[1];
    goal->values[0] = endp[0];
    goal->values[1] = endp[1];
    if (dim == 3)
    {
        start->values[2] = startp[2];
        goal->values[2] = endp[2];
    }
#else
    auto start = space->allocState()->as<ob::CompoundState>();
    auto goal = space->allocState()->as<ob::CompoundState>();
    assignJointSpaceState(start, startp.mem);
    assignJointSpaceState(goal, endp.mem);
#endif
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(start, goal);
    ob::OptimizationObjectivePtr obj(new ob::PathLengthOptimizationObjective(si));
    cost2stop = si->distance(start, goal) * cost2stop;
    obj->setCostThreshold(ob::Cost(cost2stop));
    pdef->setOptimizationObjective(obj);

#ifdef NNsample
#ifdef STAR
    auto planner = std::make_shared<og::RLRRTstar>(si, &SIM);
#elif defined BIDIRECTION
    auto planner = std::make_shared<og::RLRRTconnect>(si, &SIM);
#else
    auto planner = std::make_shared<og::RLRRT>(si, &SIM);
#endif
#else
#ifdef STAR
    auto planner = std::make_shared<og::RRTstar>(si);
#elif defined BIDIRECTION
    auto planner = std::make_shared<og::RRTConnect>(si);
#else
    auto planner = std::make_shared<og::RRT>(si);
#endif
#endif
    planner->setProblemDefinition(pdef);
#ifdef STAR
#if DIM == 6
    planner->setRewireFactor(0.1); // default to 1.1
#endif
#endif
    planner->setup();

    auto startt = std::chrono::system_clock::now();
    const ompl::time::point endTime(ompl::time::now() + ompl::time::seconds(timemax));
    ob::PlannerStatus solved = planner->ob::Planner::solve([endTime]
                                                           { return ompl::time::now() > endTime; },
                                                           1.0 / 1000.0);
    auto endt = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = endt - startt;
    std::cout << "Time(s): " << elapsed_seconds.count() << std::endl;
    if (solved)
    {
        auto path = pdef->getSolutionPath()->as<og::PathGeometric>();
        std::cout << "Found solution:" << std::endl;
        std::cout << "Cost: " << path->cost(obj) << std::endl;
        std::cout << "Path length: " << path->length() << std::endl;
        std::cout << "Path smoothness: " << path->smoothness() << std::endl;

#if DIM != 6
        ob::RealVectorStateSpace::StateType *p_ = si->allocState()->as<ob::RealVectorStateSpace::StateType>();
        ob::RealVectorStateSpace::StateType *s_ = si->allocState()->as<ob::RealVectorStateSpace::StateType>();
        ob::RealVectorStateSpace::StateType *e_ = si->allocState()->as<ob::RealVectorStateSpace::StateType>();
        si->copyState(s_, path->getStates().front()->as<ob::RealVectorStateSpace::StateType>());
        si->copyState(e_, path->getStates().back()->as<ob::RealVectorStateSpace::StateType>());
        p_->values[0] = endp.at(0);
        p_->values[1] = endp.at(1);
        if (dim == 3)
        {
            p_->values[2] = endp.at(2);
        }
#else
        ob::CompoundState *p_ = si->allocState()->as<ob::CompoundState>();
        ob::CompoundState *s_ = si->allocState()->as<ob::CompoundState>();
        ob::CompoundState *e_ = si->allocState()->as<ob::CompoundState>();
        si->copyState(s_, path->getStates().front()->as<ob::CompoundState>());
        si->copyState(e_, path->getStates().back()->as<ob::CompoundState>());
        assignJointSpaceState(p_, endp.mem);
#endif
        double finalError = si->distance(e_, p_);
        double achivement = si->distance(s_, e_) - finalError;
        std::cout << "Achivement: " << achivement << std::endl;
        std::cout << "Per traj_len achivement: " << achivement / path->length() << std::endl;
        std::cout << "Final error: " << finalError << std::endl;
        std::cout << "Collision Checks: " << StateValidityChecker->totalChecks << std::endl;
#ifdef STAR
        std::cout << "Iterations: " << planner->numIterations() << std::endl;
#endif
        std::cout << "Valid Collision Checks: " << double(StateValidityChecker->validchecks) / StateValidityChecker->totalChecks << std::endl;
        std::cout << "nonValid Collision Checks: " << double(StateValidityChecker->nonvalidchecks) / StateValidityChecker->totalChecks << std::endl;
        std::cout << "OBS number: " << SIM.colM.size() << std::endl;
#if DIM == 2
        std::cout << "obs1 number: " << SIM.circles.size() << std::endl;
        std::cout << "obs2 number: " << SIM.rectangles.size() << std::endl;
#else
        std::cout << "obs1 number: " << SIM.spheres.size() << std::endl;
        std::cout << "obs2 number: " << SIM.boxes.size() << std::endl;
#endif
#ifdef NNsample
#ifdef BIDIRECTION
        std::cout << "NNruns: " << planner->samplers.NNruns + planner->samplerg.NNruns << std::endl;
        std::cout << "NNsamples: " << planner->samplers.NNsamples + planner->samplerg.NNsamples << std::endl;
#else
        std::cout << "NNruns: " << planner->sampler.NNruns << std::endl;
        std::cout << "NNsamples: " << planner->sampler.NNsamples << std::endl;
#endif
#endif
        if (uselogger == 0)
        {
            return 0;
        }
        size_t step_size = 10;
        auto traj = new double[(step_size * 2) * dim];
        auto first = new double[dim];
        auto last = new double[dim];
        double *firstptr, *lastptr;
        int s_idx = 0, e_idx = 0;
        for (int i = 0; i < path->getStates().size(); i++)
        {
            s_idx = e_idx;
            e_idx = s_idx + step_size - 1;
            if (e_idx >= path->getStates().size() - 1)
            {
                if (s_idx == 0)
                {
                    e_idx = path->getStates().size() - 1;
                }
                else
                {
                    break;
                }
            }
            if (e_idx + step_size >= path->getStates().size() - 1)
            {
                e_idx = path->getStates().size() - 1;
            }

            if (s_idx - 1 >= 0)
            {
#if DIM != 6

                s_ = path->getStates()[s_idx - 1]->as<ompl::base::RealVectorStateSpace::StateType>();
                first[0] = s_->operator[](0);
                first[1] = s_->operator[](1);
#if DIM == 3
                first[2] = s_->operator[](2);
#endif
#else
                p_ = path->getStates().at(s_idx - 1)->as<ob::CompoundState>();
                extractJointSpaceState(p_, first);
#endif
                firstptr = first;
            }
            else
            {
                firstptr = nullptr;
            }
            if (e_idx + 1 < path->getStates().size())
            {
#if DIM != 6
                e_ = path->getStates()[e_idx + 1]->as<ompl::base::RealVectorStateSpace::StateType>();
                last[0] = e_->operator[](0);
                last[1] = e_->operator[](1);
#if DIM == 3
                last[2] = e_->operator[](2);
#endif
#else
                p_ = path->getStates().at(e_idx + 1)->as<ob::CompoundState>();
                extractJointSpaceState(p_, last);
#endif
                lastptr = last;
            }
            else
            {
                lastptr = nullptr;
            }
#if DIM != 6
            for (size_t j = s_idx; j <= e_idx; j++)
            {
                p_ = path->getStates().at(j)->as<ompl::base::RealVectorStateSpace::StateType>();
                traj[(j - s_idx) * dim] = p_->operator[](0);
                traj[(j - s_idx) * dim + 1] = p_->operator[](1);
#if DIM == 3
                traj[(j - s_idx) * dim + 2] = p_->operator[](2);
#endif
            }
            spline.interpolate(traj, e_idx - s_idx + 1, dim, colResolution, firstptr, lastptr);
            spline.derive_all();
            if (uselogger == 1)
            {
                for (size_t j = (firstptr == nullptr ? 0 : 1); j < spline.bsn; j++)
                {
#if DIM == 2
                    logger.logPos(spline.x.col(j).colmem, spline.thetas[j], time);
#elif DIM == 3
                    logger.logPos(spline.x.col(j).colmem, spline.dx.col(j).colmem, time);
#endif
                    time++;
                }
            }
#else
            for (size_t j = s_idx; j <= e_idx; j++)
            {
                p_ = path->getStates().at(j)->as<ob::CompoundState>();
                extractJointSpaceState(p_, traj + (j - s_idx) * 6);
            }
            spline.unboundSO6(traj, e_idx - s_idx + 1);
            spline.interpolate(traj, e_idx - s_idx + 1, dim, colResolution, firstptr, lastptr);
            spline.boundSO6(spline.x.memptr(), spline.x.n_cols);
            if (uselogger == 1)
            {
                for (size_t j = (firstptr == nullptr ? 0 : 1); j < spline.x.n_cols; j++)
                {
                    SIM.ur5e.setJoints(spline.x.col(j).colmem);
                    logger.logPos(SIM.ur5e.quaternions, SIM.ur5e.translations, time);
                    time++;
                }
            }
#endif
        }
        if (!loggerspawn)
        {
            logger.save("test" + std::to_string(DIM) + "D.rrd");
        }
    }
    return 0;
}
