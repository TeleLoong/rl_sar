/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_LITE3_HPP
#define RL_REAL_LITE3_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "fsm.hpp"

// Lite3 SDK
#include "sender.h"
#include "receiver.h"
#include "robot_types.h"
#include <cmath>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdint>

//Retroid Gamepad
#include "gamepad.h"
#include "retroid_gamepad.h"
#include "gamepad_keys.h"

#if defined(USE_ROS1) && defined(USE_ROS)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#elif defined(USE_ROS2) && defined(USE_ROS)
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class RL_Real : public RL
#if defined(USE_ROS2) && defined(USE_ROS)
    , public rclcpp::Node
#endif
{
public:
    RL_Real();
    ~RL_Real();

private:
    enum class JointTestWaveMode
    {
        Hold = 0,
        Square = 1,
        Sine = 2,
    };

    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
    void RunModel();
    void RobotControl();
    void RunJointPositionMonitor();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_udpRecv;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // Lite3 SDK interface
    Sender* sender_ = nullptr;
    Receiver* receiver_ = nullptr;
    RobotCmd robot_joint_cmd_{};
    RobotData* robot_data_=nullptr;
    void UDPRecv();
    void EulerToQuaternion(float roll, float pitch, float yaw, float q[4]);

    //Retroid Gamepad
    std::shared_ptr<RetroidGamepad> gamepad_ptr_;
    RetroidKeys rt_keys_record_, rt_keys_;
    bool first_flag_;

    // joint order test (raw hardware index)
    bool joint_test_mode_ = false;
    bool joint_test_inited_ = false;
    int joint_test_hw_idx_ = 0;     // 0..11 (SDK order)
    JointTestWaveMode joint_test_wave_mode_ = JointTestWaveMode::Hold;
    bool joint_test_hold_positive_ = true;
    bool joint_test_hold_others_ = false; // non-selected joints stay in passive gains by default
    bool joint_test_use_cmd_base_ = false; // default: use measured state as baseline to avoid global jumps
    double joint_test_amp_rad_ = 0.0;
    double joint_test_freq_hz_ = 0.25;
    double joint_test_kp_ = 10.0;
    double joint_test_kd_ = 0.30;
    double joint_test_kp_other_ = 10.0;
    double joint_test_kd_other_ = 0.30;
    std::array<double, 12> joint_test_q0_{};
    std::chrono::steady_clock::time_point joint_test_t0_{};
    void RunJointOrderTest();

    // read-only joint monitor mode
    bool joint_monitor_mode_ = false;
    std::chrono::steady_clock::time_point joint_monitor_last_print_t_{};
    uint32_t joint_monitor_frame_ = 0;

    // others
    int motiontime = 0;
    std::vector<double> mapped_joint_positions;
    std::vector<double> mapped_joint_velocities;

#if defined(USE_ROS1) && defined(USE_ROS)
    geometry_msgs::Twist cmd_vel;
    ros::Subscriber cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
#elif defined(USE_ROS2) && defined(USE_ROS)
    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
#endif
};

#endif // RL_REAL_LITE3_HPP
