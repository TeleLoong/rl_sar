/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_lite3.hpp"

static const char* LegNameFromIdx(int hw_idx)
{
    static const char* kLegs[] = {"FL", "FR", "HL", "HR"};
    const int leg = hw_idx / 3;
    if (leg < 0 || leg > 3) return "NA";
    return kLegs[leg];
}

static const char* JointNameFromIdx(int hw_idx)
{
    static const char* kJoints[] = {"HipX", "HipY", "Knee"};
    const int j = hw_idx % 3;
    if (j < 0 || j > 2) return "NA";
    return kJoints[j];
}

RL_Real::RL_Real()
#if defined(USE_ROS2) && defined(USE_ROS)
    : rclcpp::Node("rl_real_node")
#endif
{
#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // read params from yaml
    this->ang_vel_type = "ang_vel_world";
    this->robot_name = "lite3";
    this->ReadYamlBase(this->robot_name);

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init torch
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);


    // Network init
    int local_port = 43987;
    int robot_port = 43893;
    std::string robot_ip = "192.168.2.1";
    // init robot
    this->receiver_ = new Receiver();
    this->sender_ = new Sender(robot_ip, robot_port);
    this->sender_->RobotStateInit();
    this->InitOutputs();
    this->InitControl();
    this->receiver_->StartWork();
    this->robot_data_ = &(receiver_->GetState());

    // init gamepad
    this->gamepad_ptr_ = std::make_shared<RetroidGamepad>(12121);
    this->first_flag_ = true;
    this->gamepad_ptr_->StartDataThread();

    // loop
    this->loop_udpRecv = std::make_shared<LoopFunc>("loop_udpRecv", 0.002, std::bind(&RL_Real::UDPRecv, this), 3);
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_udpRecv->start();
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    this->loop_udpRecv->shutdown();
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    this->gamepad_ptr_->StopDataThread();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<double> *state)
{
    this->rt_keys_ = this->gamepad_ptr_->GetKeys();
    if(this->first_flag_){
        this->rt_keys_record_ = this->rt_keys_;
        this->first_flag_ = false;
    }
    if (this->rt_keys_.A != this->rt_keys_record_.A) this->control.SetGamepad(Input::Gamepad::A);
    if (this->rt_keys_.B != this->rt_keys_record_.B) this->control.SetGamepad(Input::Gamepad::B);
    if (this->rt_keys_.X != this->rt_keys_record_.X) this->control.SetGamepad(Input::Gamepad::X);
    if (this->rt_keys_.Y != this->rt_keys_record_.Y) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->rt_keys_.L1 != this->rt_keys_record_.L1) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->rt_keys_.R1 != this->rt_keys_record_.R1) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->rt_keys_.left_axis_button != this->rt_keys_record_.left_axis_button) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->rt_keys_.right_axis_button != this->rt_keys_record_.right_axis_button) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->rt_keys_.up != this->rt_keys_record_.up) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->rt_keys_.down != this->rt_keys_record_.down) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->rt_keys_.left != this->rt_keys_record_.left) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->rt_keys_.right != this->rt_keys_record_.right) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.A != this->rt_keys_record_.A)) this->control.SetGamepad(Input::Gamepad::LB_A);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.B != this->rt_keys_record_.B)) this->control.SetGamepad(Input::Gamepad::LB_B);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.X != this->rt_keys_record_.X)) this->control.SetGamepad(Input::Gamepad::LB_X);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.Y != this->rt_keys_record_.Y)) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.left_axis_button != this->rt_keys_record_.left_axis_button)) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.right_axis_button != this->rt_keys_record_.right_axis_button)) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.up != this->rt_keys_record_.up)) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.down != this->rt_keys_record_.down)) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.left != this->rt_keys_record_.left)) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if ((this->rt_keys_.L1 != this->rt_keys_record_.L1)&&(this->rt_keys_.right != this->rt_keys_record_.right)) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.A != this->rt_keys_record_.A)) this->control.SetGamepad(Input::Gamepad::RB_A);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.B != this->rt_keys_record_.B)) this->control.SetGamepad(Input::Gamepad::RB_B);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.X != this->rt_keys_record_.X)) this->control.SetGamepad(Input::Gamepad::RB_X);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.Y != this->rt_keys_record_.Y)) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.left_axis_button != this->rt_keys_record_.left_axis_button)) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.right_axis_button != this->rt_keys_record_.right_axis_button)) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.up != this->rt_keys_record_.up)) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.down != this->rt_keys_record_.down)) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.left != this->rt_keys_record_.left)) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.right != this->rt_keys_record_.right)) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if ((this->rt_keys_.R1 != this->rt_keys_record_.R1)&&(this->rt_keys_.R1 != this->rt_keys_record_.R1)) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->rt_keys_.left_axis_y;
    this->control.y = -this->rt_keys_.left_axis_x;
    this->control.yaw = -this->rt_keys_.right_axis_x;
       
    float q[4];
    EulerToQuaternion(this->robot_data_->imu.angle_roll, this->robot_data_->imu.angle_pitch, this->robot_data_->imu.angle_yaw, q);

    state->imu.quaternion[0] = q[0]; // w
    state->imu.quaternion[1] = q[1]; // x
    state->imu.quaternion[2] = q[2]; // y
    state->imu.quaternion[3] = q[3]; // z

    state->imu.gyroscope[0] = this->robot_data_->imu.angular_velocity_roll;
    state->imu.gyroscope[1] = this->robot_data_->imu.angular_velocity_pitch;
    state->imu.gyroscope[2] = this->robot_data_->imu.angular_velocity_yaw;

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].position;
        state->motor_state.dq[i] = this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].velocity;
        state->motor_state.tau_est[i] = this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].torque;
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].position = command->motor_command.q[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].velocity = command->motor_command.dq[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].kp = command->motor_command.kp[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].kd = command->motor_command.kd[i];
        this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].torque = command->motor_command.tau[i];
    }

    this->sender_->SendCmd(robot_joint_cmd_);
}

void RL_Real::RobotControl()
{
    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::T)
    {
        this->joint_test_mode_ = !this->joint_test_mode_;
        this->joint_test_inited_ = false;
        this->joint_test_amp_rad_ = 0.0;
        this->joint_test_freq_hz_ = 0.25;
        this->joint_test_wave_mode_ = JointTestWaveMode::Hold;
        this->joint_test_hold_positive_ = true;
        this->joint_test_hold_others_ = false;
        this->joint_test_use_cmd_base_ = false;
        this->joint_test_kp_ = 10.0;
        this->joint_test_kd_ = 0.30;
        this->joint_test_kp_other_ = 10.0;
        this->joint_test_kd_other_ = 0.30;

        std::cout << std::endl
                  << LOGGER::WARNING
                  << "JOINT TEST mode: " << (this->joint_test_mode_ ? "ON" : "OFF") << std::endl;
        if (this->joint_test_mode_)
        {
            std::cout << LOGGER::NOTE
                      << "Put robot on a stand (feet off ground). Keys: J/L select hw(0..11), 0-9 direct, O=10 P=11, I/K amp, H wave(hold/square/sine), X +/- (hold), G baseline(cmd/state), V hold-other joints, U/M kp +/- , Y/N kd +/- , Space amp=0, T exit."
                      << std::endl;

            // Quick mapping hint (DOF order -> hardware index).
            if (!this->params.joint_mapping.empty())
            {
                std::cout << LOGGER::INFO << "[JOINT_TEST] dof->hw mapping:" << std::endl;
                const int n = std::min<int>(this->params.num_of_dofs, static_cast<int>(this->params.joint_mapping.size()));
                for (int i = 0; i < n; ++i)
                {
                    const std::string name = (i < static_cast<int>(this->params.joint_names.size())) ? this->params.joint_names[i] : ("dof_" + std::to_string(i));
                    std::cout << LOGGER::INFO << "  dof=" << i << " name=" << name << " -> hw=" << this->params.joint_mapping[i] << std::endl;
                }
            }
        }
        this->control.current_keyboard = Input::Keyboard::None;
        this->control.current_gamepad = Input::Gamepad::None;
    }

    if (this->joint_test_mode_)
    {
        // Update state (from SDK), then send a visible position perturbation to one hardware joint index.
        this->GetState(&this->robot_state);

        // Direct index select:
        // 0..9 -> hw_idx 0..9, O -> 10, P -> 11.
        if (this->control.current_keyboard >= Input::Keyboard::Num0 &&
            this->control.current_keyboard <= Input::Keyboard::Num9)
        {
            const int idx = static_cast<int>(this->control.current_keyboard) - static_cast<int>(Input::Keyboard::Num0);
            this->joint_test_hw_idx_ = std::max(0, std::min(11, idx));
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::O)
        {
            this->joint_test_hw_idx_ = 10;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::P)
        {
            this->joint_test_hw_idx_ = 11;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }

        if (this->control.current_keyboard == Input::Keyboard::J)
        {
            this->joint_test_hw_idx_ = (this->joint_test_hw_idx_ + 11) % 12;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::L)
        {
            this->joint_test_hw_idx_ = (this->joint_test_hw_idx_ + 1) % 12;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::I)
        {
            this->joint_test_amp_rad_ = std::min(1.2, this->joint_test_amp_rad_ + 0.05);
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::K)
        {
            this->joint_test_amp_rad_ = std::max(0.0, this->joint_test_amp_rad_ - 0.05);
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::H)
        {
            if (this->joint_test_wave_mode_ == JointTestWaveMode::Hold) this->joint_test_wave_mode_ = JointTestWaveMode::Square;
            else if (this->joint_test_wave_mode_ == JointTestWaveMode::Square) this->joint_test_wave_mode_ = JointTestWaveMode::Sine;
            else this->joint_test_wave_mode_ = JointTestWaveMode::Hold;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::X)
        {
            this->joint_test_hold_positive_ = !this->joint_test_hold_positive_;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::G)
        {
            this->joint_test_use_cmd_base_ = !this->joint_test_use_cmd_base_;
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO
                      << "[JOINT_TEST] baseline=" << (this->joint_test_use_cmd_base_ ? "cmd" : "state")
                      << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::V)
        {
            this->joint_test_hold_others_ = !this->joint_test_hold_others_;
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO
                      << "[JOINT_TEST] hold_other_joints=" << (this->joint_test_hold_others_ ? "ON" : "OFF")
                      << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::U)
        {
            this->joint_test_kp_ = std::min(120.0, this->joint_test_kp_ + 2.0);
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO << "[JOINT_TEST] kp=" << this->joint_test_kp_ << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::M)
        {
            this->joint_test_kp_ = std::max(0.0, this->joint_test_kp_ - 2.0);
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO << "[JOINT_TEST] kp=" << this->joint_test_kp_ << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::Y)
        {
            this->joint_test_kd_ = std::min(5.0, this->joint_test_kd_ + 0.05);
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO << "[JOINT_TEST] kd=" << this->joint_test_kd_ << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::N)
        {
            this->joint_test_kd_ = std::max(0.0, this->joint_test_kd_ - 0.05);
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO << "[JOINT_TEST] kd=" << this->joint_test_kd_ << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::R)
        {
            this->joint_test_inited_ = false;
            std::cout << LOGGER::INFO << "[JOINT_TEST] relatch baseline" << std::endl;
            this->control.current_keyboard = Input::Keyboard::None;
        }
        if (this->control.current_keyboard == Input::Keyboard::Space)
        {
            this->joint_test_amp_rad_ = 0.0;
            this->joint_test_inited_ = false;
            this->control.current_keyboard = Input::Keyboard::None;
        }

        // Optional: use DPad for convenience (Retroid).
        if (this->control.current_gamepad == Input::Gamepad::DPadLeft)
        {
            this->joint_test_hw_idx_ = (this->joint_test_hw_idx_ + 11) % 12;
            this->joint_test_inited_ = false;
            this->control.current_gamepad = Input::Gamepad::None;
        }
        if (this->control.current_gamepad == Input::Gamepad::DPadRight)
        {
            this->joint_test_hw_idx_ = (this->joint_test_hw_idx_ + 1) % 12;
            this->joint_test_inited_ = false;
            this->control.current_gamepad = Input::Gamepad::None;
        }
        if (this->control.current_gamepad == Input::Gamepad::DPadUp)
        {
            this->joint_test_amp_rad_ = std::min(1.2, this->joint_test_amp_rad_ + 0.05);
            this->joint_test_inited_ = false;
            this->control.current_gamepad = Input::Gamepad::None;
        }
        if (this->control.current_gamepad == Input::Gamepad::DPadDown)
        {
            this->joint_test_amp_rad_ = std::max(0.0, this->joint_test_amp_rad_ - 0.05);
            this->joint_test_inited_ = false;
            this->control.current_gamepad = Input::Gamepad::None;
        }

        this->RunJointOrderTest();
        return;
    }

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0;
        this->control.y = 0;
        this->control.yaw = 0;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);
}

void RL_Real::RunJointOrderTest()
{
    if (!this->sender_ || !this->robot_data_)
    {
        return;
    }

    const int idx = std::max(0, std::min(11, this->joint_test_hw_idx_));
    if (!this->joint_test_inited_)
    {
        for (int i = 0; i < 12; ++i)
        {
            if (this->joint_test_use_cmd_base_)
            {
                // Use last sent command (hardware order) to avoid sudden jumps when entering test mode.
                this->joint_test_q0_[i] = static_cast<double>(this->robot_joint_cmd_.joint_cmd[i].position);
            }
            else
            {
                // Use measured state (hardware order).
                this->joint_test_q0_[i] = static_cast<double>(this->robot_data_->joint_data.joint_data[i].position);
            }
        }
        this->joint_test_t0_ = std::chrono::steady_clock::now();
        this->joint_test_inited_ = true;

        const double base_cmd = static_cast<double>(this->robot_joint_cmd_.joint_cmd[idx].position);
        const double base_state = static_cast<double>(this->robot_data_->joint_data.joint_data[idx].position);
        const double base_diff = base_state - base_cmd;
        const auto WaveName = [this]() -> const char*
        {
            switch (this->joint_test_wave_mode_)
            {
            case JointTestWaveMode::Hold: return "hold";
            case JointTestWaveMode::Square: return "square";
            case JointTestWaveMode::Sine: return "sine";
            }
            return "na";
        };

        std::cout << LOGGER::INFO
                  << "[JOINT_TEST] hw_idx=" << idx
                  << " (" << LegNameFromIdx(idx) << "_" << JointNameFromIdx(idx) << ")"
                  << " base=" << this->joint_test_q0_[idx]
                  << " (cmd=" << base_cmd << " state=" << base_state << " diff=" << base_diff << ")"
                  << " amp(rad)=" << this->joint_test_amp_rad_
                  << " freq(Hz)=" << this->joint_test_freq_hz_
                  << " kp=" << this->joint_test_kp_
                  << " kd=" << this->joint_test_kd_
                  << " wave=" << WaveName()
                  << " hold_other=" << (this->joint_test_hold_others_ ? "on" : "off")
                  << " baseline=" << (this->joint_test_use_cmd_base_ ? "cmd" : "state")
                  << std::endl;
    }

    const auto now = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(now - this->joint_test_t0_).count();
    double delta = 0.0;
    const double s = std::sin(2.0 * M_PI * this->joint_test_freq_hz_ * t);
    if (this->joint_test_wave_mode_ == JointTestWaveMode::Hold)
    {
        delta = this->joint_test_amp_rad_ * (this->joint_test_hold_positive_ ? 1.0 : -1.0);
    }
    else if (this->joint_test_wave_mode_ == JointTestWaveMode::Square)
    {
        delta = this->joint_test_amp_rad_ * (s >= 0.0 ? 1.0 : -1.0);
    }
    else
    {
        delta = this->joint_test_amp_rad_ * s;
    }

    RobotCmd cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    for (int i = 0; i < 12; ++i)
    {
        const double target = this->joint_test_q0_[i] + ((i == idx) ? delta : 0.0);
        cmd.joint_cmd[i].position = static_cast<float>(target);
        cmd.joint_cmd[i].velocity = 0.0f;
        cmd.joint_cmd[i].torque = 0.0f;
        if (i == idx)
        {
            cmd.joint_cmd[i].kp = static_cast<float>(this->joint_test_kp_);
            cmd.joint_cmd[i].kd = static_cast<float>(this->joint_test_kd_);
        }
        else if (this->joint_test_hold_others_)
        {
            cmd.joint_cmd[i].kp = static_cast<float>(this->joint_test_kp_other_);
            cmd.joint_cmd[i].kd = static_cast<float>(this->joint_test_kd_other_);
        }
        else
        {
            // Keep non-selected joints in a passive-like mode to avoid global jumps.
            cmd.joint_cmd[i].kp = 0.0f;
            cmd.joint_cmd[i].kd = 2.5f;
        }
    }
    this->robot_joint_cmd_ = cmd;
    this->sender_->SendCmd(cmd);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        if (this->control.navigation_mode)
        {
#if !defined(USE_CMAKE) && defined(USE_ROS)
            this->obs.commands = torch::tensor({{this->cmd_vel.linear.x, this->cmd_vel.linear.y, this->cmd_vel.angular.z}});
#endif
        }
        else
        {
            this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        }
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Real::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->robot_data_->joint_data.joint_data[this->params.joint_mapping[i]].position);
        this->plot_target_joint_pos[i].push_back(this->robot_joint_cmd_.joint_cmd[this->params.joint_mapping[i]].position);
        plt::subplot(this->params.num_of_dofs, 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

void RL_Real::UDPRecv()
{
    if (receiver_)
    {
        robot_data_ = &(receiver_->GetState());
    }
}

void RL_Real::EulerToQuaternion(float roll, float pitch, float yaw, float q[4])
{
    roll *= M_PI / 180.0f;
    pitch *= M_PI / 180.0f;
    yaw *= M_PI / 180.0f;

    float cr = cos(roll * 0.5f);
    float sr = sin(roll * 0.5f);
    float cp = cos(pitch * 0.5f);
    float sp = sin(pitch * 0.5f);
    float cy = cos(yaw * 0.5f);
    float sy = sin(yaw * 0.5f);

    q[0] = cr * cp * cy + sr * sp * sy;  // w
    q[1] = sr * cp * cy - cr * sp * sy;  // x
    q[2] = cr * sp * cy + sr * cp * sy;  // y
    q[3] = cr * cp * sy - sr * sp * cy;  // z
}


#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar;
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RL_Real>());
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    RL_Real rl_sar;
    while (1) { sleep(10); }
#endif
    return 0;
}
