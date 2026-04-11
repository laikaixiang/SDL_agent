#pragma once
#ifdef BUILD_DLL
#define API_SYMBOL __declspec(dllexport)
#else
#define API_SYMBOL __declspec(dllimport)
#endif
#include <cstdint>

/*
	返回值 : 1 : 指令成功下发或操作成功, -1 : 串口发送失败, -2 : 串口返回超时, -3 : 串口返回错误
*/
extern "C" {
	API_SYMBOL int OpenComPort(uint8_t index, uint32_t baudRate);// 开启指定端口号com口
	API_SYMBOL int CloseComPort(uint8_t index);//关闭指定端口号com口
	API_SYMBOL HANDLE GetComPortHandle(uint8_t index);


    //CSTEPC电机设备========================================================================================
    API_SYMBOL int Motor_Absolute_movement_c(uint8_t index, uint8_t id, int32_t Moveposition);//D 电机绝对值运动
    API_SYMBOL int Read_Motor_position_c(uint8_t index, uint8_t id, int32_t * position_value);//E 读取编码器的位置
    API_SYMBOL int Read_software_version_c(uint8_t index, uint8_t id, int8_t software_version_value[15]);//A 读取软件版本
    API_SYMBOL int Enable_motor_c(uint8_t index, uint8_t id, uint8_t select_value);//a 释放或励磁电机
    API_SYMBOL int Set_speed_ac_de_time_c(uint8_t index, uint8_t id, uint32_t speed, uint8_t add_time, uint8_t cut_time);// B  设置 运动 速度及加减速时间
    API_SYMBOL int Set_current_c(uint8_t index, uint8_t id, uint8_t moment_current, uint8_t run_current, uint8_t keep_current);//b 设置电流
    API_SYMBOL int Set_motor_parameter_c(uint8_t index, uint8_t id, uint16_t motor_parameter, int32_t origin_maxdistance);//C 设置电机参数及回原点最大距离
    API_SYMBOL int Set_origin_current_c(uint8_t index, uint8_t id, uint8_t origin_current);//c  设置初始化回原点电流
    API_SYMBOL int Find_status_c(uint8_t index, uint8_t id, int8_t * status);//d 读取是否运动到位
    API_SYMBOL int Set_software_origin_c(uint8_t index, uint8_t id);//f 设置当前点为软件原点
    API_SYMBOL int Zero_c(uint8_t index, uint8_t id);// G 电机初始化寻找原点
    API_SYMBOL int Find_zero_c(uint8_t index, uint8_t id, int8_t * zero_status);//g 读取初始化是否完成
    API_SYMBOL int Set_Liquid_detect_max_distance_c(uint8_t index, uint8_t id, int32_t maxdistance);//H 设置液面探测的最大距离
    API_SYMBOL int Motor_relative_movement_c(uint8_t index, uint8_t id, int32_t Moveposition, uint16_t rate);//h 电机相对值运动
    API_SYMBOL int Motor_movement_c(uint8_t index, uint8_t id, uint8_t select);//I 电机力矩运动 
    API_SYMBOL int Motor_absolute_movement_group_c(uint8_t index, uint8_t id, int32_t select);//i 电机绝对值与力矩组合运动
    API_SYMBOL int Motor_absolute_Liquiddetect_group_c(uint8_t index, uint8_t id, int32_t detect_depth, int32_t highspeed_distance);// j 电机绝对值与液面探测组合运动
    API_SYMBOL int Motor_stop_positive_inversion_c(uint8_t index, uint8_t id, uint8_t select);// K 电机JOG 正反转与急停
    API_SYMBOL int Read_sensor_status_c(uint8_t index, uint8_t id, int16_t * status);// L 读取外部传感器状态
    API_SYMBOL int Read_motor_runspeed_c(uint8_t index, uint8_t id, int32_t * runspeed);
    API_SYMBOL int Read_ac_de_time_current_c(uint8_t index, uint8_t id, int8_t * ad_time, int8_t * de_time, int8_t * moment_current, int8_t * run_current, int8_t * keep_current);//N 读取电机的加减速时间与电流
    API_SYMBOL int Read_motor_parameter_maxdistance_c(uint8_t index, uint8_t id, int16_t * motor_parameter, int32_t * origin_maxdistance);//O 读取电机参数和回原点的最大距离
    API_SYMBOL int Read_Liquid_detect_maxdistance_c(uint8_t index, uint8_t id, int32_t * maxdistance); //P  读取液面检测的最大距离
    API_SYMBOL int Set_origin_offset_distance_c(uint8_t index, uint8_t id, int32_t offsetdistance);//R 设置回原点的偏移距离 
    API_SYMBOL int Read_origin_offset_distance_c(uint8_t index, uint8_t id, int32_t * offsetdistance);//r 读取回原点的偏移距离
    API_SYMBOL int Save_c(uint8_t index, uint8_t id);//U 保存配置所有参数
    API_SYMBOL int change_motor_address_c(uint8_t index, uint8_t id, int8_t change_id);//T  变更电机地址
    API_SYMBOL int Read_origin_offset_parameter_c(uint8_t index, uint8_t id, int8_t * origin_current, int8_t * keep_current, int8_t * add_time, int8_t * dec_time, int32_t * origin_speed);//w 读取回原点偏移运动参数
    API_SYMBOL int Set_origin_threshold_value_c(uint8_t index, uint8_t id, int16_t threshold_value);//y 设置回原点参考阈值

    API_SYMBOL int Set_absolute_moment_maxprotectdistance_c(uint8_t index, uint8_t id, int32_t maxprotectdistance);// + 设置绝对值与力矩组合运动的最大保护距离
    API_SYMBOL int Read_absolute_moment_maxprotectdistance_c(uint8_t index, uint8_t id, int32_t * maxprotectdistance);// - 读取绝对值与力矩组合运动的最大保护距离
    API_SYMBOL int Set_absolute_Liquiddetect_speed_c(uint8_t index, uint8_t id, int16_t speed);//* 设置绝对值与液面探测组合运动的探测速度
    API_SYMBOL int Read_absolute_Liquiddetect_speed_c(uint8_t index, uint8_t id, int16_t * speed);//  / 读取绝对值与液面探测组合运动的探测速度

    API_SYMBOL int Set_motor_all_parameter_c(uint8_t index,
        uint8_t id,
        int16_t motor_parameter,
        int32_t origin_maxdistance,
        int8_t add_time,
        int8_t dec_time,
        int8_t moment_current,
        int8_t run_current,
        int8_t keep_current,
        int32_t runspeed,
        int32_t Liquid_maxdistance,
        int8_t origin_current,
        int16_t origin_speed,
        int32_t offsetdistance,
        int8_t origin_offset_run_current,
        int8_t origin_offset_keep_current,
        int8_t offset_addtime,
        int8_t offset_dectime,
        int32_t offset_speed,
        int16_t origin_threshold_value,
        int32_t absolute_moment_maxdistance,
        int8_t absolute_threshold_value,
        int16_t absolute_Liquiddetect_detectspeed);//X 设置驱动器所有参数

    API_SYMBOL int Read_motor_all_parameter_c(uint8_t index,
        uint8_t id,
        int16_t * motor_parameter,
        int32_t * origin_maxdistance,
        int8_t * add_time,
        int8_t * dec_time,
        int8_t * moment_current,
        int8_t * run_current,
        int8_t * keep_current,
        int32_t * runspeed,
        int32_t * Liquid_maxdistance,
        int32_t * motor_position,
        int8_t * origin_current,
        int16_t * origin_speed,
        int32_t * offsetdistance,
        int8_t * origin_offset_run_current,
        int8_t * origin_offset_keep_current,
        int8_t * offset_addtime,
        int8_t * offset_dectime,
        int32_t * offset_speed,
        int16_t * origin_threshold_value,
        int32_t * absolute_moment_maxdistance,
        int8_t * absolute_threshold_value,
        int16_t * absolute_Liquiddetect_detectspeed);// Y  读取驱动器所有参数

    //RJ28电机设备====================================================================================================
    API_SYMBOL int Electric_claw_initialized_rj(uint8_t index, uint8_t id);//电爪初始化
    API_SYMBOL int Electric_claw_spin_rj(uint8_t index, uint8_t id, int16_t pos); //运行旋转并设定角度位置
    API_SYMBOL int Find_set_electric_claw_spin_rj(uint8_t index, uint8_t id, int16_t* pos); //查询设定角度位置
    API_SYMBOL int Electric_rotation_initialized_rj(uint8_t index, uint8_t id, uint16_t select);//旋转初始化
    API_SYMBOL int Electric_stop_rj(uint8_t index, uint8_t id);//设备急停
    API_SYMBOL int Find_electric_stop_rj(uint8_t index, uint8_t id, uint16_t* status);//查询紧急停止
    API_SYMBOL int Electric_motoring_torque_rj(uint8_t index, uint8_t id, uint16_t motoring_torque);//电爪的运动力矩
    API_SYMBOL int Find_electric_motoring_torque_rj(uint8_t index, uint8_t id, uint16_t* motoring_torque);//查询电爪的运动力矩
    API_SYMBOL int Electric_motoring_speed_rj(uint8_t index, uint8_t id, uint16_t motoring_speed);//电爪的运动速度
    API_SYMBOL int Find_electric_motoring_speed_rj(uint8_t index, uint8_t id, uint16_t* motoring_speed);//查询电爪的设定运动速度
    API_SYMBOL int Electric_claw_run_rj(uint8_t index, uint8_t id, uint16_t pos);//运行电爪并配置位置
    API_SYMBOL int Find_lectric_claw_run_rj(uint8_t index, uint8_t id, uint16_t* pos);//查询电爪的设定位置
    API_SYMBOL int Electric_claw_moment_rj(uint8_t index, uint8_t id, uint16_t moment);//旋转的力矩
    API_SYMBOL int Find_electric_claw_moment_rj(uint8_t index, uint8_t id, uint16_t* moment);//查询旋转的运动力矩
    API_SYMBOL int Electric_claw_speed_rj(uint8_t index, uint8_t id, uint16_t speed);//旋转的速度
    API_SYMBOL int Find_electric_claw_speed_rj(uint8_t index, uint8_t id, uint16_t* speed);//查询旋转的运动速度
    API_SYMBOL int Find_electric_claw_initialized_status_rj(uint8_t index, uint8_t id, uint16_t* status);//查询电爪初始化状态
    API_SYMBOL int Find_spin_initialized_status_rj(uint8_t index, uint8_t id, uint16_t* status);//查询旋转初始化状态
    API_SYMBOL int Find_electric_claw_status_rj(uint8_t index, uint8_t id, uint16_t* status);//查询电爪运行状态
    API_SYMBOL int Find_spin_status_rj(uint8_t index, uint8_t id, uint16_t* status);//查询旋转运行状态
    API_SYMBOL int Find_claw_pos_rj(uint8_t index, uint8_t id, uint16_t* status);//查询电爪实时位置坐标
    API_SYMBOL int Find_spin_pos_rj(uint8_t index, uint8_t id, int16_t* pos);//查询旋转实时位置角度
    API_SYMBOL int Electric_initialized_direction_rj(uint8_t index, uint8_t id, uint16_t select);//电爪初始化方向
    API_SYMBOL int Find_electric_initialized_direction_rj(uint8_t index, uint8_t id, int16_t* direction);//查询电爪初始化方向
    API_SYMBOL int Electric_claw_spin_direction_rj(uint8_t index, uint8_t id, uint16_t select);//旋转初始化方向
    API_SYMBOL int Find_electric_claw_spin_direction_rj(uint8_t index, uint8_t id, int16_t* direction);//查询旋转初始化方向
    API_SYMBOL int Save_data_rj(uint8_t index, uint8_t id);//保存数据
    API_SYMBOL int Find_save_data_rj(uint8_t index, uint8_t id, int16_t* status);//查询电爪初始化方向
    API_SYMBOL int Change_id_rj(uint8_t index, uint8_t id, uint16_t change_id);//变更设备ID
    API_SYMBOL int Enable_electric_claw_rj(uint8_t index, uint8_t id, uint16_t select);//释放使能电爪
    API_SYMBOL int Find_enable_electric_claw_rj(uint8_t index, uint8_t id, uint16_t* status);//查询电爪励磁状态
    API_SYMBOL int Enable_electric_claw_spin_rj(uint8_t index, uint8_t id, uint16_t select);//释放使能旋转
    API_SYMBOL int Find_enable_electric_claw_spin_rj(uint8_t index, uint8_t id, uint16_t* status);//查询释放使能旋转

    //CPUMP电机设备==========================================================================================
    API_SYMBOL int Read_software_version_p(uint8_t index, uint8_t id, int8_t software_version_value[18]);// A 读取软件版本
    API_SYMBOL int Set_spitspeed_p(uint8_t index, uint8_t id, uint16_t speed);//B    设置吐液速度
    API_SYMBOL int Read_spitspeed_p(uint8_t index, uint8_t id, int16_t* speed);// b 读取吐液速度
    API_SYMBOL int Set_capacitance_detection_threshold_p(uint8_t index, uint8_t id, int16_t value);// C   电容探测阈值
    API_SYMBOL int Read_capacitance_detection_threshold_p(uint8_t index, uint8_t id, int16_t* value);//c 读取电容探测阈值
    API_SYMBOL int Find_status_p(uint8_t index, uint8_t id, int8_t* status);// d 读取是否运动到位
    API_SYMBOL int Find_Used_and_remaining_volume_p(uint8_t index, uint8_t id, int32_t* used_volume, int32_t* remaining_volume);//E 查询已使用容积和剩余容积
    API_SYMBOL int Suck_spit_mix_p(uint8_t index, uint8_t id, int16_t mix_ul, int16_t mix_cnt, int8_t* return_value);// F 吸吐混匀动作指令
    API_SYMBOL int Find_mix_remain_cnt_p(uint8_t index, uint8_t id, int16_t* remain_cnt); //f 查询混匀动作剩余次数
    API_SYMBOL int Cpump_zero_p(uint8_t index, uint8_t id);//G 空气泵初始化
    API_SYMBOL int Find_zero_p_p(uint8_t index, uint8_t id, int8_t* zero_status); //g 读取初始化是否完成
    API_SYMBOL int Set_Blocking_vomiting_vacuuming_detectionthreshold_p(uint8_t index, uint8_t id, int16_t tip_kind, int8_t filter_element, int16_t Blocking_value, int16_t vomiting_value, int16_t vacuuming_value, int16_t detection_value);// H 设置吸堵/ 吐堵/ 吸空/探测阈值
    API_SYMBOL int Read_Blocking_vomiting_vacuuming_detectionthreshold_p(uint8_t index, uint8_t id, int16_t tip_kind, int8_t filter_element, int16_t* Blocking_value, int16_t* vomiting_value, int16_t* vacuuming_value, int16_t* detection_value);// h 读取吸堵/ 吐堵/ 吸空/探测阈值
    API_SYMBOL int Pressure_monitor_switch_p(uint8_t index, uint8_t id, int8_t suck_vomiting_block_switch, int8_t suck_air_switch);//I 气压监测开关
    API_SYMBOL int Find_pressure_monitor_switch_p(uint8_t index, uint8_t id, int8_t* suck_vomiting_block_switch, int8_t* suck_air_switch);// i 查询气压监测状态
    API_SYMBOL int Set_message_p(uint8_t index, uint8_t id, int16_t first_suck, int16_t back_tip, int16_t spit_off);//J 设置首次回吸值，退 TIP  值，吐液切断
    API_SYMBOL int Read_message_p(uint8_t index, uint8_t id, int16_t* first_suck, int16_t* back_tip, int16_t* spit_off);//j 读取首次回吸值，退 TIP  值，吐液切断
    API_SYMBOL int First_suck_p(uint8_t index, uint8_t id, int8_t* return_value);//M 首次回吸空气柱动作指令
    API_SYMBOL int Pressure_detect_water_p(uint8_t index, uint8_t id, int8_t on_off);//N  气压探测液位动作指令
    API_SYMBOL int Suck_p(uint8_t index, uint8_t id, int16_t suck_ul, int8_t* return_value);//n  吸液动作指令
    API_SYMBOL int Set_tip_p(uint8_t index, uint8_t id, int16_t tip, int8_t filter);//O  设置 TIP 规格
    API_SYMBOL int Read_tip_p(uint8_t index, uint8_t id, int16_t* tip, int8_t* filter); //o  读取 TIP 规格
    API_SYMBOL int Second_suck_p(uint8_t index, uint8_t id, int8_t* return_value);//P  二次回吸空气柱动作指令
    API_SYMBOL int Spit_p(uint8_t index, uint8_t id, int16_t spit_ul, int8_t* return_value);//p  吐液动作指令
    API_SYMBOL int Back_tip_p(uint8_t index, uint8_t id);//Q  退掉tip
    API_SYMBOL int Find_tip_p(uint8_t index, uint8_t id, int8_t* return_value);//q 查询 TIP 有无
    API_SYMBOL int Set_back_compensation_value_p(uint8_t index, uint8_t id, int16_t return_value);//R   设置回程差补偿值
    API_SYMBOL int Read_back_compensation_value_p(uint8_t index, uint8_t id, int16_t* return_value);//  r 读取回程差补偿值
    API_SYMBOL int change_motor_address_p(uint8_t index, uint8_t id, int8_t change_id);//T  变更电机地址
    API_SYMBOL int Save_p(uint8_t index, uint8_t id, int8_t select);// U  保存配置所有参数
    API_SYMBOL int Set_reset_origin_speed_p(uint8_t index, uint8_t id, int16_t speed);// V  设置复位回原点速度
    API_SYMBOL int Read_reset_origin_speed_p(uint8_t index, uint8_t id, int16_t* speed);// v 读取复位回原点速度
    API_SYMBOL int Set_current_p(uint8_t index, uint8_t id, int16_t current);//W  设置电流
    API_SYMBOL int Read_current_p(uint8_t index, uint8_t id, int16_t* current);// w  读取电流
    API_SYMBOL int reset_p(uint8_t index, uint8_t id);// =  设备重启
    API_SYMBOL int Set_cutspeed_p(uint8_t index, uint8_t id, int16_t speed);//2  设置切断速度
    API_SYMBOL int Read_cutspeed_p(uint8_t index, uint8_t id, int16_t* speed);// 3  读取切断速度
    API_SYMBOL int Set_suckspeed_p(uint8_t index, uint8_t id, int16_t speed);//4  设置吸液速度
    API_SYMBOL int Read_suckspeed_p(uint8_t index, uint8_t id, int16_t* speed);// 5  读取吸液速度
    API_SYMBOL int Read_pressure_value_p(uint8_t index, uint8_t id, int16_t* value);// 6  读取当前气压值

    API_SYMBOL int Set_suck_spit_six_p(uint8_t index,
        uint8_t id,
        int16_t tip_kind,
        int8_t  filter,
        int8_t  suck_spit,
        int32_t target1,
        int32_t compensate1,
        int32_t target2,
        int32_t compensate2,
        int32_t target3,
        int32_t compensate3,
        int32_t target4,
        int32_t compensate4,
        int32_t target5,
        int32_t compensate5,
        int32_t target6,
        int32_t compensate6);//K设置 6 段吸液吐液补偿值

    API_SYMBOL int Read_suck_spit_six_p(uint8_t index,
        uint8_t id,
        int16_t tip_kind,
        int8_t filter,
        int8_t suck_spit,
        int32_t* target1,
        int32_t* compensate1,
        int32_t* target2,
        int32_t* compensate2,
        int32_t* target3,
        int32_t* compensate3,
        int32_t* target4,
        int32_t* compensate4,
        int32_t* target5,
        int32_t* compensate5,
        int32_t* target6,
        int32_t* compensate6);//k读取吸液吐液补偿值
}

