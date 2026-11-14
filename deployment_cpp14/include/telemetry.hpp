#pragma once
#include <cstdint>
#include <string>
namespace deployment {
constexpr double pi=3.14159265358979323846;
struct Telemetry {
    std::uint32_t sequence,tx_ms,enc_sequence,enc_ms,imu_sequence,imu_ms;
    std::int32_t total_ticks;
    int delta_ticks,servo_deg,pwm;
    double speed,ax,ay,gyro_dps,yaw_deg;
    unsigned calibration,flags;
    unsigned protocol_version=1;
    double gyro_x_dps=0,gyro_y_dps=0,roll_deg=0,pitch_deg=0;
    std::uint32_t read_start_ms=0,read_end_ms=0,enc_interval_ms=0;
    unsigned system_status=0,system_error=0;
};
std::uint16_t crc16(const std::string& text);
bool parse(const std::string& line,Telemetry& packet,std::string& error);
struct Pose { double x=0,y=0,yaw=0,vx=0,r=0,travel=0; };
struct Config {
    double x0=0,y0=0,yaw0=0;
    double metres_per_count=2*pi*.0325/(11*4*19*(38./13.));
    bool imu_clockwise=true;
    // 0 preserves legacy heading-linked conversion; +/-1 overrides raw gyro Z.
    int gyro_sign=0;
    double max_speed=.8;
    std::uint32_t max_age_ms=100,max_skew_ms=50,max_gap_ms=200;
};
class Odometry {
    Config cfg_;bool initialized_=false,fault_=false;
    Telemetry last_{};double imu_yaw_=0;Pose pose_;
public:
    explicit Odometry(Config config=Config());
    bool update(const Telemetry& packet,std::string& status);
    const Pose& pose() const {return pose_;}
    bool faulted() const {return fault_;}
};
}
