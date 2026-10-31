#include "telemetry.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace deployment {
std::uint16_t crc16(const std::string& text) {
    std::uint16_t crc=0xffff;
    for(unsigned char ch:text){crc^=static_cast<std::uint16_t>(ch)<<8;
        for(int b=0;b<8;++b)crc=static_cast<std::uint16_t>((crc&0x8000)?(crc<<1)^0x1021:crc<<1);}
    return crc;
}
namespace {
long long integer(const std::string& s,long long lo,long long hi,int base=10){
    if(s.empty()||s.find_first_not_of(base==16?"0123456789abcdefABCDEF":"-0123456789")!=std::string::npos)throw std::runtime_error("integer syntax");
    std::size_t used=0;auto v=std::stoll(s,&used,base);
    if(used!=s.size()||v<lo||v>hi)throw std::runtime_error("integer range");
    return v;
}
double real(const std::string& s){std::size_t used=0;auto v=std::stod(s,&used);
    if(used!=s.size()||!std::isfinite(v))throw std::runtime_error("invalid finite number");
    return v;}
double wrap(double v){return std::atan2(std::sin(v),std::cos(v));}
bool newer(std::uint32_t a,std::uint32_t b){auto d=a-b;return d>0&&d<0x80000000U;}
std::int64_t tick_delta(std::int32_t a,std::int32_t b){auto u=static_cast<std::uint32_t>(a)-static_cast<std::uint32_t>(b);
    return u<=0x7fffffffU?static_cast<std::int64_t>(u):static_cast<std::int64_t>(u)-4294967296LL;}
}
bool parse(const std::string& input,Telemetry& output,std::string& error){
    try {
        std::string s=input;if(!s.empty()&&s.back()=='\n')s.pop_back();if(!s.empty()&&s.back()=='\r')s.pop_back();
        if(s.size()>512)throw std::runtime_error("frame too long");
        auto split=s.rfind(',');if(split==std::string::npos||s.size()-split-1!=4)throw std::runtime_error("CRC field");
        if(crc16(s.substr(0,split))!=integer(s.substr(split+1),0,65535,16))throw std::runtime_error("CRC mismatch");
        std::vector<std::string> f;std::size_t from=0;
        for(std::size_t i=0;i<=s.size();++i)if(i==s.size()||s[i]==','){f.push_back(s.substr(from,i-from));from=i+1;}
        if(f.size()!=20||f[0]!="TEL"||f[1]!="1")throw std::runtime_error("unsupported TEL format/version");
        auto u=[&](int i){return static_cast<std::uint32_t>(integer(f[i],0,4294967295LL));};
        Telemetry p{};p.sequence=u(2);p.tx_ms=u(3);p.enc_sequence=u(4);p.enc_ms=u(5);
        p.total_ticks=static_cast<std::int32_t>(integer(f[6],-2147483648LL,2147483647LL));
        p.delta_ticks=static_cast<int>(integer(f[7],-32768,32767));p.speed=real(f[8]);
        p.servo_deg=static_cast<int>(integer(f[9],-2147483648LL,2147483647LL));p.pwm=static_cast<int>(integer(f[10],-2147483648LL,2147483647LL));
        p.imu_sequence=u(11);p.imu_ms=u(12);p.ax=real(f[13]);p.ay=real(f[14]);p.gyro_dps=real(f[15]);p.yaw_deg=real(f[16]);
        p.calibration=static_cast<unsigned>(integer(f[17],0,255,16));p.flags=static_cast<unsigned>(integer(f[18],0,255));
        output=p;error.clear();return true;
    }catch(const std::exception& ex){error=ex.what();return false;}
}
Odometry::Odometry(Config c):cfg_(c){
    if(!std::isfinite(c.x0)||!std::isfinite(c.y0)||!std::isfinite(c.yaw0)||!std::isfinite(c.metres_per_count)||c.metres_per_count<=0||!std::isfinite(c.max_speed)||c.max_speed<=0||c.max_gap_ms==0)throw std::runtime_error("invalid odometry configuration");
    pose_.x=c.x0;pose_.y=c.y0;pose_.yaw=c.yaw0;
}
bool Odometry::update(const Telemetry& p,std::string& status){
    if(fault_){status="fault_latched_restart_required";return false;}
    if((p.flags&3)!=3){status="sensor_invalid";return false;}
    if(p.tx_ms-p.enc_ms>cfg_.max_age_ms||p.tx_ms-p.imu_ms>cfg_.max_age_ms){status="sensor_stale";return false;}
    auto skew=p.enc_ms-p.imu_ms;auto reverse=p.imu_ms-p.enc_ms;
    if(skew>cfg_.max_skew_ms&&reverse>cfg_.max_skew_ms){status="sensor_skew";return false;}
    const double direction=cfg_.imu_clockwise?-1.:1.;
    double imu=p.yaw_deg*pi/180.;
    if(!initialized_){initialized_=true;last_=p;imu_yaw_=imu;pose_.vx=p.speed;pose_.r=direction*p.gyro_dps*pi/180.;status="initialized";return true;}
    if(!newer(p.sequence,last_.sequence)){status="duplicate_or_reordered_packet";return false;}
    if(p.enc_sequence==last_.enc_sequence){status="duplicate_encoder";return false;}
    if(!newer(p.enc_sequence,last_.enc_sequence)||!newer(p.enc_ms,last_.enc_ms)){
        fault_=true;status="encoder_reset_or_reordered";return false;}
    auto dt=p.enc_ms-last_.enc_ms;
    if(dt>cfg_.max_gap_ms){fault_=true;status="integration_gap_restart_required";return false;}
    if(p.imu_sequence!=last_.imu_sequence&&!newer(p.imu_sequence,last_.imu_sequence)){
        fault_=true;status="imu_reset_or_reordered";return false;}
    const double ds=tick_delta(p.total_ticks,last_.total_ticks)*cfg_.metres_per_count;
    if(std::abs(ds)/(dt*.001)>cfg_.max_speed){fault_=true;status="implausible_encoder_travel";return false;}
    // Unwrap relative increments; retain continuous yaw for Actor physical features.
    const double turn=direction*wrap(imu-imu_yaw_);const double mid=pose_.yaw+turn*.5;
    pose_.x+=ds*std::cos(mid);pose_.y+=ds*std::sin(mid);pose_.yaw+=turn;
    pose_.travel+=ds;pose_.vx=p.speed;pose_.r=direction*p.gyro_dps*pi/180.;
    last_=p;imu_yaw_=imu;status="updated";return true;
}
}
