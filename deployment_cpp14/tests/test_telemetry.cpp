#include "telemetry.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace deployment;
void check(bool ok){if(!ok)throw std::runtime_error("test failed");}
std::string frame(std::string payload){std::ostringstream s;s<<payload<<','<<std::hex<<std::uppercase<<std::setw(4)<<std::setfill('0')<<crc16(payload)<<"\r\n";return s.str();}
int main(){
    check(crc16("123456789")==0x29b1);Telemetry p{};std::string err;
    auto line=frame("TEL,1,1,100,10,100,1000,12,0.1000,75,100,10,100,0.0,0.0,10.0,359.00,FF,3");
    check(parse(line,p,err));check(p.total_ticks==1000&&p.calibration==255);
    auto broken=line;broken[10]='9';check(!parse(broken,p,err));
    check(!parse(frame("TEL,1,1,100,10,100,2147483648,12,0.1,75,0,10,100,0,0,0,0,FF,3"),p,err));
    check(!parse(frame("TEL,1,1,100,10,100,1,12,nan,75,0,10,100,0,0,0,0,FF,3"),p,err));
    check(parse(line,p,err));Odometry o;check(o.update(p,err));check(!o.update(p,err));
    auto q=p;q.sequence++;q.enc_sequence++;q.imu_sequence++;q.tx_ms+=20;q.enc_ms+=20;q.imu_ms+=20;q.total_ticks+=24;q.yaw_deg=1;
    check(o.update(q,err));check(std::abs(o.pose().yaw+2*pi/180)<1e-12);
    check(std::abs(o.pose().travel-24*Config().metres_per_count)<1e-12);
    check(!o.update(p,err));check(!o.faulted());
    auto gap=q;gap.sequence++;gap.enc_sequence++;gap.imu_sequence++;gap.tx_ms+=300;gap.enc_ms+=300;gap.imu_ms+=300;
    check(!o.update(gap,err)&&o.faulted());check(!o.update(q,err));
    Odometry stale;auto invalid=p;invalid.flags=1;check(!stale.update(invalid,err));invalid=p;invalid.tx_ms=300;check(!stale.update(invalid,err));
    Odometry rollover;auto a=p;a.sequence=0xffffffffU;a.enc_sequence=0xffffffffU;a.imu_sequence=0xffffffffU;a.enc_ms=0xfffffff0U;a.imu_ms=a.enc_ms;a.tx_ms=a.enc_ms;a.total_ticks=2147483640;check(rollover.update(a,err));
    auto b=a;b.sequence=0;b.enc_sequence=0;b.imu_sequence=0;b.enc_ms=4;b.imu_ms=4;b.tx_ms=4;b.total_ticks=-2147483632;
    check(rollover.update(b,err));check(std::abs(rollover.pose().travel-24*Config().metres_per_count)<1e-12);
    b.sequence++;b.enc_sequence=1;b.imu_sequence=1;b.enc_ms=24;b.imu_ms=24;b.tx_ms=24;b.total_ticks=2147483640;
    check(rollover.update(b,err));check(std::abs(rollover.pose().travel)<1e-12);
    std::cout<<"Telemetry CRC, parsing, freshness, duplicate, signed counts, wrap and gap tests PASS\n";
}
