#include "telemetry.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <poll.h>
#include <csignal>
#include <cstdlib>
namespace {volatile std::sig_atomic_t stopped=0;void stop(int){stopped=1;}
struct Port {int fd=-1;~Port(){if(fd>=0)::close(fd);}};
double number(const std::string& s){std::size_t i=0;double v=std::stod(s,&i);if(i!=s.size())throw std::runtime_error("invalid number");return v;}
}
int main(int argc,char** argv){try{
    std::string input,device,output,raw_log;double duration=30;deployment::Config config;
    for(int i=1;i<argc;++i){std::string arg=argv[i];
        if(arg=="--imu-ccw"){config.imu_clockwise=false;continue;}
        if(arg=="--help"){std::cout<<"RX only: --input TEL.log | --device /dev/ttyUSB0 [--output pose.csv] [--raw-log TEL.log] [--duration 30] [--x0 0 --y0 0 --yaw0-deg 0] [--imu-ccw]\n";return 0;}
        if(i+1>=argc)throw std::runtime_error("missing option value");
        std::string value=argv[++i];
        if(arg=="--input")input=value;else if(arg=="--device")device=value;
        else if(arg=="--output")output=value;else if(arg=="--raw-log")raw_log=value;
        else if(arg=="--duration")duration=number(value);
        else if(arg=="--x0")config.x0=number(value);else if(arg=="--y0")config.y0=number(value);
        else if(arg=="--yaw0-deg")config.yaw0=number(value)*deployment::pi/180.;
        else throw std::runtime_error("unknown option: "+arg);
    }
    if(input.empty()==device.empty())throw std::runtime_error("select exactly one --input or --device");
    if(!(duration>0&&duration<86400))throw std::runtime_error("duration must be in (0,86400) seconds");
    if((!output.empty()&&(output==input||output==raw_log))||(!raw_log.empty()&&raw_log==input))throw std::runtime_error("input/output paths must differ");
    for(const auto& path:{output,raw_log})if(!path.empty()&&std::ifstream(path).good())throw std::runtime_error("refusing to overwrite: "+path);
    std::ofstream csv,raw;if(!output.empty()){csv.open(output);if(!csv)throw std::runtime_error("cannot open output");}
    if(!raw_log.empty()){raw.open(raw_log,std::ios::binary);if(!raw)throw std::runtime_error("cannot open raw log");}
    auto& out=output.empty()?std::cout:csv;
    out<<"packet_seq,tx_ms,enc_ms,imu_ms,accepted,status,X_est_m,Y_est_m,yaw_est_rad,vx_measured_mps,r_measured_radps,signed_travel_m,enc_total,flags,calib_hex\n"<<std::setprecision(15);
    deployment::Odometry odom(config);std::string buffer;bool discard=false;unsigned good=0,bad=0,rejected=0;
    auto consume=[&](const char* data,std::size_t size){
        if(raw)raw.write(data,static_cast<std::streamsize>(size));
        for(std::size_t i=0;i<size;++i){char ch=data[i];
            if(ch!='\n'){if(!discard){buffer+=ch;if(buffer.size()>512){discard=true;buffer.clear();}}continue;}
            if(discard){++bad;discard=false;buffer.clear();continue;}
            deployment::Telemetry p{};std::string status;
            if(!deployment::parse(buffer,p,status)){++bad;std::cerr<<"Rejected frame: "<<status<<"\n";buffer.clear();continue;}
            ++good;bool accepted=odom.update(p,status);if(!accepted)++rejected;
            const auto& pose=odom.pose();out<<p.sequence<<','<<p.tx_ms<<','<<p.enc_ms<<','<<p.imu_ms<<','<<accepted<<','<<status<<','<<pose.x<<','<<pose.y<<','<<pose.yaw<<','<<pose.vx<<','<<pose.r<<','<<pose.travel<<','<<p.total_ticks<<','<<p.flags<<','<<std::hex<<p.calibration<<std::dec<<'\n';buffer.clear();
        }
    };
    if(!input.empty()){
        std::ifstream in(input,std::ios::binary);if(!in)throw std::runtime_error("cannot open input");char bytes[256];
        while(in){in.read(bytes,sizeof(bytes));consume(bytes,static_cast<std::size_t>(in.gcount()));}
    }else{
        Port port;port.fd=::open(device.c_str(),O_RDONLY|O_NOCTTY|O_NONBLOCK);
        if(port.fd<0)throw std::runtime_error(std::strerror(errno));
        termios settings{};if(tcgetattr(port.fd,&settings)!=0)throw std::runtime_error("tcgetattr failed");
        cfmakeraw(&settings);cfsetispeed(&settings,B115200);cfsetospeed(&settings,B115200);
        settings.c_cflag=(settings.c_cflag&~(CSIZE|PARENB|CSTOPB|CRTSCTS))|CS8|CLOCAL|CREAD;
        if(tcsetattr(port.fd,TCSANOW,&settings)!=0)throw std::runtime_error("tcsetattr failed");
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);const auto start=std::chrono::steady_clock::now();
        while(!stopped&&std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<duration){
            pollfd event{port.fd,POLLIN,0};int result=::poll(&event,1,100);
            if(result<0){if(errno==EINTR)continue;throw std::runtime_error("poll failed");}
            if(event.revents&(POLLERR|POLLHUP|POLLNVAL))throw std::runtime_error("serial disconnected");
            if(event.revents&POLLIN){char bytes[256];auto count=::read(port.fd,bytes,sizeof(bytes));
                if(count>0)consume(bytes,static_cast<std::size_t>(count));
                else if(count<0&&errno!=EAGAIN&&errno!=EINTR)throw std::runtime_error("serial read failed");}
        }
    }
    if(!buffer.empty()||discard){++bad;std::cerr<<"Incomplete trailing frame ignored\n";}
    std::cerr<<"CRC-valid="<<good<<" invalid_frames="<<bad<<" odometry_rejected="<<rejected<<" fault_latched="<<odom.faulted()<<'\n';
    return good==0||odom.faulted()?2:0;
}catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return 1;}}
