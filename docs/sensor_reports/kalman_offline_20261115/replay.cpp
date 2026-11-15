#include "yaw_kalman.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
int main(int argc, char** argv) {
  if(argc!=4) return 2;
  deployment::YawKalman k({std::atof(argv[1]),std::atof(argv[2]),std::atof(argv[3])});
  double t,y,r,prev=0; bool first=true;
  std::cout<<std::setprecision(17);
  while(std::cin>>t>>y>>r) {
    if(first) {k.initialize(y);first=false;}
    else k.update(r,t-prev,y);
    prev=t;
    std::cout<<k.yaw()<<","<<k.bias()<<"\n";
  }
}
