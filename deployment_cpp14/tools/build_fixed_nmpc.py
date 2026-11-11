"""Build host parity probe against the CasADi Python wheel (not Pi runtime)."""
import subprocess
from pathlib import Path
import casadi
root=Path(__file__).resolve().parents[1]
wheel=Path(casadi.__file__).parent
build=root/'build'
build.mkdir(exist_ok=True)
subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Wpedantic','-Werror',
    '-D_GLIBCXX_USE_CXX11_ABI=0','-I'+str(root/'include'),'-isystem',str(wheel/'include'),
    str(root/'src/fixed_nmpc.cpp'),str(root/'src/fixed_nmpc_probe.cpp'),
    '-L'+str(wheel),'-Wl,-rpath,'+str(wheel),'-lcasadi','-o',str(build/'fixed_nmpc_probe')],check=True)
print(build/'fixed_nmpc_probe')

common=['g++','-std=c++14','-O2','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(root/'include')]
for name,sources in [('actor_critic_probe',['online_ac.cpp','actor_critic_probe.cpp']),
                     ('online_controller_probe',['online_ac.cpp','online_controller.cpp','online_controller_probe.cpp'])]:
    subprocess.run(common+[str(root/'src'/s) for s in sources]+['-o',str(build/name)],check=True)
subprocess.run(common+['-D_GLIBCXX_USE_CXX11_ABI=0','-isystem',str(wheel/'include')]+
    [str(root/'src'/s) for s in ['fixed_nmpc.cpp','online_ac.cpp','online_controller.cpp','four_mode_probe.cpp']]+
    ['-L'+str(wheel),'-Wl,-rpath,'+str(wheel),'-lcasadi','-o',str(build/'four_mode_probe')],check=True)
print('Built Fixed NMPC, Actor/Critic math, controller and four-mode offline probes')

subprocess.run(common+[str(root/'src'/s) for s in ['online_ac.cpp','online_controller.cpp']]+[str(root/'tests/test_online_contract.cpp'),'-o',str(build/'test_online_contract')],check=True)
subprocess.run([str(build/'test_online_contract')],check=True)

subprocess.run(common+[str(root/'src'/s) for s in ['reference_path.cpp','reference_probe.cpp']]+['-o',str(build/'reference_probe')],check=True)
subprocess.run(common+['-D_GLIBCXX_USE_CXX11_ABI=0','-isystem',str(wheel/'include')]+
    [str(root/'src'/s) for s in ['fixed_nmpc.cpp','online_ac.cpp','online_controller.cpp','reference_path.cpp','route_controller.cpp']]+
    ['-L'+str(wheel),'-Wl,-rpath,'+str(wheel),'-lcasadi','-o',str(build/'route_controller')],check=True)
print('Built native reference/route controllers')
