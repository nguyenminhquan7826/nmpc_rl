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
