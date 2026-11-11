# Fixed NMPC C++14 / Python numerical parity

User approved implementing Fixed NMPC before Actor/Critic and without hardware.
Use experimental-deployment, which is already the dedicated deployment branch;
work in place with clean status rather than creating another branch/worktree.

Implement the bicycle dynamics, RK4 and multiple-shooting NLP independently
in C++14 using CasADi 3.7.2 C++ and SQP/QRQP. Do not merely load a serialized
Python solver. Preserve existing state/parameter packing and stage/terminal
cost placement. Do not modify offline training or transmit actuator commands.
Hardware profile limits steering to +/-22.835397 degrees and wheelbase .250 m;
legacy comparison profile retains Python .2515 m and +/- .55 rad. Both Python
and C++ receive the SAME explicit config and fixtures for each profile.

Alternative serialized export is easier but would not test a native model port;
closed-loop plant simulation/replay is subsequent work, not required here.

Provide a no-Python C++ numerical runner with versioned text input/output,
explicit finite/dimension validation, solver status, maximum constraint
violation and no newly valid control on failure. Test state/error/RK4,
objective, constraints, objective gradient and constraint Jacobian; compare
solutions and first controls when both solves succeed. Test warm-start
packing and shifting, invalid input, angle wrap and infeasible solve.
Python scripts only generate fixtures and compare outputs on the workstation.
Document how to build with an installed CasADi C++ library on Pi. Host timing
must not be presented as on-Pi timing; no actual vehicle testing claims.
