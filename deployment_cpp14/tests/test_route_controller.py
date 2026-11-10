"""End-to-end route comparison uses the unchanged Python evaluation environment."""
import os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np,json
from test_online_ac import ROOT,model_fixture,PARAMS,write_values,read_values
from test_fixed_nmpc import fixture
import pandas as pd
TOOL=ROOT/'deployment_cpp14/tools/compare_route.py'
class RouteControllerTests(unittest.TestCase):
    def test_native_route_parity_smoke(self):
        self.assertTrue(TOOL.is_file(),'Full-route comparison driver missing')
        actor,critic,d=model_fixture()
        cfg=dict(Q_lower=d['lower'][:6].tolist(),Q_upper=d['upper'][:6].tolist(),R_lower=d['lower'][6:].tolist(),R_upper=d['upper'][6:].tolist(),P_lower=d['critic_lower'].tolist(),P_upper=d['critic_upper'].tolist(),initial_Q=d['fixed_q'],initial_R=d['fixed_r'],initial_P=d['fixed_qf'],fixed_Qf=d['fixed_qf'],sigma_lower=.05,sigma_initial=.2,sigma_upper=.6,gamma=.91,actor_lr=.0005,critic_lr=50000,critic_p_lr_factors=[.5,.5,.1,.05,4,4],actor_output_lr_factors=d['actor_factors'],train_length_m=10.,Qe_fixed=[20,20,100,20,600,600],Re_fixed=[50,100],train_solver_failure_penalty=10.,train_failure_terminal_penalty=100.)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);ck=tmp/'checkpoint.npz';out=tmp/'comparison'
            np.savez(ck,metadata=np.array(json.dumps(cfg)),critic_physical_P=critic.p,**{'actor_'+k:v for k,v in actor.params.items()})
            run=subprocess.run([sys.executable,str(TOOL),'--checkpoint',str(ck),'--reference',str(ROOT/'nmpc_casadi/test_reference/test_xref.csv'),'--output',str(out),'--max-steps','8','--horizon','3','--seed','2'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            report=json.loads((out/'parity.json').read_text());self.assertTrue(report['all_pass']);self.assertEqual(len(report['modes']),4)
            for mode,result in report['modes'].items():self.assertTrue(result['pass']);self.assertEqual(result['steps_python'],8);self.assertEqual(result['steps_cpp'],8)
            self.assertTrue((out/'trajectories.png').is_file());self.assertTrue((out/'cte_mean_max.png').is_file())
    def test_failed_solve_preserves_warm_start(self):
        binary=ROOT/'build/route_controller';self.assertTrue(binary.is_file())
        _,_,m=model_fixture();d,_,_=fixture(hardware=False)
        d['state_lower'][0]=.001;d['state_upper'][0]=.002
        reference=ROOT/'nmpc_casadi/test_reference/test_xref.csv'
        d.update(max_steps=[5],seed=[0],ey=[.05],heading_rad=[0],tail_length=[.2],feature_length=[float(pd.read_csv(reference).s_m.iloc[-1])],min_reference_speed=[.05],max_reference_speed=[.1],task_q=[20,20,100,20,600,600],task_r=[50,100],solver_penalty=[10],failure_penalty=[100])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);mp=tmp/'model';cp=tmp/'settings';dest=tmp/'out.csv';write_values(mp,'AC_MODEL_V1',m);write_values(cp,'NMPC_ROUTE_V1',d)
            run=subprocess.run([str(binary),str(mp),str(reference),str(cp),str(dest),'fixed'],capture_output=True,text=True)
            self.assertEqual(run.returncode,2,run.stderr)
            log=pd.read_csv(dest);self.assertEqual(len(log),3);self.assertTrue((log.solver_success==0).all())
            np.testing.assert_array_equal(log.warm_start_delta_on_failure,np.zeros(3))
            self.assertEqual(log.failure_terminal.iloc[-1],1)

    def test_fixed_simultaneous_completion_and_third_failure(self):
        binary=ROOT/'build/route_controller';self.assertTrue(binary.is_file());_,_,m=model_fixture();d,_,_=fixture(hardware=False)
        d['state_lower'][0]=.001;d['state_upper'][0]=.002
        d.update(max_steps=[5],seed=[0],ey=[.05],heading_rad=[0],tail_length=[.2],feature_length=[.05],min_reference_speed=[.05],max_reference_speed=[.1],task_q=[20,20,100,20,600,600],task_r=[50,100],solver_penalty=[10],failure_penalty=[100])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);mp=tmp/'model';cp=tmp/'settings';dest=tmp/'out.csv';reference=tmp/'tiny.csv';write_values(mp,'AC_MODEL_V1',m);write_values(cp,'NMPC_ROUTE_V1',d)
            pd.DataFrame(dict(index=[0,1,2],s_m=[0,.025,.05],t_ref_s=[0,.25,.5],vx_ref_mps=[.1]*3,vy_ref_mps=[0]*3,psi_ref_rad=[0]*3,r_ref_radps=[0]*3,X_ref_m=[0,.025,.05],Y_ref_m=[0]*3,kappa_1_per_m=[0]*3)).to_csv(reference,index=False)
            run=subprocess.run([str(binary),str(mp),str(reference),str(cp),str(dest),'fixed'],capture_output=True,text=True)
            self.assertEqual(run.returncode,2,run.stderr)
            log=pd.read_csv(dest);self.assertEqual(len(log),3);self.assertEqual(log.reached_end.iloc[-1],1);self.assertEqual(log.failure_terminal.iloc[-1],1)
            # Malformed bound vectors must be rejected before any indexing.
            for key in ('input_lower','input_upper'):
                for size in (0,1):
                    bad=d.copy();bad[key]=[-2.]*size;write_values(cp,'NMPC_ROUTE_V1',bad);target=tmp/(key+str(size)+'.csv')
                    run=subprocess.run([str(binary),str(mp),str(reference),str(cp),str(target),'fixed'],capture_output=True,text=True)
                    self.assertEqual(run.returncode,1,run.stderr);self.assertIn('Invalid shape: '+key,run.stderr);self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
