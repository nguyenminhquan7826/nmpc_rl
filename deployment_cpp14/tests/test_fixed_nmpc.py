"""Independent native C++ vs existing Python NMPC; Python is host test only."""
import os,sys,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
import casadi as ca
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'nmpc_casadi'))
from nmpc_casadi import build_nmpc_solver,pack_parameters,shift_warm_start,unpack_solution
from vehicle_model_casadi_variants import build_continuous_model,build_rk4_discrete_model
EXE=Path(os.environ.get('FIXED_NMPC_PROBE',str(ROOT/'deployment_cpp14/build/fixed_nmpc_probe')))


def fixture(hardware=True,N=10,wrap=False,infeasible=False):
    L=.25 if hardware else .2515;limit=np.deg2rad(22.835397) if hardware else .55
    p=np.array([23.55/9.81,.044,.40,.35,L*132/252,L*120/252,.02,9.81,.01])
    c=dict(u_min=np.array([-2.,-limit]),u_max=np.array([2.,limit]),du_min=np.array([-.2,-.04]),du_max=np.array([.2,.04]),vx_min=.05,vx_max=.15,r_max=4.,beta_max=.6,vy_abs_max=.6,sqp_max_iter=7)
    q=np.array([12,3,120,6,153.75,152.5]);r=np.array([15,100]);qf=q*2
    fd=build_rk4_discrete_model(build_continuous_model(),.071)
    x=np.array([.4 if infeasible else .1,0,np.pi-.001 if wrap else .15,0,0,0.01])
    u=np.array([.1962,.03]);refs=[x.copy()];refs[0][5]=0
    for _ in range(N):refs.append(np.array(fd(refs[-1],u,p)).ravel())
    refs=np.array(refs)
    if wrap:refs[:,2]-=2*np.pi
    states=[];v=x.copy()
    for _ in range(N):v=np.array(fd(v,u,p)).ravel();states.extend(v)
    guess=np.r_[states,np.tile(u,N)]
    data=dict(N=[N],Ts=[.071],sqp_max_iter=[7],q=q,r=r,qf=qf,model=p,
        state_lower=[.05,-np.inf,-np.inf,-4,-np.inf,-np.inf],state_upper=[.15,np.inf,np.inf,4,np.inf,np.inf],
        input_lower=c['u_min'],input_upper=c['u_max'],du_lower=c['du_min'],du_upper=c['du_max'],
        beta_max=[.6],vy_abs_max=[.6],state=x,previous=u,reference=refs.ravel(),guess=guess)
    return data,c,fd


def write_input(path,data):
    with open(path,'w') as out:
        out.write('FIXED_NMPC_V1\n')
        for key,values in data.items():out.write(key+' '+str(len(values))+' '+' '.join(format(float(v),'.17g') for v in values)+'\n')


def read_output(path):
    values=path.read_text().split();assert values.pop(0)=='FIXED_NMPC_RESULT_V1'
    result={};i=0
    while i<len(values):
        key=values[i];n=int(values[i+1]);result[key]=np.array(list(map(float,values[i+2:i+2+n])));i+=2+n
    return result


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXE.is_file():raise AssertionError('Native Fixed NMPC runner missing: '+str(EXE))

    def run_case(self,data,expected=0):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'input.txt';dst=Path(tmp)/'output.txt';write_input(src,data)
            run=subprocess.run([str(EXE),str(src),str(dst)],text=True,capture_output=True)
            self.assertEqual(run.returncode,expected,run.stdout+run.stderr)
            return read_output(dst) if dst.exists() else None

    def test_python_cpp_dynamics_cost_derivatives_and_solve(self):
        for hardware,N,wrap in [(False,10,False),(True,10,False),(True,10,True),(True,1,False)]:
            with self.subTest(hardware=hardware,N=N,wrap=wrap):
                d,c,fd=fixture(hardware,N,wrap);out=self.run_case(d)
                py=build_nmpc_solver(fd,N,np.diag(d['q']),np.diag(d['r']),np.diag(d['qf']),c)
                par=pack_parameters(d['state'],d['previous'],d['reference'].reshape(N+1,6),d['model'])
                f,grad,g,jac=py['solver'].get_function('nlp_jac_fg')(d['guess'],par)
                for key,value in [('parameters',par),('eval_f',f),('eval_g',g),('eval_gradient',grad),('eval_jacobian',jac),('rk4',fd(d['state'],d['previous'],d['model']))]:
                    np.testing.assert_allclose(out[key],np.asarray(value).ravel(order='F'),rtol=1e-10,atol=1e-9,err_msg=key)
                sol=py['solver'](x0=d['guess'],p=par,lbx=py['lbx'],ubx=py['ubx'],lbg=py['lbg'],ubg=py['ubg'])
                self.assertTrue(py['solver'].stats()['success']);self.assertEqual(out['success'][0],1)
                self.assertLessEqual(out['max_violation'][0],1e-6)
                self.assertEqual(out['solver_status_code'][0],0)
                np.testing.assert_allclose(out['solution'],np.asarray(sol['x']).ravel(),rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(out['objective'],np.asarray(sol['f']).ravel(),rtol=1e-7,atol=1e-8)
                X,U=unpack_solution(np.asarray(sol['x']),N)
                np.testing.assert_allclose(out['control'],U[:,0],rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(out['shifted'],shift_warm_start(X,U),rtol=1e-6,atol=1e-7)
                print('PARITY',hardware,N,wrap,'control_maxerr',np.max(np.abs(out['control']-U[:,0])))

    def test_receding_horizon_warm_start(self):
        d,c,fd=fixture()
        py=build_nmpc_solver(fd,10,np.diag(d['q']),np.diag(d['r']),np.diag(d['qf']),c)
        largest=0.
        for step in range(20):
            # Straight reference first; curved reference after step 5.
            nominal=np.array([.1962,0. if step<5 else .04])
            refs=[np.asarray(d['state']).copy()];refs[0][5]-=.01
            for _ in range(10):refs.append(np.asarray(fd(refs[-1],nominal,d['model'])).ravel())
            d['reference']=np.asarray(refs).ravel()
            out=self.run_case(d)
            par=pack_parameters(d['state'],d['previous'],np.asarray(refs),d['model'])
            sol=py['solver'](x0=d['guess'],p=par,lbx=py['lbx'],ubx=py['ubx'],lbg=py['lbg'],ubg=py['ubg'])
            self.assertTrue(py['solver'].stats()['success'])
            X,U=unpack_solution(np.asarray(sol['x']),10)
            largest=max(largest,float(np.max(np.abs(out['control']-U[:,0]))))
            np.testing.assert_allclose(out['control'],U[:,0],rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(out['shifted'],shift_warm_start(X,U),rtol=1e-6,atol=1e-7)
            d['state']=np.asarray(fd(d['state'],out['control'],d['model'])).ravel()
            d['previous']=out['control'];d['guess']=out['shifted']
        print('ROLLOUT 20 steps control_maxerr',largest)

    def test_infeasible_has_no_new_valid_control(self):
        d,_,_=fixture(infeasible=True);out=self.run_case(d,2)
        self.assertEqual(out['success'][0],0);self.assertEqual(len(out['control']),0)
        self.assertNotEqual(out['solver_status_code'][0],0)

    def test_invalid_input_rejected(self):
        for change in [('state',np.zeros(5)),('state',np.array([np.nan]*6)),('N',[0]),('Ts',[-1]),('q',[-1]*6),('state_lower',[.2]*6),('reference',[np.nan]*66)]:
            d,_,_=fixture();d[change[0]]=change[1]
            with self.subTest(field=change[0]):self.assertIsNone(self.run_case(d,1))

if __name__=='__main__':unittest.main()
