"""Port checks against original Python reference_manager and CAD tail staging."""
import os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from test_online_ac import ROOT,write_values,read_values
from reference_manager import ReferenceManager
from cad_eval_reference import prepare_cad_evaluation_reference
EXE=Path(os.environ.get('REFERENCE_PROBE',str(ROOT/'build/reference_probe')))
REAL=ROOT/'nmpc_casadi/test_reference/test_xref.csv'

class ReferenceTests(unittest.TestCase):
    def compare(self,path,points,stop=False,tail=False):
        self.assertTrue(EXE.is_file(),'Native reference manager missing')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);query=tmp/'query';dest=tmp/'output'
            data=dict(xy=np.asarray(points).ravel(),Ts=[.071],N=[10],monotonic=[1],search_back=[.15],search_forward=[1.2],stop_at_end=[int(stop)],midpoint=[1],tail_length=[.2 if tail else 0])
            write_values(query,'ROUTE_QUERY_V1',data)
            run=subprocess.run([str(EXE),str(path),str(query),str(dest)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            native=read_values(dest,'ROUTE_RESULT_V1')
            if tail:
                _,files=prepare_cad_evaluation_reference(path,tmp/'python',.2);path=files['tail_file']
            rm=ReferenceManager(path,search_back_m=.15,search_forward_m=1.2,stop_at_end=stop)
            expected=dict(projection=[],horizon=[],xref=[],curvature=[])
            for x,y in points:
                r=rm.get_nmpc_reference(x,y,.071,10);p=r['projection']
                expected['projection'].extend([p[k] for k in ['s','X_proj','Y_proj','distance','segment_idx','alpha']]);expected['horizon'].extend(r['s_horizon']);expected['xref'].extend(r['xref'].ravel());expected['curvature'].extend(rm.interpolate_curvature(r['s_horizon']))
            for k,v in expected.items():np.testing.assert_allclose(native[k],v,rtol=2e-11,atol=2e-11,err_msg=k)
            np.testing.assert_allclose(native['route_rows'],rm.df[['s_m','vx_ref_mps','vy_ref_mps','psi_ref_rad','r_ref_radps','X_ref_m','Y_ref_m','kappa_1_per_m']].assign(psi_ref_rad=rm.psi).to_numpy().ravel(),rtol=2e-11,atol=2e-11)

    def test_real_route_and_tail(self):
        df=pd.read_csv(REAL);points=df[['X_ref_m','Y_ref_m']].iloc[::15].to_numpy();points=np.vstack([points,df[['X_ref_m','Y_ref_m']].iloc[-1].to_numpy()])
        self.compare(REAL,points);self.compare(REAL,points,tail=True)

    def test_interpolation_matches_numpy_operation_order(self):
        self.assertTrue(EXE.is_file())
        points=pd.read_csv(REAL)[['X_ref_m','Y_ref_m']].iloc[::17].to_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);q=tmp/'query';out=tmp/'out'
            write_values(q,'ROUTE_QUERY_V1',dict(xy=points.ravel(),Ts=[.071],N=[10],monotonic=[1],search_back=[.15],search_forward=[1.2],stop_at_end=[0],midpoint=[1],tail_length=[0]))
            run=subprocess.run([str(EXE),str(REAL),str(q),str(out)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            d=read_values(out,'ROUTE_RESULT_V1');rows=d['route_rows'].reshape(-1,8)
            expected=np.array([np.interp(d['horizon'],rows[:,0],rows[:,j]) for j in range(1,7)]).T.ravel()
            np.testing.assert_array_equal(d['xref'],expected)

    def test_wrap_monotonic_clamp_crossing_and_stop(self):
        df=pd.DataFrame(dict(s_m=[0,1,2,3,4,5],vx_ref_mps=[.1]*6,vy_ref_mps=[0]*6,psi_ref_rad=[3.13,-3.13,-3.1,3.1,0,3.14],r_ref_radps=[0]*6,X_ref_m=[0,1,1,0,0,1],Y_ref_m=[0,0,1,1,0,0],kappa_1_per_m=[0,.1,.2,.3,.4,0]))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'quoted.csv';df.to_csv(p,index=False,quoting=1)
            self.compare(p,[(0,.1),(.8,.1),(.2,.1),(1,.9),(.2,1),(0,.2),(1,0)],stop=True)

    def test_invalid_csv_rejected(self):
        self.assertTrue(EXE.is_file(),'Native reference manager missing')
        for kind in ('missing','nan','s_order','short','duplicate_header'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp);df=pd.read_csv(REAL).iloc[:4].copy()
                if kind=='missing':df=df.drop(columns=['s_m'])
                if kind=='nan':df.loc[1,'X_ref_m']=np.nan
                if kind=='s_order':df.loc[1,'s_m']=0
                if kind=='short':df=df.iloc[:1]
                p=tmp/'ref.csv';df.to_csv(p,index=False)
                if kind=='duplicate_header':p.write_text(p.read_text().replace('s_m,','s_m,s_m,',1))
                q=tmp/'query';o=tmp/'out';write_values(q,'ROUTE_QUERY_V1',dict(xy=[0,0],Ts=[.071],N=[10],monotonic=[1],search_back=[.15],search_forward=[1.2],stop_at_end=[0],midpoint=[1],tail_length=[0]))
                r=subprocess.run([str(EXE),str(p),str(q),str(o)],capture_output=True,text=True);self.assertEqual(r.returncode,1);self.assertFalse(o.exists())
if __name__=='__main__':unittest.main()
