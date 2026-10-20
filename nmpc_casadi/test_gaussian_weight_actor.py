"""Independent Gaussian Actor tests: no training driver, no RL episodes."""
import unittest
import numpy as np
from gaussian_weight_actor import GaussianWeightActor, build_actor_observation


def sample_obs(*, x=None, ref=None, xy0=None):
    return build_actor_observation(
        [.12,-.02,3.12,.1,2.4,3.5] if x is None else x,
        [.1,0.,-3.12,0.,2.41,3.48] if ref is None else ref,
        [.2,-.1], path_progress_m=4., path_start_m=0.,
        path_end_m=8., curvature_now=1.2, curvature_horizon_end=-.8,
        reference_origin_xy=[1.,1.] if xy0 is None else xy0,
        reference_initial_heading_rad=.5, train_path_length_m=8.,
    )


def make_actor():
    q = np.array([12.,3.,15.,6.,183.75,192.5])
    r = np.array([.6,16.])
    # Bounds here are test-only SYNTHETIC numbers, NOT safety screened
    # or approved physical Q/R bounds for actual closed-loop training.
    return GaussianWeightActor(
        np.random.default_rng(234),q,r,q*.5,q*1.5,r*.5,r*1.5,
        sigma_lower=.02,sigma_upper=.5,sigma_initial=.12)


class ActorIndependentTests(unittest.TestCase):
    def test_17_features_scaling_wrap(self):
        s=sample_obs()
        self.assertEqual(s.shape,(17,))
        self.assertAlmostEqual(s[0],1.2)
        self.assertAlmostEqual(s[2],(3.12-.5)/np.pi)
        self.assertAlmostEqual(s[4],(2.4-1.)/8.)
        self.assertAlmostEqual(s[6],.2)
        self.assertAlmostEqual(s[8],
            np.arctan2(np.sin(6.24),np.cos(6.24))/.2)
        self.assertAlmostEqual(s[10],-.1)
        self.assertAlmostEqual(s[11],.2)
        self.assertAlmostEqual(s[12],.1)
        self.assertAlmostEqual(s[13],-.1/.55)
        self.assertAlmostEqual(s[14],.5)
        self.assertAlmostEqual(s[15],.6)
        self.assertAlmostEqual(s[16],-.4)

    def test_translation_invariance(self):
        a=sample_obs()
        b=sample_obs(
            x=[.12,-.02,3.12,.1,22.4,-6.5],
            ref=[.1,0.,-3.12,0.,22.41,-6.52],
            xy0=[21.,-9.])
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_normalizer_rejects_invalid_reference_and_length(self):
        with self.assertRaises(ValueError):
            sample_obs(x=np.zeros(5))
        with self.assertRaises(ValueError):
            build_actor_observation(
                np.zeros(6),np.zeros(6),np.zeros(2),
                path_progress_m=0,path_start_m=0,path_end_m=1,
                curvature_now=0,curvature_horizon_end=0,
                reference_origin_xy=np.zeros(2),
                reference_initial_heading_rad=0,train_path_length_m=0)

    def test_exactly_6352_trainable_parameters_and_shapes(self):
        a=make_actor()
        self.assertEqual(a.parameter_count,6352)
        self.assertEqual(a.params["W1"].shape,(17,64))
        self.assertEqual(a.params["W2"].shape,(64,64))
        self.assertEqual(a.params["W_mu"].shape,(64,8))
        self.assertEqual(a.params["W_sigma"].shape,(64,8))
        mu,sig=a.forward(sample_obs())
        self.assertEqual(mu.shape,(8,))
        self.assertEqual(sig.shape,(8,))

    def test_deterministic_nominal_initialization(self):
        a=make_actor()
        for obs in [np.zeros(17),sample_obs(),np.ones(17)]:
            p=a.propose(obs,deterministic=True)
            np.testing.assert_allclose(p.q_diagonal,[12,3,15,6,183.75,192.5])
            np.testing.assert_allclose(p.r_diagonal,[.6,16.])
            np.testing.assert_allclose(p.sigma,np.full(8,.12))

    def test_sampled_qr_bounded_and_latent_log_probability(self):
        a=make_actor()
        obs=sample_obs()
        for _ in range(120):
            out=a.propose(obs)
            w=np.r_[out.q_diagonal,out.r_diagonal]
            self.assertEqual(out.latent.shape,(8,))
            self.assertTrue(np.all(w>=a.lower))
            self.assertTrue(np.all(w<=a.upper))
            self.assertTrue(np.isfinite(out.latent_log_probability))
            self.assertAlmostEqual(out.latent_log_probability,
                                   a.latent_log_probability(obs,out.latent))

    def test_eight_weight_coordinates_independent(self):
        a=make_actor()
        q,r=a.physical_diagonals(np.zeros(8))
        reference=np.r_[q,r]
        for i in range(8):
            z=np.zeros(8);z[i]=.3
            q,r=a.physical_diagonals(z)
            difference=np.r_[q,r]-reference
            self.assertGreater(difference[i],0)
            np.testing.assert_allclose(
                np.delete(difference,i),np.zeros(7),atol=1e-14)

    def test_requires_explicit_positive_sigma_and_qr_bounds(self):
        q=np.ones(6);r=np.ones(2)
        with self.assertRaises(TypeError):
            GaussianWeightActor(np.random.default_rng(1),q,r,
                                q*.5,q*2,r*.5,r*2)
        with self.assertRaises(ValueError):
            GaussianWeightActor(np.random.default_rng(1),q,r,
                                q,q*2,r*.5,r*2,
                                sigma_lower=.02,sigma_initial=.1,sigma_upper=.3)
        with self.assertRaises(ValueError):
            GaussianWeightActor(np.random.default_rng(1),q,r,
                                q*.5,q*2,r*.5,r*2,
                                sigma_lower=.3,sigma_initial=.1,sigma_upper=.5)

    def test_both_heads_receive_policy_gradients(self):
        a=make_actor()
        obs=np.stack([sample_obs(),np.zeros(17)])
        latent=np.stack([np.full(8,.25),np.full(8,-.15)])
        loss,g=a.loss_and_gradients(obs,latent,np.array([1.2,-.4]))
        self.assertTrue(np.isfinite(loss))
        for name in ["b_mu","b_sigma","W_mu","W_sigma"]:
            self.assertGreater(np.linalg.norm(g[name]),0,name)

    def test_finite_difference_gradient_both_heads_hidden_layers(self):
        a=make_actor()
        # Nonzero synthetic head weights activate backprop to hidden layers.
        a.params["W_mu"][:3,:]=.012
        a.params["W_sigma"][:3,:]=-.008
        obs=np.stack([sample_obs(),np.linspace(-1,1,17)])
        latent=np.stack([np.linspace(-.2,.3,8),np.linspace(.4,-.15,8)])
        td=np.array([-.3,.8])
        _, analytic=a.loss_and_gradients(obs,latent,td)
        eps=1e-5
        for name,index in [
            ("b_mu",(0,)),("b_sigma",(1,)),("W_mu",(0,0)),
            ("W_sigma",(0,1)),("W2",(0,0)),("b2",(0,)),
            ("W1",(0,0)),("b1",(0,))
        ]:
            p=a.params[name]
            old=p[index]
            p[index]=old+eps
            upper=a.loss_and_gradients(obs,latent,td)[0]
            p[index]=old-eps
            lower=a.loss_and_gradients(obs,latent,td)[0]
            p[index]=old
            self.assertAlmostEqual(
                (upper-lower)/(2*eps),analytic[name][index],
                delta=1e-6,msg=name)

    def test_zero_td_zero_gradient(self):
        a=make_actor()
        obs=np.stack([sample_obs(),np.zeros(17)])
        loss,g=a.loss_and_gradients(obs,np.full((2,8),.1),np.zeros(2))
        self.assertAlmostEqual(loss,0.)
        for value in g.values():
            np.testing.assert_array_equal(value,np.zeros_like(value))

    def test_bad_shapes_rejected(self):
        a=make_actor()
        with self.assertRaises(ValueError):
            a.latent_log_probability(sample_obs(),np.zeros(6))
        with self.assertRaises(ValueError):
            a.loss_and_gradients(np.zeros((3,17)),np.zeros((2,8)),np.ones(3))


if __name__=="__main__":
    unittest.main()
