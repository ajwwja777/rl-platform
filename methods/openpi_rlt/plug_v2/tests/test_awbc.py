import unittest
import numpy as np
import jax,jax.numpy as jnp
from methods.openpi_rlt.plug_v2 import rtc_upstream_core as rtc  # adds the pinned upstream src path
from rlt_online_rl import trainer as upstream
from rlt_online_rl.config import RLTOnlineRLConfig
from methods.openpi_rlt.plug_v2.awbc_core import awbc_weights,awbc_weights_from_advantage,train_step_awbc,AWBC_METRIC_KEYS
from methods.openpi_rlt.plug_v2.awbc_train import transition_success

def small_batch(seed,n=4):
    random=np.random.default_rng(seed)
    batch={k:jnp.asarray(random.normal(size=shape),jnp.float32) for k,shape in {
        'z_rl':(n,8),'proprio':(n,7),'action_chunk':(n,10,7),'ref_chunk':(n,10,7),
        'next_z_rl':(n,8),'next_proprio':(n,7),'next_ref_chunk':(n,10,7),'rewards':(n,10)}.items()}
    batch.update(done=jnp.array([False,True,False,True]),source_chunk=jnp.array([[0]*10,[2]*10,[1]*10,[1]*10],dtype=jnp.uint8),
                 duration=jnp.full(n,10),td_valid=jnp.ones(n,bool),success=jnp.array([1,1,0,1],jnp.int8),original_done=jnp.array([False,True,False,True]))
    return batch

class AWBCTest(unittest.TestCase):
    def test_weights_by_row_kind(self):
        q_data=jnp.array([.6,.2,.9,.5]);q_ref=jnp.array([.5,.9,.1,.5])
        success=jnp.array([1,0,0,1]);human=jnp.array([0,1,0,0])
        use_data,weight,adv=awbc_weights(q_data,q_ref,success,human,beta=.1,max_weight=5.)
        np.testing.assert_array_equal(np.asarray(use_data),[True,True,False,True])
        np.testing.assert_allclose(np.asarray(weight),[np.exp(1.),1.,1.,1.],rtol=1e-5)  # success: exp(adv/beta); human/failure: 1
        _,capped,_=awbc_weights(jnp.array([1.]),jnp.array([0.]),jnp.array([1]),jnp.array([0]),beta=.1,max_weight=5.)
        self.assertEqual(float(capped[0]),5.)
        _,flat,_=awbc_weights(q_data,q_ref,success,human,beta=1e6,max_weight=5.)
        np.testing.assert_allclose(np.asarray(flat),1.,rtol=1e-5)  # uninformative critic degrades to filtered BC

    def test_hil_prefix_relabelled_as_failure(self):
        frame=np.arange(10);source=np.zeros((10,10),np.uint8);source[6:9]=2  # policy 0-5, human 6-8, policy 9 after handover
        out=transition_success(frame,source,expert=False,episode_success=True)
        np.testing.assert_array_equal(out,[False]*6+[True]*4)
        np.testing.assert_array_equal(transition_success(frame,source,expert=True,episode_success=True),True)
        np.testing.assert_array_equal(transition_success(frame,source,expert=False,episode_success=False),False)
        np.testing.assert_array_equal(transition_success(frame,np.zeros((10,10),np.uint8),expert=False,episode_success=True),True)

    def test_step_updates_actor_on_cadence_with_finite_metrics(self):
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(3));initial=state;batch=small_batch(5)
        for step in range(2):
            state,m=train_step_awbc(state,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=0.,delta_weight=10.,beta=.1,max_weight=5.,ref_weight=.1)
            self.assertEqual(float(m['did_actor_update']),float(step==1))
        for key in AWBC_METRIC_KEYS+('critic_loss','q1_mean'):self.assertTrue(np.isfinite(float(m[key])),key)
        self.assertEqual(float(m['weighted_q']),0.)
        self.assertAlmostEqual(float(m['awbc_data_target_ratio']),.75);self.assertAlmostEqual(float(m['awbc_success_ratio']),.75)
        self.assertGreaterEqual(float(m['awbc_weight_min']),0.);self.assertLessEqual(float(m['awbc_weight_max']),5.)
        assert any(not np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(initial.actor_params),jax.tree_util.tree_leaves(state.actor_params)))
        self.assertEqual(int(state.actor_version),1)

    def test_burnin_freezes_actor_like_pinned_core(self):
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(7));other=state;batch=small_batch(2)
        for _ in range(2):
            state,m=train_step_awbc(state,batch,actor=a,critic=q,rl_config=cfg,bc_weight=0.,q_weight=0.,delta_weight=10.)
            other,n=rtc.train_step(other,batch,actor=a,critic=q,rl_config=cfg,bc_weight=0.,q_weight=0.,delta_weight=10.)
            self.assertEqual(float(m['did_actor_update']),0.)
        for x,y in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(other)):np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-6)

    def test_no_gradient_from_q_into_actor(self):
        # Scaling critic outputs changes weights only through adv; with beta huge, the actor update is critic-independent.
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(11));batch=small_batch(9)
        scaled=state.replace(critic_params=jax.tree_util.tree_map(lambda x:x*3.,state.critic_params))
        s1,_=train_step_awbc(state,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=0.,delta_weight=10.,beta=1e6,max_weight=5.,ref_weight=.1)
        s2,_=train_step_awbc(scaled,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=0.,delta_weight=10.,beta=1e6,max_weight=5.,ref_weight=.1)
        s1,_=train_step_awbc(s1,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=0.,delta_weight=10.,beta=1e6,max_weight=5.,ref_weight=.1)
        s2,_=train_step_awbc(s2,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=0.,delta_weight=10.,beta=1e6,max_weight=5.,ref_weight=.1)
        for x,y in zip(jax.tree_util.tree_leaves(s1.actor_params),jax.tree_util.tree_leaves(s2.actor_params)):np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)

    def test_precomputed_advantage_matches_q_rule_and_bypasses_the_critic(self):
        q_data=jnp.array([.6,.2,.9,.5]);q_ref=jnp.array([.5,.9,.1,.5]);success=jnp.array([1,0,0,1]);human=jnp.array([0,1,0,0])
        for x,y in zip(awbc_weights(q_data,q_ref,success,human,beta=.1,max_weight=5.),awbc_weights_from_advantage(q_data-q_ref,success,human,beta=.1,max_weight=5.)):
            np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-6)
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(13));batch=dict(small_batch(4),awbc_adv=jnp.array([.05,-.3,.2,-.02],jnp.float32))
        scaled=state.replace(critic_params=jax.tree_util.tree_map(lambda x:x*3.,state.critic_params))
        runs=[]
        for s in (state,scaled):
            for _ in range(2):s,m=train_step_awbc(s,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=0.,delta_weight=10.,beta=.1,max_weight=5.,ref_weight=.1)
            runs.append((s,m))
        for x,y in zip(jax.tree_util.tree_leaves(runs[0][0].actor_params),jax.tree_util.tree_leaves(runs[1][0].actor_params)):np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)
        # rows: success policy, human, failure policy, success policy -> weights exp(.5), 1, 1, exp(-.2)
        self.assertAlmostEqual(float(runs[0][1]['awbc_weight_mean']),float((np.exp(.5)+1+1+np.exp(-.2))/4),places=5)
        self.assertAlmostEqual(float(runs[0][1]['awbc_adv_mean']),float(np.mean([.05,-.3,.2,-.02])),places=6)

if __name__=='__main__':unittest.main()
