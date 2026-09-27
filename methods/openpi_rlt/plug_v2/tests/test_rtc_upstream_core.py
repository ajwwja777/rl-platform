import unittest,json
from pathlib import Path
import numpy as np,torch
from methods.openpi_rlt.plug_v2 import rtc_upstream_core as rtc
from methods.openpi_rlt.plug_v2.rtc_upstream_data import rtc_schedule, corrected_prefix, link_censored_rows
from methods.openpi_rlt.plug_v2.rtc_upstream_train import burnin_stop_step,resolve_online_weights,resolved_delta_weight,staged_actor_loss_weights
import jax,jax.numpy as jnp
from rlt_online_rl import trainer as upstream
from rlt_online_rl.config import RLTOnlineRLConfig

class CoreTest(unittest.TestCase):
    def test_rtc_schedule_matches_deployed_decision_clock_and_terminal(self):
        rows=rtc_schedule(48,terminal=True)
        self.assertEqual([(r['offset'],r['delay']) for r in rows],[(8,0),(12,6),(22,6),(32,6)])
        self.assertEqual([r['duration'] for r in rows],[4,10,10,16])
        self.assertEqual([r['done'] for r in rows],[False,False,False,True])
        self.assertEqual(rows[-1]['action_slice'],(38,48))
        self.assertEqual(rows[-1]['reward_index'],15)

    def test_rtc_schedule_never_invents_terminal_padding(self):
        rows=rtc_schedule(48,terminal=False)
        self.assertEqual([(r['offset'],r['delay']) for r in rows],[(0,0),(4,6),(14,6),(24,6)])
        self.assertFalse(any(r['done'] for r in rows))
        self.assertFalse(rows[-1]['td_valid'])
        self.assertEqual(rows[-1]['action_slice'],(30,40))

    def test_corrected_prefix_keeps_active_right_and_latches_passive_commands(self):
        prefix=np.arange(84,dtype=np.float32).reshape(6,14)
        passive=np.arange(14,dtype=np.float32)+1000
        out=corrected_prefix(prefix,passive)
        np.testing.assert_array_equal(out[:,7:13],prefix[:,7:13])
        np.testing.assert_array_equal(out[:,:7],np.broadcast_to(passive[:7],(6,7)))
        np.testing.assert_array_equal(out[:,13],passive[13])

    def test_short_handover_gap_becomes_factual_smdp_link(self):
        times=np.arange(100,dtype=np.float64)/30
        rows=[
            dict(key='policy',next_key=None,frame=38,next_frame=38,td_valid=False,done=False),
            dict(key='human',next_key=None,frame=64,next_frame=64,td_valid=False,done=False),
        ]
        self.assertEqual(link_censored_rows(rows,times,max_gap_sec=1.),1)
        self.assertEqual(rows[0]['next_key'],'human')
        self.assertEqual(rows[0]['next_frame'],64)
        self.assertEqual(rows[0]['duration'],26)
        self.assertTrue(rows[0]['td_valid'])

    def test_long_operator_pause_is_not_linked(self):
        times=np.r_[np.arange(40)/30,np.arange(40,100)/30+2]
        rows=[
            dict(key='policy',next_key=None,frame=38,next_frame=38,td_valid=False,done=False),
            dict(key='human',next_key=None,frame=64,next_frame=64,td_valid=False,done=False),
        ]
        self.assertEqual(link_censored_rows(rows,times,max_gap_sec=1.),0)
        self.assertFalse(rows[0]['td_valid'])

    def test_resume_burnin_is_relative_to_loaded_global_step(self):
        self.assertEqual(burnin_stop_step(14135,1000),15135)
        self.assertEqual(burnin_stop_step(0,1000),1000)

    def test_requested_online_weights_override_resumed_release(self):
        self.assertEqual(resolve_online_weights({'online_bc_weight':5.,'online_q_weight':.1},None,None),(5.,.1))
        self.assertEqual(resolve_online_weights({'online_bc_weight':5.,'online_q_weight':.1},8.,.01),(8.,.01))

    def test_requested_delta_weight_overrides_resumed_release(self):
        self.assertEqual(resolved_delta_weight({'delta_weight':300.},10.),10.)
        self.assertEqual(resolved_delta_weight({'delta_weight':300.},30.),30.)

    def test_q_term_is_disabled_during_actor_bc_burnin(self):
        cfg=RLTOnlineRLConfig(warmup_bc_weight=10.,warmup_q_weight=.1,online_bc_weight=5.,online_q_weight=.2)
        self.assertEqual(staged_actor_loss_weights(cfg,{'global_step':999,'warmup_required_updates':5000},1000),(0.,0.))
        self.assertEqual(staged_actor_loss_weights(cfg,{'global_step':1000,'warmup_required_updates':5000},1000),(10.,.1))

    def test_c10_two_steps_match_original_params_metrics_and_rng(self):
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(42));other=state
        random=np.random.default_rng(12);batch={k:jnp.asarray(random.normal(size=shape),jnp.float32) for k,shape in {
            'z_rl':(4,8),'proprio':(4,7),'action_chunk':(4,10,7),'ref_chunk':(4,10,7),
            'next_z_rl':(4,8),'next_proprio':(4,7),'next_ref_chunk':(4,10,7),'rewards':(4,10)}.items()}
        batch.update(done=jnp.array([False,True,False,True]),source_chunk=jnp.array([[0]*10,[2]*10,[0]*5+[2]*5,[0]*10]))
        extra=dict(batch,duration=jnp.full(4,10),td_valid=jnp.ones(4,bool))
        for step in range(2):
            state,m=upstream.train_step(state,batch,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=.1,delta_weight=10.)
            other,n=rtc.train_step(other,extra,actor=a,critic=q,rl_config=cfg,bc_weight=10.,q_weight=.1,delta_weight=10.)
            for x,y in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(other)):
                np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-6)
            for key in m:np.testing.assert_allclose(m[key],n[key],rtol=2e-5,atol=2e-6)
        self.assertIs(upstream.train_step.__wrapped__.__globals__['update_critic'],upstream.update_critic)
    def test_critic_burnin_does_not_change_actor(self):
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(7)); initial=state
        random=np.random.default_rng(2)
        batch={k:jnp.asarray(random.normal(size=shape),jnp.float32) for k,shape in {
            'z_rl':(4,8),'proprio':(4,7),'action_chunk':(4,10,7),'ref_chunk':(4,10,7),
            'next_z_rl':(4,8),'next_proprio':(4,7),'next_ref_chunk':(4,10,7),'rewards':(4,10)}.items()}
        batch.update(done=jnp.array([False,True,False,True]),source_chunk=jnp.zeros((4,10),dtype=jnp.uint8),
                     duration=jnp.full(4,10),td_valid=jnp.ones(4,bool))
        for _ in range(2):
            state,m=rtc.train_step(state,batch,actor=a,critic=q,rl_config=cfg,bc_weight=0.,q_weight=0.,delta_weight=10.)
            self.assertEqual(float(m['did_actor_update']),0.)
        for x,y in zip(jax.tree_util.tree_leaves(initial.actor_params),jax.tree_util.tree_leaves(state.actor_params)):
            np.testing.assert_array_equal(x,y)
        self.assertEqual(int(state.actor_version),int(initial.actor_version))
        assert any(not np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(initial.critic_params),jax.tree_util.tree_leaves(state.critic_params)))

    def test_right_anchor_context_roundtrip(self):
        c=np.arange(198,dtype=np.float32).reshape(2,99);p=rtc.rtc_proprio(c)
        np.testing.assert_array_equal(p[:,:7],c[:,7:14]);np.testing.assert_array_equal(rtc.original_context(p),c)
    def test_censored_batch_does_not_update_critic_from_zero_optimizer(self):
        cfg=RLTOnlineRLConfig(z_dim=8,proprio_dim=7,actor_hidden_dim=16,critic_hidden_dim=16)
        state,a,q=upstream.init_train_state(cfg,rng=jax.random.PRNGKey(9))
        b={k:jnp.zeros(shape) for k,shape in {'z_rl':(2,8),'proprio':(2,7),'action_chunk':(2,10,7),'next_z_rl':(2,8),'next_proprio':(2,7),'next_ref_chunk':(2,10,7),'rewards':(2,16)}.items()}
        b.update(duration=jnp.array([3.,16.]),done=jnp.array([False,True]),td_valid=jnp.zeros(2,bool))
        new,m=rtc.update_critic(state,b,a,q,cfg)
        self.assertEqual(float(m['critic_loss']),0.)
        for x,y in zip(jax.tree_util.tree_leaves(state.critic_params),jax.tree_util.tree_leaves(new.critic_params)):np.testing.assert_array_equal(x,y)

if __name__=='__main__':unittest.main()
