"""Conservative physical Stack diagnostics. Does not change rewards or PPO.

check_stack(env) checks the current frame. StackStabilityTracker requires a
continuous sequence of observed passing frames. Supports StackPoolEnv or the
underlying two-box robosuite Stack environment; thresholds are explicit.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class StackThresholds:
    horizontal_tolerance_m: float = .010
    surface_gap_tolerance_m: float = .004
    max_tilt_degrees: float = 10.
    max_linear_speed_m_s: float = .020
    max_angular_speed_rad_s: float = .25
    max_relative_speed_m_s: float = .020
    table_edge_margin_m: float = .002
    max_penetration_m: float = .005
    max_support_normal_tilt_degrees: float = 30.

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in vars(self).values()):
            raise ValueError('All stack thresholds must be finite and positive')
        if max(self.max_tilt_degrees, self.max_support_normal_tilt_degrees) >= 90:
            raise ValueError('Tilt tolerances must be less than 90 degrees')


SIGNS = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
DEFAULT_THRESHOLDS = StackThresholds()


def check_stack(env, thresholds=DEFAULT_THRESHOLDS):
    """Return current-frame conditions, failures and measurements; not a latch.

    Contact alone is insufficient: require a released, aligned, level, slow
    red-on-green arrangement with green supported on the table. Gripper/robot
    clearance is contact-based, not inferred from the commanded action.
    """
    env = getattr(env, 'sim_env', env)
    model, data = env.sim.model, env.sim.data
    a, b = (data.body_xpos[index].copy() for index in (env.cubeA_body_id,env.cubeB_body_id))
    ra, rb = (data.body_xmat[index].reshape(3,3).copy() for index in (env.cubeA_body_id,env.cubeB_body_id))
    av, bv = (np.asarray(data.get_body_xvelp(obj.root_body)) for obj in (env.cubeA,env.cubeB))
    aw, bw = (np.asarray(data.get_body_xvelr(obj.root_body)) for obj in (env.cubeA,env.cubeB))
    if not all(np.isfinite(x).all() for x in (a,b,ra,rb,av,bv,aw,bw,data.qpos,data.qvel)):
        return dict(is_stacked=False,conditions={'finite':False},failed_conditions=['finite'],metrics={})
    sa, sb = np.asarray(env.cubeA.size), np.asarray(env.cubeB.size)
    # Account for actual orientation, not just upright body-center heights.
    a_bottom = a[2]-np.abs(ra[2]).dot(sa)
    b_top = b[2]+np.abs(rb[2]).dot(sb)
    b_bottom = b[2]-np.abs(rb[2]).dot(sb)
    table_id = model.geom_name2id('table_collision')
    table_pos = data.geom_xpos[table_id]
    table_rot = data.geom_xmat[table_id].reshape(3,3)
    table_size = model.geom_size[table_id]
    table_top = table_pos[2]+np.abs(table_rot[2]).dot(table_size)
    inside = []
    for pos,rot,size in ((a,ra,sa),(b,rb,sb)):
        corners = (SIGNS*size) @ rot.T + pos
        local = (corners-table_pos) @ table_rot
        inside.append(bool(np.all(np.abs(local[:,:2]) <= table_size[:2]-thresholds.table_edge_margin_m)))
    ageoms = {model.geom_name2id(name) for name in env.cubeA.contact_geoms}
    bgeoms = {model.geom_name2id(name) for name in env.cubeB.contact_geoms}
    robot_geoms = set()
    for robot in env.robots:
        for obj in [robot.robot_model, *robot.gripper.values()]:
            robot_geoms.update(model.geom_name2id(name) for name in obj.contact_geoms)
    contact_ab = support_ab = support_b_table = robot_contact = a_table_contact = False
    penetration = 0.
    normal_limit = math.cos(math.radians(thresholds.max_support_normal_tilt_degrees))
    for contact in data.contact[:data.ncon]:
        g1,g2 = int(contact.geom1),int(contact.geom2)
        pair = {g1,g2}
        if pair & (ageoms|bgeoms):
            penetration = max(penetration, -float(contact.dist))
        # Ignore positive-distance contact candidates for actual support.
        touching = contact.dist <= 0
        normal_z = abs(float(np.asarray(contact.frame).reshape(3,3)[0,2]))
        if (g1 in ageoms and g2 in bgeoms) or (g2 in ageoms and g1 in bgeoms):
            contact_ab |= touching
            support_ab |= touching and normal_z >= normal_limit
        if table_id in pair and pair & bgeoms:
            support_b_table |= touching and normal_z >= normal_limit
        if table_id in pair and pair & ageoms:
            a_table_contact |= touching
        if pair & robot_geoms and pair & (ageoms|bgeoms):
            robot_contact |= touching
    xy_error = float(np.linalg.norm(a[:2]-b[:2]))
    gap = float(a_bottom-b_top)
    level_limit = math.cos(math.radians(thresholds.max_tilt_degrees))
    # Cubes can rest on any face; a 90-degree cube rotation is not a failure.
    level = max(abs(ra[2])) >= level_limit and max(abs(rb[2])) >= level_limit
    linear = [float(np.linalg.norm(v)) for v in (av,bv)]
    angular = [float(np.linalg.norm(v)) for v in (aw,bw)]
    relative = float(np.linalg.norm(av-bv))
    conditions = dict(finite=True,
        cube_contact=bool(contact_ab), upward_cube_support=bool(support_ab),
        centered=xy_error <= thresholds.horizontal_tolerance_m,
        correct_height=a[2]>b[2] and abs(gap)<=thresholds.surface_gap_tolerance_m,
        faces_level=bool(level), robot_released=not robot_contact,
        green_supported_by_table=bool(support_b_table) and abs(b_bottom-table_top)<=thresholds.surface_gap_tolerance_m,
        both_over_table=all(inside), red_clear_of_table=not a_table_contact,
        slow_translation=max(linear)<=thresholds.max_linear_speed_m_s,
        slow_rotation=max(angular)<=thresholds.max_angular_speed_rad_s,
        slow_relative_motion=relative<=thresholds.max_relative_speed_m_s,
        acceptable_penetration=penetration<=thresholds.max_penetration_m)
    conditions = {key:bool(value) for key,value in conditions.items()}
    values = dict(horizontal_error_m=xy_error,red_bottom_to_green_top_m=gap,
        green_bottom_to_table_m=float(b_bottom-table_top),
        linear_speeds_m_s=linear,angular_speeds_rad_s=angular,
        relative_speed_m_s=relative,max_penetration_m=penetration,
        red_position_m=a.tolist(),green_position_m=b.tolist())
    return dict(is_stacked=all(conditions.values()),conditions=conditions,
                failed_conditions=[key for key,value in conditions.items() if not value],metrics=values)


class StackStabilityTracker:
    """Call reset at episode start, then update once after each control step.

    stable_now means the current passing streak has lasted long enough.
    stable_ever is latched, so report BOTH it and stable_now at episode end.
    Duration is based on consecutive control observations, not every physics substep.
    """
    def __init__(self, required_seconds=1., control_freq=20., thresholds=DEFAULT_THRESHOLDS):
        if not all(math.isfinite(x) and x>0 for x in (required_seconds,control_freq)):
            raise ValueError('Duration and control frequency must be positive')
        self.required_steps = math.ceil(required_seconds*control_freq)
        self.control_freq = control_freq
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        self.streak = self.longest = 0
        self.ever = False
        self.last_step = None
        self.last_report = None

    def observe(self, report, step):
        """Sequence bookkeeping, also usable with recorded per-step check reports."""
        if type(step) is not int or step<0:raise ValueError('Invalid control step')
        if self.last_step is not None:
            if step==self.last_step:return self.last_report
            if step<self.last_step:raise ValueError('Reset tracker at the start of a new episode')
            if step!=self.last_step+1:self.streak=0
        self.streak = self.streak+1 if report['is_stacked'] and step>0 else 0
        self.longest = max(self.longest,self.streak)
        now = self.streak>=self.required_steps
        self.ever |= now
        self.last_step=step
        self.last_report=dict(report,stable_now=now,stable_ever=self.ever,
            current_stable_seconds=self.streak/self.control_freq,
            longest_stable_seconds=self.longest/self.control_freq,
            required_steps=self.required_steps)
        return self.last_report

    def update(self, env):
        raw=getattr(env,'sim_env',env)
        if not math.isclose(raw.control_freq,self.control_freq):
            raise ValueError('Tracker frequency must match environment control frequency')
        return self.observe(check_stack(raw,self.thresholds),int(raw.timestep))


def main():
    """Validate the diagnostic against controlled positive/negative states."""
    import argparse
    import json
    from dataclasses import asdict
    from datetime import datetime
    from pathlib import Path
    from uniform.stack_pool_env import StackPoolEnv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    args=parser.parse_args()
    output=Path(__file__).parent.parent/'stack_check_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    result=dict(status='running',thresholds=asdict(DEFAULT_THRESHOLDS),cases={},no_training=True)
    env=StackPoolEnv(args.pool)
    try:
        env.reset(options={'configuration_id':3})
        held=check_stack(env)
        assert not held['is_stacked'] and not held['conditions']['robot_released']
        result['cases']['held_above_target']=held
        tracker=StackStabilityTracker()
        observed_unstable_impact=False
        for _ in range(150):
            env.step(np.r_[np.zeros(6),-1.])
            report=tracker.update(env)
            if report['conditions']['cube_contact'] and not report['conditions']['slow_translation']:
                observed_unstable_impact=True
        assert report['stable_now'] and report['stable_ever']
        positive=check_stack(env)
        result['cases']['scripted_release_settled']=report
        print('PASS: held cube rejected; physically released stable stack accepted',flush=True)
        raw=env.sim_env
        qpos=raw.sim.data.qpos.copy();qvel=raw.sim.data.qvel.copy()
        model=raw.sim.model
        joint_a=model.joint_name2id(raw.cubeA.joints[0])
        a_pos=int(model.jnt_qposadr[joint_a]);a_vel=int(model.jnt_dofadr[joint_a])
        def restore_positive():
            raw.sim.data.qpos[:]=qpos
            raw.sim.data.qvel[:]=qvel
            raw.sim.forward()
        # Deliberately perturbed negative fixtures test the checker, not the
        # validity of training-pool construction; none enter a training pool.
        restore_positive()
        for obj in (raw.cubeA,raw.cubeB):
            joint=model.joint_name2id(obj.joints[0])
            raw.sim.data.qpos[int(model.jnt_qposadr[joint])]+=1.
        raw.sim.forward()
        off_table=check_stack(raw)
        assert not off_table['is_stacked'] and not off_table['conditions']['green_supported_by_table']
        assert not off_table['conditions']['both_over_table']
        result['cases']['unsupported_off_table']=dict(off_table,stock_success=bool(raw._check_success()))
        restore_positive()
        raw.sim.data.qpos[a_pos:a_pos+3]=raw.sim.data.body_xpos[raw.cubeB_body_id]+[.0449,0,0]
        raw.sim.forward()
        side=check_stack(raw)
        assert side['conditions']['cube_contact'] and not side['conditions']['correct_height']
        assert not side['conditions']['upward_cube_support'] and not side['is_stacked']
        result['cases']['side_contact']=side
        restore_positive()
        raw.sim.data.qvel[a_vel]=.2
        raw.sim.forward()
        moving=check_stack(raw)
        assert not moving['is_stacked'] and not moving['conditions']['slow_translation']
        result['cases']['moving_red_cube']=moving
        restore_positive()
        raw.sim.data.qvel[a_vel+3]=.8
        raw.sim.forward()
        rotating=check_stack(raw)
        assert not rotating['is_stacked'] and not rotating['conditions']['slow_rotation']
        result['cases']['rotating_red_cube']=rotating
        restore_positive()
        angle=math.radians(25)
        raw.sim.data.qpos[a_pos+3:a_pos+7]=[math.cos(angle/2),math.sin(angle/2),0,0]
        raw.sim.forward()
        tilted=check_stack(raw)
        assert not tilted['is_stacked'] and not tilted['conditions']['faces_level']
        result['cases']['tilted_red_cube']=tilted
        print('PASS: off-table, side contact, fast motion, rotation and tilt rejected',flush=True)
        sequence=StackStabilityTracker()
        for step in range(1,20):assert not sequence.observe(positive,step)['stable_now']
        assert sequence.observe(positive,19)['current_stable_seconds']==.95
        assert not sequence.observe(side,20)['stable_now']
        for step in range(21,41):last=sequence.observe(positive,step)
        assert last['stable_now'] and last['current_stable_seconds']==1.
        last=sequence.observe(side,41)
        assert last['stable_ever'] and not last['stable_now']
        skipped=StackStabilityTracker()
        skipped.observe(positive,1);assert skipped.observe(positive,3)['current_stable_seconds']==.05
        result.update(status='passed',tracker_tests='20 consecutive checks; duplicate/gap safety; collapse clears stable_now',
                      unstable_impact_observed=observed_unstable_impact)
        print('PASS: one-second streak, reset on failure, duplicate/gap protection and later collapse tracking',flush=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        env.close()
        (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Report:',output/'validation.json',flush=True)


if __name__=='__main__':main()
