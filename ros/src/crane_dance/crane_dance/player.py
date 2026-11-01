"""再生ノード：お手本（振付ファイル）か npz の action を、全体を 1 本の軌道として t0 に 1 回だけ制御器へ送る。

格子ごとの値は /crane_dance/joint_target に出す（記録側が読む）。時刻は整数ナノ秒で扱い、
式の正本は contract_math（grid_index・grid_time_ns）。使い方は repo の README.md。
"""
import hashlib
import json
import os
import signal
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration, Time
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from crane_dance import contract_math, npz_io, plan as plan_mod

NS = contract_math.NS
TARGET_TOPIC = '/crane_dance/joint_target'


def say(text: str) -> None:
    print(text, flush=True)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def duration_msg(seconds_ns: int) -> Duration:
    return Duration(sec=seconds_ns // NS, nanosec=seconds_ns % NS)


class Player(Node):
    def __init__(self) -> None:
        super().__init__('crane_dance_player')
        self.declare_parameter('source', '')
        self.declare_parameter('npz', '')
        self.declare_parameter('record', False)
        self.declare_parameter('contract', '/workspace/contract/contract.json')
        self.declare_parameter('choreography', '/workspace/contract/choreography.json')
        self.declare_parameter('arm_topic', '/crane_plus_arm_controller/joint_trajectory')
        self.declare_parameter('gripper_topic', '/crane_plus_gripper_controller/joint_trajectory')
        self.declare_parameter('arrive_tol', 0.05)
        self.declare_parameter('git_sha', '')
        self.declare_parameter('run_id', '')
        self.plan = None
        self.latest = {}
        self.t0_ns = None
        self.last_k = None
        self.skipped = 0
        self.prev_now = None
        self.finished = False
        self.aborted = False
        self.timer = None
        self.max_err = {}
        self.end_ns = None
        self.sent_full = False

    def p(self, name):
        return self.get_parameter(name).value

    def stop_playback(self) -> None:
        """再生の timer を止める。中断のときは、止める指令を送る前に必ず呼ぶ。"""
        self.aborted = True
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = None

    def halt(self) -> bool:
        """止める共通の道（Ctrl+C・SIGTERM・例外・時刻の後戻りなどの異常終了）。

        先に再生の timer を止め、そのあと今の位置を 1 点だけ持つ軌道（0.1 s・stamp 0）で、
        制御器に残った将来の軌道を置き換える。送れたら True。
        """
        self.stop_playback()
        if self.plan is None or not rclpy.ok():
            return False
        if all(j in self.latest for j in self.joint_names):
            here = [self.latest[j] for j in self.joint_names]
        else:
            here = self.plan.initial()
        self.send_hold(here, NS // 10)
        self.spin_for(0.3)
        return True

    # ---- 準備
    def setup_io(self, contract: dict) -> None:
        ctl = contract['controller']
        self.joint_names = contract['joint_names']
        self.arm_idx = [self.joint_names.index(j) for j in ctl['arm_controller_joints']]
        self.arm_names = list(ctl['arm_controller_joints'])
        self.grip_idx = [self.joint_names.index(j) for j in ctl['gripper_controller_joints']]
        self.grip_names = list(ctl['gripper_controller_joints'])
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.VOLATILE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.pub_target = self.create_publisher(JointState, TARGET_TOPIC, qos)
        self.pub_arm = self.create_publisher(JointTrajectory, self.p('arm_topic'), 10)
        self.pub_grip = self.create_publisher(JointTrajectory, self.p('gripper_topic'), 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint_states, 10)

    def on_joint_states(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.latest[name] = pos
        # 追従誤差：t0 以降、その時刻の格子点の目標との差を関節ごとの最大で覚える
        if self.t0_ns is None or self.plan is None or self.finished or self.aborted:
            return
        stamp = msg.header.stamp.sec * NS + msg.header.stamp.nanosec
        if stamp <= 0:
            stamp = self.get_clock().now().nanoseconds
        k = contract_math.grid_index(stamp, self.t0_ns, self.plan.fps)
        if not 0 <= k < self.plan.n:
            return
        want = self.plan.target(k)
        for name, w in zip(self.joint_names, want):
            if name in self.latest:
                err = abs(self.latest[name] - w)
                if err > self.max_err.get(name, 0.0):
                    self.max_err[name] = err

    # ---- 送る
    def send_full(self, contract: dict, points: list, t0_ns: int) -> int:
        """全体を 1 本の軌道として腕と手に送る（どちらも header.stamp = t0）。送った点の数を返す。"""
        cp = plan_mod.controller_points(contract, points)
        stamp = Time(sec=t0_ns // NS, nanosec=t0_ns % NS)
        for pub, key in ((self.pub_arm, 'arm'), (self.pub_grip, 'gripper')):
            names, pts = cp[key]
            msg = JointTrajectory()
            msg.header.stamp = stamp
            msg.joint_names = list(names)
            for t_ns, vals in pts:
                pt = JointTrajectoryPoint()
                pt.positions = [float(v) for v in vals]
                pt.time_from_start = duration_msg(t_ns)
                msg.points.append(pt)
            pub.publish(msg)
        return len(cp['arm'][1])

    def send_hold(self, positions: list, tfs_ns: int) -> None:
        """1 点だけの軌道（header.stamp は 0＝すぐ）を腕と手に送る。初期姿勢へ行く・中断で止めるのに使う。"""
        for pub, names, idx in ((self.pub_arm, self.arm_names, self.arm_idx),
                                (self.pub_grip, self.grip_names, self.grip_idx)):
            msg = JointTrajectory()
            msg.joint_names = list(names)
            pt = JointTrajectoryPoint()
            pt.positions = [float(positions[i]) for i in idx]
            pt.time_from_start = duration_msg(tfs_ns)
            msg.points = [pt]
            pub.publish(msg)

    def send_target(self, positions: list, stamp_ns: int) -> None:
        msg = JointState()
        msg.header.stamp = Time(sec=stamp_ns // NS, nanosec=stamp_ns % NS)
        msg.name = list(self.joint_names)
        msg.position = [float(x) for x in positions]
        self.pub_target.publish(msg)

    # ---- 待つ（壁時計）
    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_until(self, cond, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and rclpy.ok():
            if cond():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(cond())

    def max_diff(self, want: list) -> float:
        worst = 0.0
        for name, w in zip(self.joint_names, want):
            if name not in self.latest:
                return float('inf')
            worst = max(worst, abs(self.latest[name] - w))
        return worst

    # ---- 再生の timer（シム時刻）
    def on_timer(self) -> None:
        if self.finished or self.aborted:
            return
        now = self.get_clock().now().nanoseconds
        if self.prev_now is not None and now < self.prev_now:
            say('シム時刻が戻った → このエピソードは無効。Gazebo をやり直してからもう一度')
            self.stop_playback()
            return
        self.prev_now = now
        fps = self.plan.fps
        k = contract_math.grid_index(now, self.t0_ns, fps)
        if k < 0:
            return
        k = min(k, self.plan.n - 1)
        if self.last_k is not None and k <= self.last_k:
            return
        self.skipped += k if self.last_k is None else k - self.last_k - 1
        self.last_k = k
        self.send_target(self.plan.target(k), contract_math.grid_time_ns(k, self.t0_ns, fps))
        if k in self.plan.bar_heads:
            say(f'小節 {self.plan.bar_heads[k]}（k={k}）')
        if k >= self.plan.n - 1:
            self.finished = True
            self.timer.cancel()


def load_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run(node: Player) -> int:
    source = node.p('source')
    if not source:
        say('source が指定されていない（choreography か npz）→ -p source:=choreography か -p source:=npz を付ける')
        return 2
    if source not in ('choreography', 'npz'):
        say(f'source {source!r} は知らない値 → choreography か npz を使う')
        return 2
    npz_path = node.p('npz')
    if source == 'npz' and not npz_path:
        say('source:=npz なのに npz のパスが無い → -p npz:=<コンテナ内のパス> を付ける')
        return 2
    try:
        contract = load_json(node.p('contract'))
        choreo = load_json(node.p('choreography'))
        choreo_sha = sha256_of(node.p('choreography'))
        if source == 'choreography':
            plan = plan_mod.from_choreography(contract, choreo)
        else:
            # npz は再生の前に契約と照合する（初期姿勢への指令より前に拒む）
            plan = plan_mod.from_npz(contract, choreo, npz_io.read_npz(npz_path),
                                     os.path.basename(npz_path), choreo_sha256=choreo_sha)
    except (OSError, ValueError, KeyError, IndexError) as e:
        say(f'計画を作れない（{e}）→ 入力のファイルと契約を確かめてからもう一度')
        return 1
    node.plan = plan
    node.setup_io(contract)
    fps = plan.fps
    say(f'source = {source}')
    say(f'kind = {plan.kind}')
    if source == 'npz':
        say(f'npz = {npz_path}（sha256 {sha256_of(npz_path)}）')
    say(f'choreography sha256 = {choreo_sha}')
    for w in plan.warnings:
        say(f'警告：{w}')
    say(f'git_sha = {node.p("git_sha")}')
    if node.p('run_id'):
        say(f'run_id = {node.p("run_id")}')
    say(f'格子の周期 = {fps} Hz・本数 N = {plan.n}')
    say(f'制御器への渡し方 = 全体を 1 本の軌道（N 点＋戻り 1 点）・header.stamp = t0')

    # 1. 制御器の購読者を待つ
    for topic, pub in ((node.p('arm_topic'), node.pub_arm), (node.p('gripper_topic'), node.pub_grip)):
        if not node.wait_until(lambda pub=pub: pub.get_subscription_count() >= 1, 60.0):
            say(f'制御器が立ち上がっていない（{topic}）→ Gazebo が動いているか確かめてからもう一度')
            return 1

    # 2. 初期姿勢へ
    init = plan.initial()
    node.send_hold(init, 2 * NS)
    tol = node.p('arrive_tol')
    if not node.wait_until(lambda: node.max_diff(init) <= tol, 15.0):
        diff = node.max_diff(init)
        say(f'初期姿勢に着かない（最大の差 {diff:.3f} rad）→ Gazebo と制御器の状態を確かめてからもう一度')
        return 1

    # 3. 記録側を待つ
    if node.p('record'):
        def recorder_seen() -> bool:
            return any('rosbag2' in info.node_name
                       for info in node.get_subscriptions_info_by_topic(TARGET_TOPIC))
        if not node.wait_until(recorder_seen, 60.0):
            say('記録側（rosbag2）が /crane_dance/joint_target を購読していない → ros2 bag record を先に動かしてからもう一度')
            return 1

    # 4. t0 を決めて、全体を 1 本の軌道として送る
    if not node.wait_until(lambda: node.get_clock().now().nanoseconds > 0, 60.0):
        say('シム時刻が進まない → Gazebo が動いているか確かめてからもう一度')
        return 1
    t0 = node.get_clock().now().nanoseconds + NS // 2
    node.t0_ns = t0
    say(f't0 = {t0}（シム時刻）・壁時計 {time.strftime("%Y-%m-%dT%H:%M:%S%z")}')
    n_pts = node.send_full(contract, plan_mod.full_trajectory(plan), t0)
    node.sent_full = True
    say(f'軌道を送った（腕 {n_pts} 点・手 {n_pts} 点）')

    # 5. 格子ごとの joint_target（timer・シム時刻）
    node.end_ns = t0 + (plan.n - 1) * (NS // fps) + 3 * NS
    node.timer = node.create_timer(0.005, node.on_timer)
    while rclpy.ok() and not node.finished and not node.aborted:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.aborted:
        return 1
    if not node.finished:
        say('再生の途中で ROS が終わった')
        return 1

    # 6. 戻りの点に着くまで待って終わる
    while rclpy.ok() and node.get_clock().now().nanoseconds <= node.end_ns:
        rclpy.spin_once(node, timeout_sec=0.05)
    say(f'終了（飛んだ格子 {node.skipped} 本）')
    names = ['j1', 'j2', 'j3', 'j4', 'hand']
    errs = [node.max_err.get(j) for j in node.joint_names]
    say('追従誤差の最大（rad）：' + ' '.join(
        f'{n}={"なし" if e is None else format(e, ".3f")}' for n, e in zip(names, errs)))
    say(f'終わりの姿勢と初期姿勢の差の最大 = {node.max_diff(init):.3f} rad')
    return 0 if rclpy.ok() else 1


class Terminated(KeyboardInterrupt):
    """SIGTERM を SIGINT と同じ道で扱うための例外。"""


def _on_term(signum, frame):
    raise Terminated()


def main(args=None) -> None:
    rclpy.init(args=args)
    # 既定の SIGINT 処理に戻し、SIGTERM も同じ道にする（context を閉じず例外にして、止める前に指令を送れるようにする）
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _on_term)
    node = Player()
    code = 1
    try:
        code = run(node)
        if code != 0 and node.sent_full:
            # 全体の軌道を送ったあとの異常終了（時刻の後戻りなど）も、Ctrl+C と同じ道で止める
            try:
                sent = node.halt()
            except Exception:
                sent = False
            say('異常終了 → ' + ('今の位置で止める指令を送って終わる' if sent
                                  else '止める指令を送れなかった（ROS が終わっている）'))
    except BaseException as e:  # SIGINT・SIGTERM・例外
        is_term = isinstance(e, Terminated)
        # 先に再生の timer を止める。そのあと、今の位置で止める 1 点の軌道を送り、制御器に残った将来の軌道を取り消す
        try:
            node.halt()
        except Exception:
            node.stop_playback()
        if isinstance(e, KeyboardInterrupt):
            say(f'中断（{"SIGTERM" if is_term else "Ctrl+C"}）→ 今の位置で止める指令を送って終わる')
            code = 143 if is_term else 130
        else:
            say(f'中断（例外 {type(e).__name__}: {e}）→ 今の位置で止める指令を送って終わる')
            code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
