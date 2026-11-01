"""crane_dance.player の止め方の検査（ROS なし：rclpy とメッセージ型を小さな偽物に差し替えて main を動かす）。

見るもの：全体の軌道を送ったあとに止まるとき、通常終了以外はどの道でも
「再生の timer を取り消す → 今の位置で止める 1 点の軌道（0.1 s）を腕と手に送る」こと。
"""
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SRC = str(REPO / "ros" / "src" / "crane_dance")
NS = 10**9
STEP_NS = 10 * 10**6  # 偽の spin 1 回で進むシム時刻（10 ms）


class Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_fake_ros(world):
    """world（dict）に、送った指令・timer・時計の振る舞いを集める偽の rclpy 一式。"""

    def joint_trajectory():
        return Msg(header=Msg(stamp=None), joint_names=[], points=[])

    msgs = {
        "builtin_interfaces.msg": dict(Duration=lambda sec=0, nanosec=0: Msg(sec=sec, nanosec=nanosec),
                                       Time=lambda sec=0, nanosec=0: Msg(sec=sec, nanosec=nanosec)),
        "sensor_msgs.msg": dict(JointState=lambda: Msg(header=Msg(stamp=None), name=[], position=[])),
        "trajectory_msgs.msg": dict(JointTrajectory=joint_trajectory,
                                    JointTrajectoryPoint=lambda: Msg(positions=[], time_from_start=None)),
        "rclpy.qos": dict(QoSProfile=lambda **kw: None,
                          QoSDurabilityPolicy=Msg(VOLATILE=0), QoSHistoryPolicy=Msg(KEEP_LAST=0),
                          QoSReliabilityPolicy=Msg(RELIABLE=0)),
    }
    mods = {}
    for name, attrs in msgs.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
    mods["builtin_interfaces"] = types.ModuleType("builtin_interfaces")
    mods["sensor_msgs"] = types.ModuleType("sensor_msgs")
    mods["trajectory_msgs"] = types.ModuleType("trajectory_msgs")

    class Timer:
        def __init__(self, cb):
            self.cb = cb
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class Pub:
        def __init__(self, topic):
            self.topic = topic

        def publish(self, msg):
            world["sent"].append((self.topic, msg))
            if self.topic.endswith("joint_trajectory") and len(msg.points) == 1:
                # 1 点の指令（初期姿勢・止める指令）は、その位置に着いたことにする
                for n, x in zip(msg.joint_names, msg.points[0].positions):
                    world["pos"][n] = x

        def get_subscription_count(self):
            return 1

    class Node:
        def __init__(self, name):
            self._params = {}
            world["node"] = self

        def declare_parameter(self, name, default):
            self._params[name] = world["params"].get(name, default)

        def get_parameter(self, name):
            return Msg(value=self._params[name])

        def create_publisher(self, typ, topic, qos):
            return Pub(topic)

        def create_subscription(self, typ, topic, cb, qos):
            world["js_cb"] = cb

        def create_timer(self, period, cb):
            t = Timer(cb)
            world["timers"].append(t)
            return t

        def destroy_timer(self, t):
            world["destroyed"].append(t)

        def get_clock(self):
            return Msg(now=lambda: Msg(nanoseconds=world["now"]))

        def get_subscriptions_info_by_topic(self, topic):
            return []

        def destroy_node(self):
            pass

    def spin_once(node, timeout_sec=None):
        world["spins"] += 1
        hook = world.get("hook")
        if hook is not None:
            hook(world)  # 例外（Ctrl+C など）や時刻の後戻りを差し込む
        world["now"] += STEP_NS
        if world.get("js_cb") is not None and world["pos"]:
            names = sorted(world["pos"])
            stamp = Msg(sec=world["now"] // NS, nanosec=world["now"] % NS)
            world["js_cb"](Msg(header=Msg(stamp=stamp), name=names, position=[world["pos"][n] for n in names]))
        for t in list(world["timers"]):
            if not t.cancelled and t not in world["destroyed"]:
                t.cb()

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: world.update(ok=True)
    rclpy.ok = lambda: world["ok"]
    rclpy.shutdown = lambda: world.update(ok=False)
    rclpy.spin_once = spin_once
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node
    mods["rclpy"] = rclpy
    mods["rclpy.node"] = node_mod
    return mods


class PlayerStopTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        choreo = json.loads((REPO / "contract" / "choreography.json").read_text(encoding="utf-8"))
        choreo["song"]["duration_s"] = 0.2  # N = 10（小さい振付）
        self.choreo_path = Path(self._tmp.name) / "choreo.json"
        self.choreo_path.write_text(json.dumps(choreo), encoding="utf-8")

    def run_main(self, hook=None):
        world = dict(sent=[], timers=[], destroyed=[], pos={}, now=NS, spins=0, ok=False, hook=hook,
                     params=dict(source="choreography", contract=str(REPO / "contract" / "contract.json"),
                                 choreography=str(self.choreo_path)))
        mods = make_fake_ros(world)
        with mock.patch.dict(sys.modules, mods), mock.patch.object(sys, "path", [SRC] + sys.path), \
                mock.patch("signal.signal"), mock.patch("builtins.print"):
            for name in [m for m in sys.modules if m == "crane_dance" or m.startswith("crane_dance.")]:
                sys.modules.pop(name)
            player = importlib.import_module("crane_dance.player")
            with self.assertRaises(SystemExit) as cm:
                player.main()
            world["Terminated"] = player.Terminated
        for name in [m for m in sys.modules if m == "crane_dance" or m.startswith("crane_dance.")]:
            sys.modules.pop(name)
        world["code"] = cm.exception.code
        return world

    @staticmethod
    def arm_msgs(world):
        return [m for t, m in world["sent"] if t.endswith("arm_controller/joint_trajectory")]

    @staticmethod
    def grip_msgs(world):
        return [m for t, m in world["sent"] if t.endswith("gripper_controller/joint_trajectory")]

    def assert_halted_after_full(self, world):
        arm, grip = self.arm_msgs(world), self.grip_msgs(world)
        full = [i for i, m in enumerate(arm) if len(m.points) > 1]
        self.assertEqual(len(full), 1, "全体の軌道はちょうど 1 回")
        for msgs in (arm, grip):
            # 初期姿勢（1 点）→ 全体（N＋1 点）→ 止める指令（1 点・0.1 s）の順で、止める指令が最後
            self.assertEqual([len(m.points) for m in msgs], [1, 11, 1])
            last = msgs[-1].points[0].time_from_start
            self.assertEqual((last.sec, last.nanosec), (0, NS // 10))
        self.assertTrue(world["timers"] and all(t.cancelled for t in world["timers"]), "timer を取り消した")

    def test_normal_end_sends_no_stop_command(self):
        w = self.run_main()
        self.assertEqual(w["code"], 0)
        self.assertEqual([len(m.points) for m in self.arm_msgs(w)], [1, 11])
        self.assertEqual([len(m.points) for m in self.grip_msgs(w)], [1, 11])

    def test_time_going_back_after_full_trajectory_halts(self):
        # 全体の軌道を送ったあと（timer ができてから）にシム時刻を 1 秒戻す
        def hook(world):
            if world["timers"] and not world.get("jumped") and world["spins"] % 7 == 0:
                world["jumped"] = True
                world["now"] -= NS
        w = self.run_main(hook)
        self.assertTrue(w.get("jumped"))
        self.assertEqual(w["code"], 1)
        self.assert_halted_after_full(w)

    def test_ctrl_c_and_sigterm_halt(self):
        for code, exc in ((130, KeyboardInterrupt), (143, "Terminated")):
            def hook(world, exc=exc):
                if world["timers"] and not world.get("raised"):
                    world["raised"] = True
                    raise world["Terminated"]() if exc == "Terminated" else exc()
            # Terminated はモジュールを読んでから決まるので、読んだ直後に world に入れる
            w = self.run_main_with_terminated(hook)
            self.assertEqual(w["code"], code, exc)
            self.assert_halted_after_full(w)

    def run_main_with_terminated(self, hook):
        def wrapped(world):
            if "Terminated" not in world:
                world["Terminated"] = sys.modules["crane_dance.player"].Terminated
            hook(world)
        return self.run_main(wrapped)


if __name__ == "__main__":
    unittest.main()
