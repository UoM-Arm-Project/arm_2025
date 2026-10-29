"""poker_skill_executor: poker-level actions -> the existing control interfaces.

Action servers
    /poker/bot_action     poker_interfaces/PokerAction   (fold, check, call, raise, post blind)
    /poker/dealer_action  poker_interfaces/DealerAction  (deal hole, burn, deal board, reset deck)
Uses (unchanged control stack)
    /move_pose            poker_interfaces/MovePose      action client
    /pump_control         std_msgs/Int32                 1 = suction on, 0 = off
    /arm_busy             std_msgs/Bool                  waits for false before each move
Topics
    /poker/tray_refill    std_msgs/Empty                 chip trays refilled to config counts

Coordinates come from config/arm_layout.yaml (arm base frame). Use
``dry_run:=true`` to log the motion plan without moving the arm.
"""
import os
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Int32

from poker_interfaces.action import DealerAction, MovePose as MovePoseAction, PokerAction

from ..core.skills import ArmState, MovePose, Pump, SkillCompiler, Wait, validate_layout

BOT_KIND = {PokerAction.Goal.FOLD: "fold", PokerAction.Goal.CHECK: "check",
            PokerAction.Goal.CALL: "call", PokerAction.Goal.BET_RAISE: "raise",
            PokerAction.Goal.POST_BLIND: "post_blind"}
DEALER_KIND = {DealerAction.Goal.DEAL_HOLE: "deal_hole", DealerAction.Goal.BURN: "burn",
               DealerAction.Goal.DEAL_BOARD: "deal_board", DealerAction.Goal.RESET_DECK: "reset_deck",
               DealerAction.Goal.HOME: "home"}


class SkillExecutorNode(Node):
    def __init__(self):
        super().__init__("poker_skill_executor")
        self.declare_parameter("layout_file", "")
        self.declare_parameter("seats", 5)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("move_timeout_s", 30.0)
        path = self.get_parameter("layout_file").value or os.path.join(
            get_package_share_directory("poker_logic"), "config", "arm_layout.yaml")
        with open(path) as f:
            self.layout = yaml.safe_load(f)
        problems = validate_layout(self.layout, int(self.get_parameter("seats").value))
        if problems:
            raise RuntimeError(f"{path}: " + "; ".join(problems))
        self.compiler = SkillCompiler(self.layout)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.move_timeout = float(self.get_parameter("move_timeout_s").value)
        m = self.layout.get("motion", {})
        self.pump_on_settle = float(m.get("pump_on_settle", 0.5))
        self.pump_off_settle = float(m.get("pump_off_settle", 0.5))

        self.arm = ArmState()
        self._refill()
        self._busy = threading.Lock()
        self._arm_busy = False

        cb = ReentrantCallbackGroup()
        self.move_client = ActionClient(self, MovePoseAction, "/move_pose", callback_group=cb)
        self.pump_pub = self.create_publisher(Int32, "/pump_control", 10)
        self.create_subscription(Bool, "/arm_busy", self._on_arm_busy, 10, callback_group=cb)
        self.create_subscription(Empty, "/poker/tray_refill", lambda _m: self._refill(), 10)
        ActionServer(self, PokerAction, "/poker/bot_action", execute_callback=self._exec_bot,
                     goal_callback=self._goal_cb, cancel_callback=lambda _g: CancelResponse.ACCEPT,
                     callback_group=cb)
        ActionServer(self, DealerAction, "/poker/dealer_action", execute_callback=self._exec_dealer,
                     goal_callback=self._goal_cb, cancel_callback=lambda _g: CancelResponse.ACCEPT,
                     callback_group=cb)
        self.get_logger().info(f"poker_skill_executor ready (layout {path}, dry_run={self.dry_run})")

    # ------------------------------------------------------------------ bookkeeping
    def _refill(self):
        trays = self.layout["chips"]["trays"]
        self.arm = ArmState({int(k): int(v.get("count", 0)) for k, v in trays.items()}, self.arm.deck_count)
        self.get_logger().info(f"chip trays: {self.arm.tray_counts}")

    def _on_arm_busy(self, msg: Bool):
        self._arm_busy = bool(msg.data)

    def _goal_cb(self, _goal):
        return GoalResponse.REJECT if self._busy.locked() else GoalResponse.ACCEPT

    # ------------------------------------------------------------------ actions
    def _exec_bot(self, gh):
        g = gh.request
        kind = BOT_KIND.get(g.type)
        result = PokerAction.Result()
        try:
            steps, new_state, change = self.compiler.bot_action(kind, int(g.chips), self.arm)
        except (ValueError, KeyError) as e:
            gh.abort()
            result.success, result.message = False, f"cannot plan {kind}: {e}"
            return result
        ok, msg = self._run(gh, steps, PokerAction.Feedback, f"{kind} {g.chips}")
        if ok:
            self.arm = new_state
            result.chip_values = [d for d, n in change.items() for _ in range(n)]
            gh.succeed()
        elif gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.abort()
        result.success, result.message = ok, msg
        return result

    def _exec_dealer(self, gh):
        g = gh.request
        kind = DEALER_KIND.get(g.type)
        result = DealerAction.Result()
        try:
            steps, new_state, _ = self.compiler.dealer_action(kind, self.arm, seat=int(g.seat),
                                                              index=int(g.card_index), face_up=bool(g.face_up))
        except (ValueError, KeyError, IndexError) as e:
            gh.abort()
            result.success, result.message = False, f"cannot plan {kind}: {e}"
            return result
        ok, msg = self._run(gh, steps, DealerAction.Feedback, kind)
        if ok:
            self.arm = new_state
            gh.succeed()
        elif gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.abort()
        result.success, result.message = ok, msg
        return result

    # ------------------------------------------------------------------ step runner
    def _run(self, gh, steps, feedback_type, name):
        if not self._busy.acquire(blocking=False):
            return False, "executor busy"
        try:
            self.get_logger().info(f"skill '{name}': {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                if gh.is_cancel_requested:
                    self._pump(False)
                    return False, "canceled"
                fb = feedback_type()
                fb.phase, fb.step, fb.total_steps = step.label, i, len(steps)
                gh.publish_feedback(fb)
                if isinstance(step, MovePose):
                    ok, err = self._move(step)
                    if not ok:
                        self._pump(False)
                        return False, f"step {i} '{step.label}' failed: {err}"
                elif isinstance(step, Pump):
                    self._pump(step.on)
                elif isinstance(step, Wait):
                    time.sleep(step.seconds)
            return True, f"{name} done"
        finally:
            self._busy.release()

    def _pump(self, on: bool):
        if self.dry_run:
            self.get_logger().info(f"[dry-run] pump {'ON' if on else 'OFF'}")
        else:
            self.pump_pub.publish(Int32(data=1 if on else 0))
        time.sleep(self.pump_on_settle if on else self.pump_off_settle)

    def _move(self, s: MovePose):
        if self.dry_run:
            self.get_logger().info(f"[dry-run] {s.label}: x={s.x:.3f} y={s.y:.3f} z={s.z:.3f} "
                                   f"pitch={s.pitch:.2f} roll={s.roll:.2f} t={s.duration}")
            return True, ""
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            return False, "/move_pose server not available"
        deadline = time.monotonic() + 5.0
        while self._arm_busy and time.monotonic() < deadline:   # controller rejects goals mid-move
            time.sleep(0.05)
        goal = MovePoseAction.Goal()
        goal.x, goal.y, goal.z = float(s.x), float(s.y), float(s.z)
        goal.pitch, goal.roll, goal.duration = float(s.pitch), float(s.roll), float(s.duration)
        done = threading.Event()
        box = {}

        def on_result(f):
            box["result"] = f.result()
            done.set()

        def on_goal(f):
            handle = f.result()
            if handle is None or not handle.accepted:
                box["rejected"] = True
                done.set()
                return
            handle.get_result_async().add_done_callback(on_result)

        self.move_client.send_goal_async(goal).add_done_callback(on_goal)
        if not done.wait(s.duration + self.move_timeout):
            return False, "timeout"
        if box.get("rejected"):
            return False, "goal rejected (arm busy?)"
        res = box["result"].result
        return bool(res.success), f"final_error={res.final_error:.4f}"


def main(args=None):
    rclpy.init(args=args)
    node = SkillExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
