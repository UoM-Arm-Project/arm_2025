"""poker_game_manager: buttons in, on-demand camera reads, bot decisions / dealer moves out.

Subscribes
    /poker/input      poker_interfaces/PlayerInput          (console / dashboard / extra buttons)
    /button_count     std_msgs/Int32                        (existing poker_gpio button, optional)
Service clients (poker_vision's vision_server)
    /vision/read_cards  poker_interfaces/ReadCards          (called only when cards must be known)
    /vision/read_pot    poker_interfaces/ReadPot            (called only after a turn press)
Publishes
    /poker/state      poker_interfaces/GameState            (2 Hz)
    /poker/announce   std_msgs/String                       (instructions: who presses, when)
Action clients
    /poker/bot_action     poker_interfaces/PokerAction      (served by poker_skill_executor)
    /poker/dealer_action  poker_interfaces/DealerAction

All game logic lives in poker_logic.core.GameDirector; this node only converts
messages and runs it on a timer (single-threaded executor, no locks needed).
Service calls are asynchronous, so a slow CPU inference never blocks the node.
"""
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Int32, String

from poker_interfaces.action import DealerAction, PokerAction
from poker_interfaces.msg import GameState, PlayerInput
from poker_interfaces.srv import ReadCards, ReadPot

from ..core.director import Announce, ArmCommand, DirectorConfig, GameDirector, VisionRequest
from ..core.policy import BotConfig, make_policy
from ..core.table import PhysicalTable, TableConfig

BOT_TYPES = {
    "fold": PokerAction.Goal.FOLD,
    "check": PokerAction.Goal.CHECK,
    "call": PokerAction.Goal.CALL,
    "raise": PokerAction.Goal.BET_RAISE,
    "post_blind": PokerAction.Goal.POST_BLIND,
}
DEALER_TYPES = {
    "deal_card": DealerAction.Goal.DEAL_HOLE,
    "burn": DealerAction.Goal.BURN,
    "deal_board": DealerAction.Goal.DEAL_BOARD,
    "reset_deck": DealerAction.Goal.RESET_DECK,
}


class GameManagerNode(Node):
    def __init__(self):
        super().__init__("poker_game_manager")
        p = self._declare_params()

        mode = p["mode"]
        bot_seat = None if mode == "dealer" else int(p["bot_seat"])
        table = PhysicalTable(TableConfig(
            seats=int(p["seats"]), bot_seat=bot_seat, small_blind=int(p["small_blind"]),
            big_blind=int(p["big_blind"]), starting_stack=int(p["starting_stack"]),
            chip_unit=int(p["chip_unit"]), first_button=int(p["first_button"])))
        policy = make_policy(p["policy"], BotConfig(int(p["samples"]), p["range_model"], p["style"]))
        tol = float(p["pot_tolerance"])
        self.director = GameDirector(
            table,
            DirectorConfig(mode=mode, pot_settle_s=float(p["pot_settle_s"]),
                           pot_tolerance=None if tol < 0 else tol,
                           gesture_on_check=bool(p["gesture_on_check"]),
                           verify_bot_chips=bool(p["verify_bot_chips"]),
                           read_retries=int(p["read_retries"])),
            policy)
        self.vision_timeout = float(p["vision_timeout_s"])
        self.cards_client = self.create_client(ReadCards, "/vision/read_cards")
        self.pot_client = self.create_client(ReadPot, "/vision/read_pot")
        self._vision_deadline = None

        self.create_subscription(PlayerInput, "/poker/input", self._on_input, 10)
        self._last_button = None
        if p["use_gpio_button"]:
            self.create_subscription(Int32, "/button_count", self._on_button, 10)
        self.state_pub = self.create_publisher(GameState, "/poker/state", 10)
        self.announce_pub = self.create_publisher(String, "/poker/announce", 10)
        self.bot_client = ActionClient(self, PokerAction, "/poker/bot_action")
        self.dealer_client = ActionClient(self, DealerAction, "/poker/dealer_action")

        self.create_timer(0.1, self._tick)
        self.create_timer(0.5, self._publish_state)
        self.get_logger().info(
            f"poker_game_manager: mode={mode} seats={p['seats']} bot_seat={bot_seat} policy={p['policy']}")

    # ------------------------------------------------------------------ params
    def _declare_params(self):
        defaults = dict(
            mode="player", seats=5, bot_seat=0, first_button=0, small_blind=1, big_blind=2,
            starting_stack=200, chip_unit=1, policy="equity", samples=600, style="balanced",
            range_model="public_actions", pot_settle_s=0.8, pot_tolerance=-1.0,
            gesture_on_check=True, verify_bot_chips=True, read_retries=1, vision_timeout_s=30.0,
            use_gpio_button=True)
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        return {k: self.get_parameter(k).value for k in defaults}

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _now():
        return time.monotonic()

    def _on_button(self, msg: Int32):
        prev, self._last_button = self._last_button, msg.data
        if prev is not None and prev != msg.data:
            self.director.press_turn_done(self._now())

    def _on_input(self, msg: PlayerInput):
        d, now = self.director, self._now()
        try:
            if msg.event == PlayerInput.TURN_DONE:
                d.press_turn_done(now)
            elif msg.event == PlayerInput.FOLD:
                d.press_fold(now)
            elif msg.event == PlayerInput.NEXT_HAND:
                d.request_next_hand()
            elif msg.event == PlayerInput.MANUAL_ACTION:
                ph = d.table.phase()
                seat = ph.seat if msg.seat < 0 else msg.seat
                d.manual_action(seat, msg.action, msg.raise_to if msg.action == "raise" else None)
            elif msg.event == PlayerInput.SET_CARDS:
                d.set_manual_cards(msg.zone, list(msg.cards))
            elif msg.event == PlayerInput.SET_POT:
                d.set_manual_pot(None if msg.pot_value < 0 else float(msg.pot_value))
        except ValueError as e:
            self._announce(Announce(f"Input rejected: {e}", "warn"))

    # ------------------------------------------------------------------ loop
    def _tick(self):
        try:
            self.director.poll(self._now())
        except Exception as e:  # keep the node alive; the operator can override
            self.get_logger().error(f"director error: {e!r}")
            self._announce(Announce(f"Game logic error: {e}", "error"))
        while self.director.outbox:
            item = self.director.outbox.popleft()
            if isinstance(item, ArmCommand):
                self._send_arm(item)
            elif isinstance(item, VisionRequest):
                self._send_vision(item)
            else:
                self._announce(item)
        self._check_vision_timeout()

    def _announce(self, a: Announce):
        # rclpy fixes one severity per call site, so each level needs its own call
        if a.level == "error":
            self.get_logger().error(a.text)
        elif a.level == "warn":
            self.get_logger().warning(a.text)
        else:
            self.get_logger().info(a.text)
        self.announce_pub.publish(String(data=a.text))

    # ------------------------------------------------------------------ camera reads
    def _send_vision(self, req: VisionRequest):
        client = self.cards_client if req.kind == "cards" else self.pot_client
        if not client.service_is_ready():
            self.director.vision_done(req.id, False, "vision_server not running", now=self._now())
            return
        if req.kind == "cards":
            srv = ReadCards.Request()
            srv.zone, srv.expected = req.zone, int(req.expected)
        else:
            srv = ReadPot.Request()
        self._vision_deadline = (req.id, self._now() + self.vision_timeout)
        client.call_async(srv).add_done_callback(lambda f, r=req: self._on_vision(r, f))

    def _on_vision(self, req: VisionRequest, fut):
        self._vision_deadline = None
        try:
            res = fut.result()
        except Exception as e:   # service died mid-call
            self._guard(self.director.vision_done, req.id, False, f"vision call failed: {e}", now=self._now())
            return
        if req.kind == "cards":
            codes = [c.code for c in res.cards]
            self.get_logger().info(f"read {req.zone}: {codes} ({res.inference_ms:.0f} ms) {res.message}")
            self._guard(self.director.vision_done, req.id, bool(res.success), res.message,
                        cards=codes, now=self._now())
        else:
            self.get_logger().info(f"read pot: {res.value:.0f} ({res.inference_ms:.0f} ms) {res.message}")
            self._guard(self.director.vision_done, req.id, bool(res.success), res.message,
                        value=float(res.value), now=self._now())

    def _guard(self, fn, *args, **kwargs):
        """Run a director callback without letting a game-logic error kill the node."""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            self.get_logger().error(f"director error: {e!r}")
            self._announce(Announce(f"Game logic error: {e}", "error"))

    def _check_vision_timeout(self):
        if self._vision_deadline and self._now() > self._vision_deadline[1]:
            req_id, _ = self._vision_deadline
            self._vision_deadline = None
            self.director.vision_done(req_id, False, "vision read timed out", now=self._now())

    # ------------------------------------------------------------------ arm
    def _send_arm(self, cmd: ArmCommand):
        p = cmd.params
        if cmd.skill in ("bet", "check", "fold", "post_blind"):
            kind = p.get("action", cmd.skill) if cmd.skill == "bet" else cmd.skill
            goal = PokerAction.Goal()
            goal.type = BOT_TYPES[kind]
            goal.chips = int(p.get("chips", 0))
            goal.raise_to = int(p.get("raise_to") or 0)
            goal.all_in = bool(p.get("all_in", False))
            client = self.bot_client
        else:
            goal = DealerAction.Goal()
            goal.type = DEALER_TYPES[cmd.skill]
            goal.seat = int(p.get("chair", 0))
            goal.card_index = int(p.get("card_index", p.get("slot", 0)))
            goal.face_up = bool(p.get("face_up", cmd.skill == "deal_board"))
            client = self.dealer_client
        if not client.server_is_ready():
            self._arm_done(cmd, False, "skill executor not running")
            return
        fut = client.send_goal_async(goal)
        fut.add_done_callback(lambda f, c=cmd: self._on_goal_response(c, f))

    def _on_goal_response(self, cmd, fut):
        handle = fut.result()
        if handle is None or not handle.accepted:
            self._arm_done(cmd, False, "goal rejected")
            return
        handle.get_result_async().add_done_callback(lambda f, c=cmd: self._on_result(c, f))

    def _on_result(self, cmd, fut):
        wrapped = fut.result()
        res = wrapped.result if wrapped is not None else None
        ok = bool(res and res.success)
        self._arm_done(cmd, ok, res.message if res else "no result")

    def _arm_done(self, cmd, ok, message):
        self._guard(self.director.arm_finished, cmd.id, ok, message, self._now())

    # ------------------------------------------------------------------ state
    def _publish_state(self):
        d = self.director
        t = d.table
        m = GameState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.mode = d.cfg.mode
        m.hand_number = t.hand_number
        m.waiting = d.waiting
        m.expect = d.expect
        m.arm_busy = d._pending_arm is not None
        m.pot_bias = float(d._bias)
        m.bot_seat = -1 if t.cfg.bot_seat is None else t.cfg.bot_seat
        m.button = t.button
        if t.state is not None:
            ph = t.phase()
            m.stage = t.stage
            m.phase = ph.kind
            m.actor = ph.seat if ph.kind == "bet" and ph.seat is not None else -1
            m.board = t.board
            m.pot = t.pot
            m.stacks = [t.stack_of(c) for c in range(t.cfg.seats)]
            m.bets = [t.bet_of(c) for c in range(t.cfg.seats)]
            m.in_hand = [t.in_hand(c) for c in range(t.cfg.seats)]
        else:
            m.actor = -1
            m.stacks = [t.stacks[c] for c in range(t.cfg.seats)]
        self.state_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = GameManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
