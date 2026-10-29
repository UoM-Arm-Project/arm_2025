"""poker_game_manager: vision + buttons in, bot decisions / dealer moves out.

Subscribes
    /vision/cards     poker_interfaces/CardDetectionArray   (every overhead frame)
    /vision/pot       poker_interfaces/PotEstimate          (side camera)
    /poker/input      poker_interfaces/PlayerInput          (console / dashboard / extra buttons)
    /button_count     std_msgs/Int32                        (existing poker_gpio button, optional)
Publishes
    /poker/state      poker_interfaces/GameState            (2 Hz and on change)
    /poker/announce   std_msgs/String                       (what the table should hear/see)
Action clients
    /poker/bot_action     poker_interfaces/PokerAction      (served by poker_skill_executor)
    /poker/dealer_action  poker_interfaces/DealerAction

All game logic lives in poker_logic.core.GameDirector; this node only converts
messages and runs it on a timer (single-threaded executor, no locks needed).
"""
import os
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Int32, String

from poker_interfaces.action import DealerAction, PokerAction
from poker_interfaces.msg import CardDetectionArray, GameState, PlayerInput, PotEstimate

from ..core.director import Announce, ArmCommand, DirectorConfig, GameDirector
from ..core.policy import BotConfig, make_policy
from ..core.table import PhysicalTable, TableConfig
from ..core.tracker import CardTracker, PotTracker, Zone, assign_zone

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
                           pot_min_samples=int(p["pot_min_samples"]),
                           pot_tolerance=None if tol < 0 else tol,
                           gesture_on_check=bool(p["gesture_on_check"])),
            policy,
            CardTracker(int(p["card_window"]), float(p["card_ratio"]), float(p["card_min_confidence"])),
            PotTracker(float(p["pot_min_confidence"])))
        self.zones = self._load_zones(p["zones_file"])

        self.create_subscription(CardDetectionArray, "/vision/cards", self._on_cards, 10)
        self.create_subscription(PotEstimate, "/vision/pot", self._on_pot, 10)
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
            mode="player", seats=5, bot_seat=0, first_button=0, small_blind=5, big_blind=10,
            starting_stack=1000, chip_unit=5, policy="equity", samples=600, style="balanced",
            range_model="public_actions", pot_settle_s=0.8, pot_min_samples=3, pot_tolerance=-1.0,
            gesture_on_check=True, card_window=10, card_ratio=0.7, card_min_confidence=0.5,
            pot_min_confidence=0.3, use_gpio_button=True, zones_file="")
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        return {k: self.get_parameter(k).value for k in defaults}

    def _load_zones(self, path):
        if not path:
            path = os.path.join(get_package_share_directory("poker_logic"), "config", "vision_zones.yaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        zones = [Zone(z["name"], z["x_min"], z["x_max"], z["y_min"], z["y_max"]) for z in data.get("zones", [])]
        self.get_logger().info(f"loaded {len(zones)} vision zones from {path}")
        return zones

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _now():
        return time.monotonic()

    def _on_cards(self, msg: CardDetectionArray):
        dets = [(c.code, c.zone or assign_zone(self.zones, c.x, c.y), c.confidence) for c in msg.cards]
        self.director.cards.update(dets)

    def _on_pot(self, msg: PotEstimate):
        self.director.pot.add(self._now(), msg.value, msg.confidence)

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
            else:
                self._announce(item)

    def _announce(self, a: Announce):
        if a.level == "error": 
            self.get_logger().error(a.text)
        elif a.level == "warn": 
            self.get_logger().warning(a.text)
        else:
            self.get_logger().info(a.text)
        self.announce_pub.publish(String(data=a.text))

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
        self.director.arm_finished(cmd.id, ok, message, self._now())

    # ------------------------------------------------------------------ state
    def _publish_state(self):
        d = self.director
        t = d.table
        m = GameState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.mode = d.cfg.mode
        m.hand_number = t.hand_number
        m.waiting = d.waiting
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
