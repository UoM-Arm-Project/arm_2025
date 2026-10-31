"""poker_table_console: type table events instead of using buttons / cameras.

Useful for bring-up before the cameras work, and as an operator override.
Overrides answer the game's camera reads instead of the vision_server.

Commands
    d | done                  current player finished (= the turn button)
    f | fold                  current player folds
    n | next                  start the next hand
    act <seat|-> <fold|check|call|raise> [raise_to]
                              apply an action directly (operator override)
    cards <zone> [c1 c2 ...]  next read of that zone returns these cards
                              (board, bot_hole, seat_3 ...); no cards clears it
    pot <value> | pot off     pot reads return this value / use the pot camera again
    add <chips>               add chips to the fake pot (a player bets)
    s | state                 print the latest /poker/state
    q | quit
"""
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from poker_interfaces.msg import GameState, PlayerInput


class TableConsole(Node):
    def __init__(self):
        super().__init__("poker_table_console")
        self.input_pub = self.create_publisher(PlayerInput, "/poker/input", 10)
        self.create_subscription(GameState, "/poker/state", self._on_state, 10)
        self.create_subscription(String, "/poker/announce", lambda m: print(f"  >> {m.data}"), 10)
        self.fake_pot = None
        self.state = None

    def _on_state(self, msg):
        self.state = msg

    def _send_pot(self):
        self.send(PlayerInput.SET_POT, pot_value=-1.0 if self.fake_pot is None else self.fake_pot)

    def send(self, event, **kw):
        m = PlayerInput()
        m.event = event
        m.seat = int(kw.get("seat", -1))
        m.action = kw.get("action", "")
        m.raise_to = int(kw.get("raise_to", 0))
        m.zone = kw.get("zone", "")
        m.cards = list(kw.get("cards", []))
        m.pot_value = float(kw.get("pot_value", 0.0))
        self.input_pub.publish(m)

    def handle_line(self, line: str) -> bool:
        parts = line.split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("d", "done"):
            self.send(PlayerInput.TURN_DONE)
        elif cmd in ("f", "fold"):
            self.send(PlayerInput.FOLD)
        elif cmd in ("n", "next"):
            self.send(PlayerInput.NEXT_HAND)
        elif cmd == "act" and len(args) >= 2:
            seat = -1 if args[0] == "-" else int(args[0])
            raise_to = int(args[2]) if len(args) > 2 else 0
            self.send(PlayerInput.MANUAL_ACTION, seat=seat, action=args[1], raise_to=raise_to)
        elif cmd == "cards" and args:
            self.send(PlayerInput.SET_CARDS, zone=args[0], cards=args[1:])
        elif cmd == "pot" and args:
            self.fake_pot = None if args[0] == "off" else float(args[0])
            self._send_pot()
        elif cmd == "add" and args:
            self.fake_pot = (self.fake_pot or 0.0) + float(args[0])
            self._send_pot()
            print(f"  fake pot = {self.fake_pot:.0f}")
        elif cmd in ("s", "state"):
            s = self.state
            if s is None:
                print("  no /poker/state yet")
            else:
                print(f"  hand {s.hand_number} {s.stage}/{s.phase} actor={s.actor} button={s.button} "
                      f"pot={s.pot} board={list(s.board)}\n  stacks={list(s.stacks)} bets={list(s.bets)}"
                      f"\n  waiting for: {s.expect or 'nothing (busy)'} | {s.waiting}")
        else:
            print(__doc__)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = TableConsole()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    print(__doc__)
    try:
        for line in sys.stdin:
            if not node.handle_line(line):
                break
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
