# poker_logic: decision layer for the poker arm

This package is the logic layer that sits between perception (cards, pot, buttons) and the
existing control stack. It is built from `poker_bot_demo_english` (the PokerKit rules engine
plus the Monte Carlo equity bot) and makes four changes to it:

| Demo (heads-up, on screen)                  | `poker_logic` (physical table)                                     |
|---------------------------------------------|--------------------------------------------------------------------|
| 2 players, random deck                       | 2 to 6 chairs (default 5). Cards come from vision or the dealer arm |
| Human clicks Fold / Call / Raise             | One "turn done" press. The action is inferred from the pot change  |
| Bot action is a log line                     | Bot action is a ROS 2 action that becomes `/move_pose` and `/pump_control` steps |
| No dealer                                    | Three modes: `player`, `dealer`, `dealer_player`                   |

Nothing in `poker_control`, `poker_gpio` or the controller had to change.

## Architecture

```mermaid
flowchart LR
  subgraph Perception
    VS[poker_vision<br/>vision_server]
    BTN[poker_gpio button] -->|/button_count| GM
    CON[table_console / dashboard] -->|/poker/input<br/>PlayerInput| GM
  end
  subgraph poker_logic
    GM[game_manager<br/>GameDirector + PokerKit + EquityBot]
    SE[skill_executor<br/>SkillCompiler + arm_layout.yaml]
    GM -->|/poker/bot_action<br/>PokerAction| SE
    GM -->|/poker/dealer_action<br/>DealerAction| SE
  end
  subgraph poker_control existing
    CTRL[poker_controller<br/>IK + LQR]
    GPIO[poker_gpio pump]
  end
  SE -->|/move_pose action| CTRL
  SE -->|/pump_control| GPIO
  CTRL -->|/arm_busy| SE
  GM -->|/poker/state, /poker/announce| UI[display / logs]
  GM -->|/vision/read_cards<br/>/vision/read_pot<br/>services, on demand| VS
```

There are two nodes and a pure-Python core:

- **`game_manager`** (`nodes/game_manager_node.py`) is a thin ROS wrapper around
  `core/director.py`. The director is a state machine. It waits for button presses,
  requests a camera read only when it needs one (`/vision/read_cards` for a zone,
  `/vision/read_pot` after a turn press), keeps the PokerKit hand in sync with the real
  table, asks the policy for a decision on the bot's turn, and emits arm commands. Service
  calls are asynchronous, so a slow CPU inference never blocks the node.
- **`skill_executor`** (`nodes/skill_executor_node.py`) turns `PokerAction` and
  `DealerAction` goals into a sequence of `/move_pose` goals and pump on/off commands. It
  uses the same pattern as `poker_demo/pick_demo_node.py` and waits for `/arm_busy` to
  clear between moves. Run it with `dry_run:=true` to log the motion plan instead of
  moving the arm.
- **`core/`** has no ROS dependency, so it can be unit-tested and simulated on a laptop.

| File | Role |
|---|---|
| `core/table.py` | `PhysicalTable`: PokerKit with cards supplied from outside. Unseen cards are dealt as `??`. Maps physical chairs to engine seats, rotates the button, handles busted players, refunds uncalled bets, and splits pots in whole chips |
| `core/observation.py` | The demo's observation, generalised to N players. It contains the bot's own cards and public information only |
| `core/equity.py` | The demo's equity estimator, extended to multiway pots with a per-opponent raise-weighted range. Uses `treys` (about 18x faster) with a fallback to PokerKit |
| `core/policy.py` | The demo's `EquityBot`. Thresholds are re-centred on the multiway fair share, and bet sizes are rounded to the smallest chip |
| `core/inference.py` | Turns a pot delta into fold, check, call or raise |
| `core/director.py` | `GameDirector` state machine, including the three modes |
| `core/skills.py` | Compiles skills into `MovePose` and `Pump` steps from `arm_layout.yaml` |
| `core/chips.py` | Chooses chips from the trays with an exact, inventory-aware split |
| `sim.py` | Offline table simulator: fake humans and dealer press the button when prompted; camera reads are answered from the true table, with occasional misreads; the arm is instant |

## Modes

| Mode | Who deals | Bot plays | Arm does |
|---|---|---|---|
| `player` | Humans. The bot's two cards are placed face-up in its holder | yes | Bets, calls, checks (tap) and folds (cards to muck) |
| `dealer` | Arm | no | Resets the deck, deals hole cards face-down to each chair, burns, and deals the board face-up |
| `dealer_player` | Arm | yes | Everything above. The bot's own cards are dealt face-up into its holder |

Set the mode with `ros2 launch poker_logic poker_game.launch.py mode:=dealer_player`.

## How a hand flows

Nothing is read until the right person presses the button, and every press moment is
announced once on `/poker/announce` as an instruction saying **who** presses and
**when**. `GameState.expect` says which press is awaited (`hole`, `board`, `turn`,
`showdown`, `next`, `fixed`, or empty while the arm or camera is busy). Presses while
busy are ignored with a warning.

| Moment | Instruction (announced) | On press | Read |
|---|---|---|---|
| Hole cards (player mode) | `DEALER: deal two cards to every player and put the bot's two cards face-up in its holder, then press the button.` | read the holder | `read_cards(bot_hole, 2)` |
| Flop / turn / river (player mode) | `DEALER: burn one card and deal the flop (3 cards) face-up on the board, then press the button.` | read the board, keep the new cards | `read_cards(board, 3/4/5)` |
| A human's turn | `CHAIR 3: your turn (to call put in 2, ...). Then press the button.` | wait `pot_settle_s`, photograph the pot | `read_pot` |
| Showdown | `SHOWDOWN, CHAIR 3: turn your two cards face-up in your seat area, then press the button (fold button = muck).` | read the seat | `read_cards(seat_3, 2)` |
| Hand over | `DEALER: push the pot to the winner and collect the cards, then press the button to start hand N.` | start the next hand | none |

In the dealer modes the arm deals, and the hole-card and board reads fire automatically
when its last card lands. No press is needed unless a read fails.

**A bad read** (wrong count, a card clashing with known cards, a camera error) is retried
automatically `read_retries` times. After that the operator is told what was seen and
asked to fix the cards and press again. At showdown the fold button mucks, except before
an all-in run-out, where hands must be tabled. The bot's own chips are photographed after
they land (`verify_bot_chips`), which also keeps the pot calibration (`pot_bias`) current.

1. **Next hand.** The button moves clockwise. If the bot is in a blind, it posts the
   blind with `PokerAction POST_BLIND`.
2. **Hole cards.** In dealer modes the arm deals two rounds starting at the small blind. The
   bot's cards go face-up into `bot_hole` so the overhead camera can read them. Other
   players' cards enter the engine as unknown `??` cards.
3. **Betting.** PokerKit knows whose turn it is and how much they owe. A human puts chips in
   the middle and presses the button (any change on `/button_count` counts). After
   `pot_settle_s`, one pot photo is compared with the expected pot:

   | Chips added (snapped to `chip_unit`) | Action |
   |---|---|
   | 0, nothing owed | check |
   | 0, chips owed (or the FOLD input) | fold |
   | the amount owed, or the whole stack | call |
   | more than owed | bet or raise to (street bet + chips added) |

   An under-raise, an over-bet or a short call is announced with what to correct.
   On the bot's turn, the director builds an observation, the `EquityBot` decides, and a
   `PokerAction` goal is sent. The engine only applies the action after the arm reports
   success.
4. **Board.** The read must contain exactly the expected number of *new* cards. Previously
   dealt cards that are momentarily hidden are kept, with a warning.
5. **Showdown.** The announcement says who gets the pot.

**Chip values.** The defaults are small stakes, matching the chip model: White 1, Red 5,
Blue 10, blinds 1/2, stacks 200, `chip_unit: 1`. With 1-value chips there is no rounding
margin, so a miscounted chip changes the inferred action. If the pot camera is noisy, use
`shots` voting in poker_vision, or confirm with the console.

The director never blocks. If the arm fails, the game pauses with an error. The operator can
fix the table by hand and press the button to continue as if the move had succeeded.

## Interfaces added to `poker_interfaces`

| Interface | Direction | Purpose |
|---|---|---|
| `srv/ReadCards` | logic → vision | zone + expected count → cards (code, table x/y, confidence), success, message |
| `srv/ReadPot` | logic → vision | → pot value, confidence, chips per colour |
| `msg/CardDetection`, `msg/CardDetectionArray` | vision → logs | one card / one read's cards (also published per read on `/vision/cards`) |
| `msg/PotEstimate` | vision → logs | one pot read (published per read on `/vision/pot`) |
| `msg/PlayerInput` | buttons / UI → logic | `TURN_DONE`, `FOLD`, `NEXT_HAND`, `MANUAL_ACTION`, `SET_CARDS`, `SET_POT` |
| `msg/GameState` | logic → UI | hand, stage, phase, actor, board, pot, stacks, the current instruction (`waiting`) and awaited press (`expect`) |
| `action/PokerAction` | logic → executor | `FOLD`, `CHECK`, `CALL`, `BET_RAISE`, `POST_BLIND` with a chip amount |
| `action/DealerAction` | logic → executor | `DEAL_HOLE`, `BURN`, `DEAL_BOARD`, `RESET_DECK`, `HOME` |

**Vision contract.** Card codes use rank then suit: `A K Q J T 9 … 2` followed by
`h d c s` (for example `Th`). `10h` / `10H` are accepted too. Zones (`board`,
`bot_hole`, `seat_<n>`) are defined in poker_vision's `vision.yaml`. The logic layer only
asks for them by name.

## Running

```bash
# once
pip install -r requirements.txt          # adds pokerkit + treys to the workspace venv
./build.sh && source ./source_ws.sh

# terminal 1: control stack (unchanged)
ros2 launch poker_bringup poker_arm.launch.py mode:=pi_hardware_headless
# terminal 2: on-demand vision (cards + pot services)
ros2 launch poker_vision vision.launch.py
# terminal 3: decision layer + skill executor
ros2 launch poker_logic poker_game.launch.py mode:=player seats:=5 bot_seat:=0
# terminal 4: manual table input (an override, or instead of the cameras)
ros2 run poker_logic table_console
```

**Dry run without an arm or cameras.** Launch poker_logic with `dry_run:=true` and no
vision server. Overrides then answer the camera reads, and `d` is the button:

```
pot 3                           # pot reads return 3 (the 1/2 blinds)
cards bot_hole Ah Kh            # the bot holder read returns these
d                               # DEALER presses: hole cards read
add 2   then  d                 # chair 3 puts in 2 and presses: call
cards board 2c 7h 9s   then  d  # flop dealt, DEALER presses
s                               # print /poker/state (incl. which press is awaited)
```

**Offline simulation (no ROS):**

```bash
ros2 run poker_logic poker_sim --mode dealer_player --seats 5 --hands 50 -v   # or: cd src/poker_logic && python3 -m poker_logic.sim ...
python3 -m pytest src/poker_logic/test
```

## Calibration checklist

1. **`config/arm_layout.yaml`** is in the arm base frame, the same frame `/move_pose` uses.
   Teach each pose with the dashboard's Cartesian panel: home, check tap, the chip trays
   (`base_z` is the height of the last chip, plus the chip thickness), the pot drop point,
   the deck holder, burn, muck, the five board slots, the two bot holder slots, and one
   target per chair. The face-up release reuses the flip motion from `pick_demo`
   (`approach` and `release`), so tune it once per slot height.
2. **Zones** now live in poker_vision's `vision.yaml`, in the ArUco table frame.
3. **`config/game.yaml`** holds blinds, `chip_unit` (the smallest chip value), stacks,
   `samples` (try 300 on a Pi), `pot_settle_s`, `read_retries` and `vision_timeout_s`.
4. Chip trays restart from the `count` values in the layout file. Publish to
   `/poker/tray_refill` (`std_msgs/Empty`) after topping them up.

## Known limits and assumptions

- **Python version.** PokerKit needs Python 3.11 or newer, so use ROS 2 Jazzy on Ubuntu
  24.04. The Humble/22.04 default of Python 3.10 will not import it.
- **One shared pot camera.** Actions are inferred from changes in the total pot, so
  players should put chips straight into the middle and press after each action. Per-seat
  bet zones would make this more robust.
- **Payouts are announced, not physical.** The table pushes the pot and returns uncalled
  bets by hand, including chips owed to the bot. The bot's chip trays are tracked only as
  inventory counts.
- **Bot hole cards.** The bot must be able to see its own hole cards, so the design
  assumes a holder the overhead camera can see but opponents cannot.
- **Policy strength.** The policy is the demo's explainable heuristic generalised to
  multiway pots. It is not GTO, and the multiway threshold scaling is a hand-tuned
  assumption.
