# Snapshot of the poker game for dashboards / logging.
std_msgs/Header header
string mode            # player | dealer | dealer_player
uint32 hand_number
string stage           # preflop | flop | turn | river
string phase           # deal_hole | burn | board | bet | showdown | over
int8 actor             # chair whose turn it is (-1 = none)
int8 button
int8 bot_seat          # -1 in dealer mode
string[] board
int32 pot
int32[] stacks         # per chair
int32[] bets           # current street contribution per chair
bool[] in_hand
string waiting         # what the system is waiting for, human readable
bool arm_busy
float32 pot_bias       # vision pot minus engine pot (calibration drift)
