# Table-side input: buttons, console or dashboard.
uint8 TURN_DONE=0      # current player finished (action inferred from the pot); also "next hand"
uint8 FOLD=1           # current player folds explicitly
uint8 NEXT_HAND=2
uint8 MANUAL_ACTION=3  # operator override: seat + action + raise_to
uint8 SET_CARDS=4      # operator override: zone + cards (empty cards clears the override)
uint8 SET_POT=5        # operator override: pot_value used instead of the pot camera (<0 clears it)

uint8 event
int8 seat              # chair for MANUAL_ACTION (-1 = current actor)
string action          # fold | check | call | raise
int32 raise_to         # street total for raise
string zone            # for SET_CARDS
string[] cards         # for SET_CARDS
float32 pot_value      # for SET_POT
