# One card seen by the overhead camera (YOLO detector + ArUco table frame).
string code          # rank+suit, e.g. "Ah", "Td", "2c"
string zone          # optional: board, bot_hole, seat_<n>, muck ("" = let the game manager decide)
float32 x            # table frame, metres
float32 y
float32 confidence   # 0..1
