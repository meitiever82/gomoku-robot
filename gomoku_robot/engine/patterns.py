"""
五子棋威胁模式定义

模式编码说明：
- 1: 己方棋子
- 0: 空位
- 模式从高威胁到低威胁排列
- 每个模式包含: (pattern, score, name)
"""

# 评分常量
SCORE_FIVE = 1_000_000       # 五连 (胜)
SCORE_LIVE_FOUR = 100_000    # 活四
SCORE_RUSH_FOUR = 10_000     # 冲四 (一端被堵)
SCORE_LIVE_THREE = 5_000     # 活三
SCORE_SLEEP_THREE = 500      # 眠三
SCORE_LIVE_TWO = 200         # 活二
SCORE_SLEEP_TWO = 20         # 眠二
SCORE_LIVE_ONE = 10          # 活一

# 威胁模式: (模式字符串, 分数)
# 'X' = 己方, 'O' = 对方, '_' = 空位
# 模式在四个方向 (横/竖/两对角线) 上匹配
THREAT_PATTERNS: list[tuple[str, int]] = [
    # 五连
    ("XXXXX", SCORE_FIVE),

    # 活四: _XXXX_
    ("_XXXX_", SCORE_LIVE_FOUR),

    # 冲四: 一端堵死或中间有空
    ("OXXXX_", SCORE_RUSH_FOUR),
    ("_XXXXO", SCORE_RUSH_FOUR),
    ("XXX_X",  SCORE_RUSH_FOUR),
    ("X_XXX",  SCORE_RUSH_FOUR),
    ("XX_XX",  SCORE_RUSH_FOUR),

    # 活三: _XXX_  中间无空，两端开
    ("__XXX_", SCORE_LIVE_THREE),
    ("_XXX__", SCORE_LIVE_THREE),
    ("_X_XX_", SCORE_LIVE_THREE),
    ("_XX_X_", SCORE_LIVE_THREE),

    # 眠三: 一端堵死
    ("OXXX__", SCORE_SLEEP_THREE),
    ("__XXXO", SCORE_SLEEP_THREE),
    ("OX_XX_", SCORE_SLEEP_THREE),
    ("_XX_XO", SCORE_SLEEP_THREE),
    ("OXX_X_", SCORE_SLEEP_THREE),
    ("_X_XXO", SCORE_SLEEP_THREE),

    # 活二
    ("__XX__", SCORE_LIVE_TWO),
    ("_X_X_",  SCORE_LIVE_TWO),
    ("_X__X_", SCORE_LIVE_TWO),

    # 眠二
    ("OXX___", SCORE_SLEEP_TWO),
    ("___XXO", SCORE_SLEEP_TWO),
    ("OX_X__", SCORE_SLEEP_TWO),
    ("__X_XO", SCORE_SLEEP_TWO),

    # 活一
    ("__X__", SCORE_LIVE_ONE),
]
