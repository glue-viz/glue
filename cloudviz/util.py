def relim(lo, hi, log=False):
    x, y = lo, hi
    if log:
        if lo < 0:
            x = 1e-5
        if hi < 0:
            y = 1e5
    return (x,y)
