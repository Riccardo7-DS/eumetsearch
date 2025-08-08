def bbox_mtg():
    return [-18.105469,-37.857507,60.820313,71.413177]

def hoa_bbox(invert:bool = False):
    minx = -5.48369565
    miny = 32.01630435
    maxx = 15.48369565
    maxy = 51.48369565

    if invert is False:
        return [miny, minx, maxy, maxx]
    else:
        return [minx, miny, maxx, maxy]