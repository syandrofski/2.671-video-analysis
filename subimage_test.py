import numpy as np
def subimage(frame, centers):
    corners = []
    subframes = []
    h = len(frame[0])  # 1920, columns
    w = len(frame)  # 1080, rows
    buf = 50
    for center in centers:

        try:
            ll_x = center[0]-buf
        except:
            ll_x = 0

        try:
            ll_y = center[1]-buf
        except:
            ll_y = 0

        try:
            tr_x = center[0]+buf
        except:
            tr_x = h-1

        try:
            tr_y = center[1]+buf
        except:
            tr_y = w-1

        corners.append([(ll_x, ll_y), (tr_x, tr_y)])

    for pair in corners:
        subframes.append(frame[pair[0][0]:pair[1][0], pair[0][1]:pair[1][1]])
    return corners, subframes


frame = np.ones((1920, 1080))
frame[450:550, 450:550] = 2
centers = [(500, 500), (750, 750)]

corns, subs = subimage(frame, centers)
print(corns)
print(subs)