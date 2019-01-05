import numpy as np
norm = np.linalg.norm
tmp = np.arange(51)
pred = np.zeros((64, 51))
gt = np.zeros((64, 51))
for i in range(64):
    pred[i, :] = tmp
    gt[i, :] = tmp
threshold = np.linspace(0, 1, 11)
threshold = [0.5]

assert pred.shape == gt.shape
pred = np.reshape(pred, (64, 3, 17))
gt = np.reshape(gt, (64, 3, 17))
pred = np.random.rand(pred.shape[0],pred.shape[1],pred.shape[2])
gt = np.random.rand(pred.shape[0],pred.shape[1],pred.shape[2])


def get_dist_pck(pred, gt):
    dist_ratio = np.zeros((1, pred.shape[0], pred.shape[2]))
    for imgidx in range(64):
        refDist = norm(gt[imgidx, :, 13]-gt[imgidx, :, 11])

        dist_ratio[0, imgidx, :] = np.sqrt(
            sum(
                np.square(
                    pred[imgidx, :, :]-gt[imgidx, :, :]
                ), 0
            )
        )/refDist

    return dist_ratio


def compute_pck(dist_ratio, pck_thresh):
    pck = np.zeros((len(threshold), dist_ratio.shape[2]+1))
    for jidx in range(dist_ratio.shape[2]):
        for i, t in enumerate(threshold):
            pck[i, jidx] = 100*np.mean(np.squeeze(dist_ratio[0, :, jidx]) <= t)
    for i, t in enumerate(threshold):
            pck[i, -1] = 100*np.mean(np.reshape(np.squeeze(dist_ratio[0, :, :]),
                                                (dist_ratio.shape[1]*dist_ratio.shape[2], 1)) <= t)
    return pck