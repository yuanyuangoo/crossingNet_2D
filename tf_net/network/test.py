import numpy as np


def get_dist_pck(pred, gt):
    dist_ratio = np.zeros((pred.shape[0], pred.shape[2]))
    for imgidx in range(64):
        refDist = np.linalg.norm(gt[imgidx, :, 12]-gt[imgidx, :, 5])

        dist_ratio[imgidx, :] = np.sqrt(
            sum(
                np.square(
                    pred[imgidx, :, :]-gt[imgidx, :, :]
                ), 0)
        )/refDist

    return dist_ratio


def compute_pck(dist_ratio, label, threshold):
    pck = np.zeros((len(threshold),label.shape[1]+1, dist_ratio.shape[1]+1))
    for jidx in range(dist_ratio.shape[1]):
        for i_label in range(label.shape[1]):
            idx_of_label = np.where(label[:, i_label] == 1)[0]
            if len(idx_of_label) == 0:
                continue
            for i, t in enumerate(threshold):
                pck[i, i_label, jidx] = 100 * \
                    np.mean(dist_ratio[idx_of_label[0], jidx] <= t)
    for i_label in range(label.shape[1]):
        idx_of_label = np.where(label[:, i_label] == 1)[0]
        if len(idx_of_label) == 0:
                continue
        for i, t in enumerate(threshold):
                pck[i, i_label, -1] = 100 * \
                    np.mean(dist_ratio[idx_of_label[0], :] <= t)

    for jidx in range(dist_ratio.shape[1]):
        for i, t in enumerate(threshold):
            pck[i, -1, jidx] = 100 * \
                np.mean(dist_ratio[:, jidx] <= t)

    for i, t in enumerate(threshold):
        pck[i, -1, -1] = 100 * \
            np.mean(dist_ratio[:, :] <= t)
    return pck


def eval_pck(pred, gt, label, symmetry_joint_id, joint_name, name):
    pred = np.reshape(pred, (pred.shape[0], 3, 17))
    gt = np.reshape(gt, (gt.shape[0], 3, 17))

    dist = get_dist_pck(pred, gt)
    pck_all = compute_pck(dist, label, [0.5, 0.2])
    return pck_all


result = np.load('result_1.out.npy')
label = np.zeros((result.shape[0], 15))
label[1:2, 0] = 1
label[2:4, 1] = 1
label[4:6, 2] = 1
label[6:17, 3] = 1
label[17:26, 4] = 1
label[26:32, 5] = 1
label[32:48, 6] = 1
label[48:, 7] = 1

gt = np.random.randn(result.shape[0], result.shape[1])*40
pck = eval_pck(result, gt, label, 1, 1, 1)
print(pck.shape)
