import numpy as np
import matplotlib.pyplot as plt


def two_moons(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + .5, -np.sin(x)], axis=1) * 10.
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - .5, np.sin(x)], axis=1) * 10.
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    p_num = np.random.randint(0, n//2, 5)
    n_num = np.random.randint(n//2+1, n, 5)
    y[0:20] = 1
    y[-20:] = -1
    gt = np.zeros(n)
    gt[0:n//2] = 1
    gt[n//2:] = -1
    return x, y, gt


def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


def visualize(theta, x, y, gt, grid_size=100, x_min=-20, x_max=20, is_supervised=False):
    grid = np.linspace(x_min, x_max, grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)

    x_tmp, y_tmp = x, y
    f = 0
    if is_supervised:
        x_tmp, y_tmp = x[y != 0], y[y != 0]
        f = build_design_mat(x, x_tmp, bandwidth=1.) @ theta
    else:
        f = build_design_mat(x, x, bandwidth=1.) @ theta
    scores = np.sign(f)
    acc = np.sum(scores[y == 0] == gt[y == 0]) / np.size(gt[y == 0])
    print(acc)
    design_mat = build_design_mat(x_tmp, mesh_grid, bandwidth=1.)
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.contourf(X, Y, np.reshape(np.sign(design_mat.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)

    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$O$', c='blue')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='+', c='red')
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='x', c='gray')
    plt.text(0, 18, str(f'Accuracy: {acc:.2f}'))
    plt.figure()
    plt.show()


def cal_error(theta, x, y, h):
    K = build_design_mat(x, x, h)
    scores = K @ theta
    print(np.where(scores.flatten() > 0, 1, -1))


if __name__ == '__main__':
    x, y, gt = two_moons()
    x_labeled, y_labeled = x[y != 0], y[y != 0]
    x_unlabeled, y_unlabeled = x[y == 0], y[y == 0]
    # parameters
    h = 1
    w = 1
    lamda = 1
    nu = 1

    # supervised learning
    K = build_design_mat(x_labeled, x_labeled, h)
    A = K.T @ K + lamda * np.identity(K.shape[0])
    b = K.T @ y_labeled
    theta_supervised = np.linalg.solve(A, b)
    visualize(theta_supervised, x, y, gt, is_supervised=True)

    # semi-supervised learning
    W = build_design_mat(x, x, h)
    D = np.diag([np.sum(W[i]) for i in range(W.shape[0])])
    L = D - W
    Phi = build_design_mat(x, x, w)
    Phi_tilde = build_design_mat(x_labeled, x, h)
    I = np.identity(Phi_tilde.shape[1])
    A = Phi_tilde.T @ Phi_tilde + lamda * I + 2 * nu * Phi.T @ L @ Phi
    b = Phi_tilde.T @ y_labeled
    theta = np.linalg.solve(A, b)
    visualize(theta, x, y, gt)
