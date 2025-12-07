import mygauss
import numpy as np


def gauss_joradn_inverse():
    n = 100
    A_np = np.random.rand(n, n) + np.eye(n) * n

    try:
        result_list = mygauss.inverse(A_np)
    except Exception as e:
        print(f"Error: {e}")

    inv_cpp = np.array(result_list)

    inv_np = np.linalg.inv(A_np)

    diff = np.linalg.norm(inv_cpp - inv_np)
    identity_approx = A_np @ inv_cpp
    identity_diff = np.linalg.norm(identity_approx - np.eye(n))

    if diff < 1e-8 and identity_diff < 1e-8:
        print("[PASS] Results pretty match.")
    else:
        print("[FAIL] Results diverge.")


if __name__ == "__main__":
    gauss_joradn_inverse()
