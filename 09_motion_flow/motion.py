from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import skimage
import numpy as np
from tqdm.auto import tqdm


def create_video(save_dir, size, img_format='jpg', vido_format='avi'):
    out_name = Path(save_dir).parts[-1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(Path(save_dir) / Path(f'{out_name}.{vido_format}')),
                          fourcc, 20, tuple(size.astype(int)))

    for fname in tqdm(sorted(map(str, Path(save_dir).glob(f'*.{img_format}')))):
        imag = skimage.io.imread(fname)
        out.write(imag)
    out.release()


def get_video_details(cap):
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(cnt, w, h, fps)


def sparse_motion():
    color = (0, 255, 0)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture("shibuya.mp4")
    get_video_details(cap)
    cv2.startWindowThread()
    cv2.namedWindow('imageWindow')
    is_ok, first_frame = cap.read()

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    mask = np.zeros_like(first_frame)
    idx = 0
    while (cap.isOpened()):
        is_ok, frame = cap.read()
        if not is_ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)

        prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

        good_old = prev[status == 1].astype(int)
        good_new = next[status == 1].astype(int)

        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            frame = cv2.circle(frame, (a, b), 3, color, -1)

        output = cv2.add(frame, mask)
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)

        # Opens a new window and displays the output frame
        cv2.imshow("sparse optical flow", output)
        cv2.imwrite(f'tmp/frame_{str(idx).zfill(5)}.jpg', output)
        idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()


def dense_motion():
    # cap = cv2.VideoCapture("shibuya.mp4")
    cap = cv2.VideoCapture(0)
    is_ok, first_frame = cap.read()

    farn_params = dict(pyr_scale=0.5, levels=3, winsize=35,
                       iterations=3, poly_n=5, poly_sigma=1.2,
                       flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    while (cap.isOpened()):
        is_ok, frame = cap.read()
        if not is_ok:
            break
        cv2.imshow("input", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **farn_params)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        cv2.imshow("dense optical flow", rgb)

        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # sparse_motion()
    dense_motion()

    # create_video('tmp', size=np.array((1280, 720)), vido_format='mp4')
