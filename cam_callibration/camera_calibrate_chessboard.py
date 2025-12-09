#!/usr/bin/env python3
"""
camera_calibrate_with_outlier_removal.py

Calibrate camera intrinsics from a folder of chessboard images, detect outliers
based on per-image reprojection error, move (or delete) outliers, and recalibrate
until stable.

Usage example:
  python camera_calibrate_with_outlier_removal.py --images-dir calibration/chessboard \
      --glob '*.jpg' --rows 6 --cols 9 --square-size 0.025 \
      --out calibration/camera_intrinsics_final.npz --bad-dir calibration/bad_images

Key options:
  --error-threshold : mean reprojection error (px) above which an image is considered an outlier (default 1.0)
  --max-iter        : maximum iterations of detect-move-recalibrate (default 3)
  --min-images      : minimum images required to perform calibration (default 10)
  --delete          : permanently delete outliers instead of moving them (default False)
  --fix-high-ks     : apply calibration flags to stabilize high-order K terms + zero tangential dist
  --debug-dir       : save debug images (corners drawn) into this dir
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import shutil
import os
import sys

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--images', nargs='+', help='List of chessboard images (space-separated)')
    group.add_argument('--images-dir', help='Directory containing chessboard images (use with --glob)')
    p.add_argument('--glob', default='*.jpg', help='Glob pattern for images in --images-dir (default: *.jpg)')
    p.add_argument('--rows', type=int, required=True, help='Number of inner corners per row (height)')
    p.add_argument('--cols', type=int, required=True, help='Number of inner corners per column (width)')
    p.add_argument('--square-size', type=float, required=True, help='Chessboard square size (meters or any unit)')
    p.add_argument('--out', default='calibration/camera_intrinsics_final.npz', help='Output .npz file')
    p.add_argument('--min-images', type=int, default=10, help='Minimum good images required (default 10)')
    p.add_argument('--error-threshold', type=float, default=1.0, help='Per-image mean reprojection error threshold (px) to classify outliers')
    p.add_argument('--max-iter', type=int, default=3, help='Max iterations to remove outliers and recalibrate')
    p.add_argument('--show', action='store_true', help='Show detected corners during processing')
    p.add_argument('--debug-dir', default=None, help='Save images with drawn corners to this directory (optional)')
    p.add_argument('--bad-dir', default='calibration/bad_images', help='Where to move bad calibration images')
    p.add_argument('--delete', action='store_true', help='Permanently delete bad images instead of moving (use with caution)')
    p.add_argument('--fix-high-ks', action='store_true', help='Fix high-order distortion coefficients (K3..K6) and tangential dist for stability')
    return p.parse_args()

def gather_images(args):
    if args.images:
        return [str(Path(p)) for p in args.images]
    else:
        pth = Path(args.images_dir)
        return sorted(str(p) for p in pth.glob(args.glob))

def detect_corners_on_images(image_paths, pattern_size, square_size, debug_dir=None, show=False):
    """
    Detect chessboard corners for each image path.
    Returns:
      - objpoints_list: list of object point arrays (one per good image)
      - imgpoints_list: list of image corners (one per good image)
      - good_image_paths: matching image paths that produced corners
      - img_shape: image size (w,h) or None if none read
    """
    rows = pattern_size[1]
    cols = pattern_size[0]
    objp_template = np.zeros((rows * cols, 3), np.float32)
    objp_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp_template *= square_size

    detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                       cv2.CALIB_CB_NORMALIZE_IMAGE |
                       cv2.CALIB_CB_FAST_CHECK)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []
    good_images = []
    img_shape = None

    for fname in image_paths:
        img = cv2.imread(str(fname))
        if img is None:
            print(f"[skip] cannot read: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (w,h)
        else:
            if gray.shape[::-1] != img_shape:
                print(f"[warn] skipping {fname}: resolution mismatch {gray.shape[::-1]} != {img_shape}")
                continue

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=detection_flags)
        if not ret:
            # fallback without FAST_CHECK
            ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            print(f"[warn] no corners: {fname}")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp_template.copy())
        imgpoints.append(corners_refined)
        good_images.append(fname)
        print(f"[ok] corners found: {fname}")

        if debug_dir:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, True)
            outp = Path(debug_dir) / Path(fname).name
            cv2.imwrite(str(outp), vis)

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, True)
            cv2.imshow('corners', vis)
            key = cv2.waitKey(200)
            if key == 27:
                print("[info] Aborted by user (ESC).")
                break

    if show:
        cv2.destroyAllWindows()

    return objpoints, imgpoints, good_images, img_shape

def calibrate_and_compute_errors(objpoints, imgpoints, image_paths, img_shape, calib_flags=0):
    """Run cv2.calibrateCamera and compute per-image mean reprojection errors.
       Returns ret, K, dist, rvecs, tvecs, per_image_errors (aligned with image_paths list)."""
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None, flags=calib_flags)
    per_image_errors = []
    total_err = 0.0
    total_points = 0
    for i, objp_i in enumerate(objpoints):
        proj, _ = cv2.projectPoints(objp_i, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1,2)
        obs = imgpoints[i].reshape(-1,2)
        err = np.linalg.norm(obs - proj, axis=1)
        mean_err = float(np.mean(err))
        per_image_errors.append(mean_err)
        total_err += np.sum(err)
        total_points += err.size
    overall_mean = total_err / total_points if total_points > 0 else float('nan')
    return ret, K, dist, rvecs, tvecs, per_image_errors, overall_mean

def move_or_delete_images(image_paths, indices, bad_dir, delete=False):
    """Move or delete images given by indices (list of indices into image_paths)."""
    moved = []
    bad_dir_p = Path(bad_dir)
    bad_dir_p.mkdir(parents=True, exist_ok=True)
    for i in sorted(indices, reverse=True):
        p = Path(image_paths[i])
        if not p.exists():
            continue
        if delete:
            p.unlink()
            print(f"[delete] {p}")
        else:
            dest = bad_dir_p / p.name
            # if already exists, append suffix
            if dest.exists():
                dest = bad_dir_p / f"{p.stem}_dup{p.suffix}"
            shutil.move(str(p), str(dest))
            print(f"[move] {p} -> {dest}")
            moved.append(str(dest))
    return moved

def main():
    args = parse_args()
    image_paths_all = gather_images(args)
    if not image_paths_all:
        print("[fail] No images found. Provide --images or --images-dir.")
        return

    pattern_size = (args.cols, args.rows)
    iter_count = 0
    current_image_paths = image_paths_all.copy()
    final_K = None
    final_dist = None
    final_reproj = None
    final_per_image = None

    calib_flags = 0
    if args.fix_high_ks:
        calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if hasattr(cv2, 'CALIB_FIX_K3'): calib_flags |= cv2.CALIB_FIX_K3
        if hasattr(cv2, 'CALIB_FIX_K4'): calib_flags |= cv2.CALIB_FIX_K4
        if hasattr(cv2, 'CALIB_FIX_K5'): calib_flags |= cv2.CALIB_FIX_K5
        if hasattr(cv2, 'CALIB_FIX_K6'): calib_flags |= cv2.CALIB_FIX_K6
        print("[info] Using calibration flags for stability:", calib_flags)

    while iter_count < args.max_iter:
        iter_count += 1
        print(f"\n--- Iteration {iter_count} (images: {len(current_image_paths)}) ---")
        objpoints, imgpoints, good_images, img_shape = detect_corners_on_images(
            current_image_paths, pattern_size, args.square_size,
            debug_dir=args.debug_dir, show=args.show
        )

        if len(objpoints) < args.min_images:
            print(f"[fail] Only {len(objpoints)} usable images with corners detected. Need >= {args.min_images}. Aborting.")
            return

        ret, K, dist, rvecs, tvecs, per_image_errors, overall_mean = calibrate_and_compute_errors(
            objpoints, imgpoints, good_images, img_shape, calib_flags
        )

        print(f"[info] Iter {iter_count} - RMS reported by cv2.calibrateCamera: {ret:.6f}")
        print(f"[info] Iter {iter_count} - Overall mean reprojection error (per corner): {overall_mean:.6f}")
        # print first 10 per-image errors
        print(f"[info] Iter {iter_count} - Per-image mean errors (first 10): {per_image_errors[:10]}")

        # find outlier indices (relative to good_images list)
        outlier_indices = [i for i, e in enumerate(per_image_errors) if e > args.error_threshold]
        if not outlier_indices:
            print(f"[info] No outliers detected (threshold {args.error_threshold}px). Calibration stable.")
            final_K = K; final_dist = dist; final_reproj = ret; final_per_image = per_image_errors
            break

        # Map these indices to their paths in current_image_paths.
        # Note: good_images is a subset of current_image_paths (those with detected corners),
        # so we need indices of those filenames in current_image_paths.
        outlier_paths = [good_images[i] for i in outlier_indices]
        print(f"[warn] Detected {len(outlier_indices)} outlier images (mean error > {args.error_threshold}px):")
        for idx in outlier_indices:
            print(f"  [bad] {good_images[idx]}  (error = {per_image_errors[idx]:.3f})")

        # Find indices in current_image_paths to move/delete
        current_path_index_map = {p: i for i, p in enumerate(current_image_paths)}
        indices_in_current = [current_path_index_map[p] for p in outlier_paths if p in current_path_index_map]

        if not indices_in_current:
            print("[error] Could not map outlier filenames back to current image list. Aborting.")
            return

        # Move/Delete outliers
        moved = move_or_delete_images(current_image_paths, indices_in_current, args.bad_dir, delete=args.delete)

        # Rebuild current_image_paths excluding moved/deleted items
        current_image_paths = [p for p in current_image_paths if p not in outlier_paths]

        # Continue next iteration (recalibrate on remaining images)
        print(f"[info] Iter {iter_count} - Removed {len(indices_in_current)} outliers. Remaining images: {len(current_image_paths)}")

        # If too few images remain, abort
        if len(current_image_paths) < args.min_images:
            print(f"[fail] After removing outliers only {len(current_image_paths)} images remain (need >= {args.min_images}). Aborting.")
            return

        # If last iteration reached, accept current result (after removal) and stop
        if iter_count >= args.max_iter:
            print("[info] Reached max iterations; accepting current calibration.")
            # run final calibration on the remaining images to produce final_K
            objpoints, imgpoints, good_images, img_shape = detect_corners_on_images(
                current_image_paths, pattern_size, args.square_size,
                debug_dir=args.debug_dir, show=False
            )
            ret, K, dist, rvecs, tvecs, per_image_errors, overall_mean = calibrate_and_compute_errors(
                objpoints, imgpoints, good_images, img_shape, calib_flags
            )
            final_K = K; final_dist = dist; final_reproj = ret; final_per_image = per_image_errors
            break

    # If loop ended without setting final_K (unlikely), set from last run
    if final_K is None:
        final_K = K; final_dist = dist; final_reproj = ret; final_per_image = per_image_errors

    # Save final intrinsics + diagnostics
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(outp), K=final_K, dist=final_dist, reproj_rms=float(final_reproj), per_image_errors=np.array(final_per_image))
    print(f"\n[done] Saved final intrinsics to: {outp}")
    print("Final camera matrix K:\n", final_K)
    print("Final dist coeffs:\n", final_dist.ravel())
    print(f"Final reported RMS: {final_reproj:.6f}")
    print(f"Final per-image mean reproj errors (first 10): {final_per_image[:10]}")

if __name__ == "__main__":
    main()
