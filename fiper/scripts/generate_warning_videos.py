"""Generate rollout videos with uncertainty scores and failure detection overlay.

Usage:
    python gv.py --method rnd_oe
    python gv.py --method entropy --threshold_style tvt_quantile --quantile 0.95 --window 30
    python gv.py --method rnd_oe --task mytask --fps 15
    python gv.py --list_methods
"""

import os, sys, pickle, argparse
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Generate failure-detection videos from FIPER eval results")
    parser.add_argument("--method", type=str, default="rnd_oe",
                        help="Method name (rnd_oe, rnd_a, entropy, logpzo, similarity, tc)")
    parser.add_argument("--task", type=str, default="mytask", help="Task name")
    parser.add_argument("--threshold_style", type=str, default="tvt_cp_band",
                        help="Threshold style (tvt_cp_band, tvt_quantile, ct_quantile)")
    parser.add_argument("--quantile", type=float, default=0.9, help="Quantile value")
    parser.add_argument("--window", type=int, default=45, help="Window size for uncertainty aggregation")
    parser.add_argument("--base_data", type=str, default="/home/zhiyuanjia/fiper/data", help="Base data directory")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    parser.add_argument("--frames_only", action="store_true", help="Only generate frames, skip ffmpeg")
    parser.add_argument("--list_methods", action="store_true", help="List available methods and exit")
    args = parser.parse_args()

    base = args.base_data

    if args.list_methods:
        results_dir = os.path.join(base, args.task, "results")
        if os.path.isdir(results_dir):
            methods = sorted(os.listdir(results_dir))
            print("Available methods: %s" % ", ".join(methods))
        else:
            print('No results found for task "%s"' % args.task)
        return

    results_file = os.path.join(base, args.task, "results", args.method, "eval_results.pkl")
    rollout_dir  = os.path.join(base, args.task, "rollouts", "test")
    frames_dir   = os.path.join(base, "results", "videos_with_warnings", args.task, "rollouts", "test_frames")
    videos_dir   = os.path.join(base, "results", "videos_with_warnings", args.task, "rollouts")

    if not os.path.exists(results_file):
        print("ERROR: results file not found: %s" % results_file)
        sys.exit(1)
    if not os.path.isdir(rollout_dir):
        print("ERROR: rollout directory not found: %s" % rollout_dir)
        sys.exit(1)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    with open(results_file, "rb") as f:
        res = pickle.load(f)

    tsbt = res.get("test_scores_by_threshold", {})
    if args.threshold_style not in tsbt:
        print('ERROR: threshold_style "%s" not found. Available: %s' %
              (args.threshold_style, list(tsbt.keys())))
        sys.exit(1)
    if args.quantile not in tsbt[args.threshold_style]:
        avail_q = sorted(tsbt[args.threshold_style].keys())
        print("ERROR: quantile %.2f not found. Available: %s" % (args.quantile, avail_q))
        sys.exit(1)
    if args.window not in tsbt[args.threshold_style][args.quantile]:
        avail_w = sorted(tsbt[args.threshold_style][args.quantile].keys())
        print("ERROR: window %d not found. Available: %s" % (args.window, avail_w))
        sys.exit(1)

    norm_scores = tsbt[args.threshold_style][args.quantile][args.window]

    raw_scores_data = res["test_uncertainty_scores"].get(1)
    if raw_scores_data is None:
        raw_scores_data = res["test_uncertainty_scores"][list(res["test_uncertainty_scores"].keys())[0]]

    successful = res["successful_test_rollouts"]
    ood = res.get("ood_test_rollouts", [False] * len(successful))
    method_name = res.get("method", args.method)

    pkl_files = sorted([f for f in os.listdir(rollout_dir) if f.endswith(".pkl")])

    print("=" * 60)
    print("Method:          %s" % method_name)
    print("Task:            %s" % args.task)
    print("Threshold style: %s" % args.threshold_style)
    print("Quantile:        %.2f" % args.quantile)
    print("Window:          %d" % args.window)
    print("Episodes:        %d" % len(pkl_files))
    print("FPS:             %d" % args.fps)
    print("=" * 60)
    print()

    for k, pkl_file in enumerate(pkl_files):
        ep_name = "episode_%02d" % k
        print("  %s..." % ep_name, end=" ", flush=True)

        with open(os.path.join(rollout_dir, pkl_file), "rb") as f:
            rollout_data = pickle.load(f)

        frames = rollout_data["rollout"]
        num_frames = len(frames)

        norm_arr = np.array(norm_scores[k])
        if isinstance(raw_scores_data[k], dict):
            raw_arr = raw_scores_data[k]["uncertainty_scores"]
        else:
            raw_arr = raw_scores_data[k]

        above = np.where(norm_arr > 1)[0]
        first_warn = above[0] if len(above) > 0 else -1

        is_ood = bool(ood[k])
        is_success = bool(successful[k])

        ep_dir = os.path.join(frames_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        for i in range(num_frames):
            rgb = frames[i]["rgb"]
            img = Image.fromarray(rgb).convert("RGBA")

            raw_score = float(raw_arr[i]) if i < len(raw_arr) else 0.0
            norm_score = float(norm_arr[i]) if i < len(norm_arr) else 0.0

            ep_failed = first_warn >= 0 and i >= first_warn

            panel_h = 55
            overlay = Image.new("RGBA", (360, panel_h), (0, 0, 0, 180))
            img.paste(overlay, (0, 360 - panel_h), overlay)

            draw = ImageDraw.Draw(img)

            status = "OOD" if is_ood else "ID"
            sf = "Success" if is_success else "Fail"
            line1 = "Frame %04d/%d | %s | %s" % (i, num_frames - 1, status, sf)
            draw.text((8, 360 - panel_h + 2), line1, fill=(200, 200, 200))

            line2 = "%s raw: %.6f" % (method_name.upper(), raw_score)
            draw.text((8, 360 - panel_h + 17), line2, fill=(255, 255, 200))

            nc = (255, 100, 100) if norm_score > 1.0 else (100, 255, 100)
            line3 = "%s norm: %.4f" % (method_name.upper(), norm_score)
            draw.text((8, 360 - panel_h + 32), line3, fill=nc)

            if ep_failed:
                bw = 5
                for x in range(360):
                    for dy in range(bw):
                        img.putpixel((x, dy), (255, 0, 0, 255))
                        img.putpixel((x, 359 - dy), (255, 0, 0, 255))
                for y in range(360):
                    for dx in range(bw):
                        img.putpixel((dx, y), (255, 0, 0, 255))
                        img.putpixel((359 - dx, y), (255, 0, 0, 255))

                banner = Image.new("RGBA", (360, 30), (200, 0, 0, 200))
                img.paste(banner, (0, 10), banner)
                draw = ImageDraw.Draw(img)
                draw.text((360 // 2 - 65, 13), "FAILURE DETECTED", fill=(255, 255, 255))

            fp = os.path.join(ep_dir, "frame_%04d.jpg" % i)
            img.convert("RGB").save(fp, quality=90)

        if not args.frames_only:
            vp = os.path.join(videos_dir, "%s.mp4" % ep_name)
            ffmpeg_cmd = "ffmpeg -y -framerate %d -i %s/frame_%%04d.jpg -c:v libx264 -pix_fmt yuv420p %s 2>/dev/null" % (
                args.fps, ep_dir, vp)
            ret = os.system(ffmpeg_cmd)
            print("OK" if ret == 0 else "FFMPEG ERROR")
        else:
            print("FRAMES OK")

    print()
    print("Done! Output: %s" % videos_dir)


if __name__ == "__main__":
    main()
