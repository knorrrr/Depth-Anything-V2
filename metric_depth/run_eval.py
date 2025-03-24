import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

def calculate_absolute_error(pred_depth, true_depth):
    """
    Calculate the accuracy of the predicted depth compared to the true depth.
    Args:
        pred_depth (np.array): Predicted depth map.
        true_depth (np.array): Ground truth depth map.
        threshold (float): Threshold for accuracy calculation.
    Returns:
        float: Accuracy of the predicted depth.
    """
    valid_mask = (true_depth > 0)&(true_depth <= args.max_depth)
    pred_depth = pred_depth[valid_mask]
    true_depth = true_depth[valid_mask]
    
    absolute_error = np.abs(pred_depth - true_depth)
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error 

def get_sorted_png_files(path):
    return sorted([f for f in glob.glob(os.path.join(path, '**/*.png'), recursive=True)])

def process_image(depth_anything, true_depth_file, img_filename, batch ,args, cmap):
    """
    Process a single image: infer depth, calculate error, and save results.
    Args:
        depth_anything (DepthAnythingV2): Depth estimation model.
        raw_image (np.array): Input image.
        true_depth_file (str): Path to the ground truth depth file.
        img_filename (str): Filename of the input image.
        args (argparse.Namespace): Command line arguments.
        cmap (matplotlib.colors.Colormap): Colormap for visualization.
    """
    raw_image = cv2.imread(img_filename)
    raw_image = np.array(list(raw_image))
    depth = depth_anything.infer_image(raw_image, args.input_size)
    
    if args.save_numpy:
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(img_filename))[0] + '_raw_depth_meter.npy')
        np.save(output_path, depth)

    # Load the ground truth depth data
    true_depth = cv2.imread(true_depth_file, cv2.IMREAD_COLOR)
    true_depth = true_depth[:, :, 0]  # Use only one channel since all channels are the same
    # Round the predicted depth to the nearest integer
    rounded_depth = np.round(depth)
    absolute_error = calculate_absolute_error(rounded_depth, true_depth)
    # print(f'Absolute error for IMG {img_filename}: {absolute_error:.4f}')

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    depth_rgb = np.stack((depth, depth, depth), axis=-1)
    # Save the RGB depth image
    output_path_rgb = os.path.join(args.outdir, os.path.splitext(os.path.basename(img_filename))[0] + '_depth_rgb.png')
    cv2.imwrite(output_path_rgb, depth_rgb)       
    # print(f'Saved RGB depth image to {output_path_rgb}')

    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(img_filename))[0] + '.png')
    if args.pred_only:
        cv2.imwrite(output_path, depth)
    else:
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth])
        
        cv2.imwrite(output_path, combined_result)

    return absolute_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--event-img-path', type=str, help='Path to the event images')

    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_vits_ev.pth')
    parser.add_argument('--max-depth', type=float, default=80)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--true-depth-path', type=str, help='Path to the ground truth depth data')

    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                img_filenames = f.read().splitlines()
        else:
            img_filenames = [args.img_path]
    else:
        img_filenames = get_sorted_png_files(args.img_path)

    if os.path.isfile(args.event_img_path):
        if args.event_img_path.endswith('txt'):
            with open(args.event_img_path, 'r') as f:
                event_img_filenames = f.read().splitlines()
        else:
            event_img_filenames = [args.event_img_path]
    else:
        event_img_filenames = get_sorted_png_files(args.event_img_path)

    if os.path.isfile(args.true_depth_path):
        if args.true_depth_path.endswith('txt'):
            with open(args.true_depth_path, 'r') as f:
                true_depth_files = f.read().splitlines()
        else:
            true_depth_files = [args.true_depth_path]
    else:
        true_depth_files = get_sorted_png_files(args.true_depth_path)

    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')

    rgb_errors = []
    rgb_event_errors = []
    filenames = []

    for k, (img_filename, event_img_filename, true_depth_file) in enumerate(zip(img_filenames, event_img_filenames ,true_depth_files)):
        print(f'Progress {k+1}/{len(img_filenames)}: {img_filename} and {event_img_filename}, {true_depth_file}')
        process_image(depth_anything, true_depth_file, img_filename, args, cmap)
        raw_image = cv2.imread(img_filename)
        event_image = cv2.imread(event_img_filename)
        # Process RGB image
        rgb_error = process_image(depth_anything, true_depth_file, img_filename, args, cmap)
        print(f'Absolute error for IMG {img_filename}: {rgb_error:.4f}')    
        rgb_errors.append(rgb_error)
        # Process RGB+Event image
        rgb_event_error =process_image(depth_anything, true_depth_file, event_img_filename, args, cmap)
        print(f'Absolute error for IMG {event_img_filename}: {rgb_event_error:.4f}')
        rgb_event_errors.append(rgb_event_error)
        
        filenames.append(os.path.basename(img_filename))
        # if (k ==100):
        #     break
    
    average_rgb_error = np.mean(rgb_errors)
    average_rgb_event_error = np.mean(rgb_event_errors)
    print(f'Average RGB Error: {average_rgb_error:.4f}')
    print(f'Average RGB+Event Error: {average_rgb_event_error:.4f}')
    # Plot the errors
    plt.figure(figsize=(10, 5))
    plt.plot(filenames, rgb_errors, label='RGB Error', marker='o')
    plt.plot(filenames, rgb_event_errors, label='RGB+Event Error', marker='x')
    plt.xlabel('Image Filename')
    plt.ylabel('Mean Absolute Error')
    plt.title('Comparison of Depth Estimation Errors')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'depth_estimation_errors.png'))
    plt.show()