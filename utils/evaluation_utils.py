import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset import get_imagenet_dataloader


def _process_single_image(image_path):
    """Process a single image into a tensor."""
    # Read the RGB image
    img = Image.open(image_path).convert("RGB")
    # Convert to numpy array (already in uint8)
    img = np.array(img)
    # Transpose from (H, W, C) to (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    # Convert to torch tensor while preserving uint8
    return torch.from_numpy(img).to(dtype=torch.uint8)


def _process_image_batch(image_paths, num_workers) -> torch.Tensor:
    """Process a batch of images in parallel."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_tensors = list(executor.map(_process_single_image, image_paths))
    return torch.stack(batch_tensors, dim=0)


def _batch_generator(paths, batch_size):
    """Generate batches of paths."""
    for i in range(0, len(paths), batch_size):
        yield paths[i : i + batch_size]


def read_samples(path, batch_size, num_workers: int = None):
    """
    Read images in parallel batches from the given path.

    Args:
        path: Directory path containing images
        batch_size: Number of images to process in each batch
        num_workers: Number of parallel workers (defaults to CPU count - 1)

    Returns:
        torch.Tensor: Stacked tensor of all images
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Get all image paths first
    image_paths = list(Path(path).rglob("*.png"))
    all_tensors = []

    # Process images in parallel batches
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for batch_paths in tqdm(
        _batch_generator(image_paths, batch_size),
        total=total_batches,
        desc=f"Processing images with {num_workers} workers",
    ):
        batch_tensor = _process_image_batch(batch_paths, num_workers)
        all_tensors.append(batch_tensor)

    # Stack all batches together
    stacked_tensor = torch.cat(all_tensors, dim=0)
    print(f"Read {len(stacked_tensor)} images")
    return stacked_tensor


def get_dataset_samples(n_samples, dataset_name, batch_size, seed, data_path, test_size=0.01):
    if dataset_name != "imagenet":
        raise ValueError(f"Dataset {dataset_name} not implemented.")
    dataloader, _ = get_imagenet_dataloader(
        batch_size=batch_size,
        seed=seed,
        data_dir=data_path,
        test_size=test_size,
        normalize=False,
    )

    curr_n_samples = 0
    dataloader_iter = iter(dataloader)

    samples = []

    while curr_n_samples < n_samples:
        _samples = next(dataloader_iter)
        if isinstance(_samples, tuple) or isinstance(_samples, list):
            _samples, _ = _samples

        _samples = (_samples * 255.0).to(dtype=torch.uint8)
        samples.append(_samples)

        curr_n_samples += batch_size
        print(f"Read {curr_n_samples}/{n_samples} samples.")

    samples = torch.cat(samples, dim=0)[:n_samples]
    return samples


def save_images(images, labels, path, idx_start, idx_end):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if labels is None:
        for idx in tqdm(range(idx_start, idx_end)):
            additional_name = f"{idx}.png"
            filename = Path(path) / additional_name

            img = images[idx - idx_start]
            img = img.cpu().numpy().transpose(1, 2, 0)
            pil_img = Image.fromarray(img)

            pil_img.save(filename)

        print(f"Images from {idx_start} to {idx_end - 1} are saved in {path}")

    else:
        for idx_label, label in tqdm(enumerate(labels)):
            for idx in tqdm(range(idx_start, idx_end)):
                additional_name = f"{idx}_{label}.png"

                filename = Path(path) / additional_name

                idx_img = (idx_end - idx_start) * idx_label + idx
                img = images[idx_img]
                img = img.cpu().numpy().transpose(1, 2, 0)
                pil_img = Image.fromarray(img)

                pil_img.save(filename)

            print(f"{idx_end - idx_start} images for class {label} are saved in path {path}")


def process_and_save_image(image_tensor_or_array, filename_path):
    """
    Processes a single image (tensor or array) and saves it to disk.
    """
    try:
        # If it's a PyTorch tensor, move to CPU, convert to NumPy, and transpose
        if hasattr(image_tensor_or_array, "cpu") and hasattr(
            image_tensor_or_array, "numpy"
        ):  # Heuristic for PyTorch tensor
            img_np = image_tensor_or_array.cpu().numpy().transpose(1, 2, 0)
        elif isinstance(image_tensor_or_array, np.ndarray):
            # Assuming if it's already numpy, it might be in CHW, try to transpose
            # This part might need adjustment based on your exact numpy array structure
            if image_tensor_or_array.ndim == 3 and image_tensor_or_array.shape[0] in [1, 3, 4]:  # CHW heuristic
                img_np = image_tensor_or_array.transpose(1, 2, 0)
            else:  # Assume HWC or grayscale
                img_np = image_tensor_or_array
        else:
            raise TypeError("Unsupported image data type. Expected PyTorch tensor or NumPy array.")

        # Ensure data type is appropriate for PIL (e.g., uint8 for typical images)
        # If img_np is float (typically 0-1 range), scale to 0-255 and convert to uint8
        if img_np.dtype in [np.float32, np.float64, np.float16]:
            if img_np.min() >= 0 and img_np.max() <= 1:  # Assuming 0-1 range
                img_np = (img_np * 255).astype(np.uint8)
            # Add other normalizations if necessary, e.g. for -1 to 1 range
            # elif img_np.min() >= -1 and img_np.max() <= 1: # Assuming -1 to 1 range
            #    img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
            else:
                # If float but not in a known range, try to convert directly.
                # PIL might handle some float types, or it might require explicit conversion.
                # For safety, converting to uint8 if it looks like scaled float is good.
                # If values are already e.g. 0-255 float, then just astype(np.uint8)
                if img_np.max() > 1:  # Potentially already scaled to 0-255 but float
                    img_np = img_np.astype(np.uint8)

        # Handle single-channel images correctly for PIL (remove channel dim if HxWx1)
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)

        pil_img = Image.fromarray(img_np)
        pil_img.save(filename_path)
        return None  # Success
    except Exception as e:
        return f"Error saving {filename_path}: {e}"


def save_images_parallel(images, labels, path, idx_start, idx_end, max_workers=None):
    """
    Saves images to disk in parallel.

    Args:
        images (list or Tensor): A list of image tensors (e.g., PyTorch tensors) or NumPy arrays.
                                 If 'labels' is None, len(images) should be idx_end - idx_start.
                                 If 'labels' is not None, 'images' should be a flat list where
                                 images for label0 come first, then label1, etc.
                                 Total images = len(labels) * (idx_end - idx_start).
        labels (list, optional): A list of labels. If None, images are saved with index names.
        path (str or Path): The directory to save images.
        idx_start (int): The starting index for naming images.
        idx_end (int): The ending index (exclusive) for naming images.
        max_workers (int, optional): Maximum number of threads to use.
                                     Defaults to a sensible value based on CPU cores.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        # A common default for I/O-bound tasks
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    # Ensure max_workers is at least 1 for the logic below
    if max_workers <= 0:
        max_workers = 1

    if labels is None:
        tasks = []
        # Assuming 'images' contains (idx_end - idx_start) elements for this call
        num_images_to_save = idx_end - idx_start
        if len(images) != num_images_to_save:
            print(
                f"Warning: Number of images provided ({len(images)}) does not match "
                f"expected count ({num_images_to_save}) based on idx_start/idx_end."
            )
            # Adjust num_images_to_save to actual images provided to avoid errors,
            # but this indicates a potential mismatch in arguments.
            num_images_to_save = min(len(images), num_images_to_save)

        for i in range(num_images_to_save):
            current_idx_name = idx_start + i  # The name index
            additional_name = f"{current_idx_name}.png"
            filename = path_obj / additional_name
            image_data = images[i]  # Direct index into the provided 'images' slice
            tasks.append((image_data, filename))

        if not tasks:
            print(f"No images to save from {idx_start} to {idx_end - 1} in {path_obj}.")
            return

        print(f"Preparing to save {len(tasks)} images in {path_obj} using up to {max_workers} threads...")

        results = []
        # Determine workers for this specific batch
        current_batch_workers = min(max_workers, len(tasks))

        with ThreadPoolExecutor(max_workers=current_batch_workers) as executor:
            futures = [executor.submit(process_and_save_image, img_data, fname) for img_data, fname in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Saving images"):
                results.append(future.result())

        successful_saves = sum(1 for r in results if r is None)
        failed_saves = [r for r in results if r is not None]

        print(
            f"{successful_saves} images (indices {idx_start} to {idx_start + num_images_to_save -1}) saved {path_obj}"
        )
        if failed_saves:
            print(f"Failed to save {len(failed_saves)} images:")
            for error_msg in failed_saves:
                print(f" - {error_msg}")

    else:  # labels is not None
        total_images_processed_successfully = 0
        total_images_failed = 0
        all_failed_messages = []

        images_per_segment = idx_end - idx_start  # Number of images per label for this call's segment

        expected_total_images = len(labels) * images_per_segment
        if len(images) != expected_total_images:
            print(
                f"Warning: Total number of images provided ({len(images)}) does not match "
                f"expected count ({expected_total_images}) based on labels and idx_start/idx_end."
            )
            # This might lead to IndexError if not handled or if assumptions are wrong.
            # For robustness, one might cap iterations, but better to ensure inputs are correct.

        for idx_label, label in enumerate(labels):
            tasks_for_current_label = []

            for i in range(images_per_segment):  # 'i' is local index within this label's current segment
                # Filename uses a counter from idx_start up to idx_end-1 for each label
                naming_idx = idx_start + i
                additional_name = f"{label}_{naming_idx}.png"
                filename = path_obj / additional_name

                # Calculate the actual index in the flat 'images' list
                image_actual_idx = (images_per_segment * idx_label) + i

                if image_actual_idx >= len(images):
                    all_failed_messages.append(
                        f"Skipped saving for label {label}, name {additional_name} due to index out of bounds."
                    )
                    total_images_failed += 1
                    continue

                image_data = images[image_actual_idx]
                tasks_for_current_label.append((image_data, filename))

            if not tasks_for_current_label:
                if images_per_segment > 0:  # Only print if we expected tasks
                    print(f"No valid tasks generated for class {label} (images {idx_start}-{idx_end-1}).")
                continue

            current_batch_workers = min(max_workers, len(tasks_for_current_label))

            # Guard against creating an executor with 0 workers if tasks list became empty due to errors
            if current_batch_workers == 0:
                print(f"Skipping saving for label {label} as no workers or tasks available.")
                continue

            print(f"Processing {len(tasks_for_current_label)} images for class {label} to {path_obj}")

            results_current_label = []
            with ThreadPoolExecutor(max_workers=current_batch_workers) as executor:
                futures = [
                    executor.submit(process_and_save_image, img_data, fname)
                    for img_data, fname in tasks_for_current_label
                ]
                for future in tqdm(
                    as_completed(futures), total=len(tasks_for_current_label), desc=f"Saving for label {label}"
                ):
                    results_current_label.append(future.result())

            successful_current = sum(1 for r in results_current_label if r is None)
            failed_current = [r for r in results_current_label if r is not None]

            total_images_processed_successfully += successful_current
            if failed_current:
                total_images_failed += len(failed_current)
                all_failed_messages.extend(failed_current)
                print(
                    f"{successful_current} images for class {label} saved in {path_obj}. {len(failed_current)} failed."
                )
            else:
                print(f"{successful_current} images for class {label} are saved in path {path_obj}")

        if (
            total_images_failed > 0 or len(all_failed_messages) > 0
        ):  # Print summary if any errors or non-image processing failures
            print("\n--- Overall Summary ---")
            print(f"Total {total_images_processed_successfully} images saved successfully across all labels.")
            print(f"Total {total_images_failed} images/operations failed or were skipped:")
            for error_msg in all_failed_messages:  # all_failed_messages may include structural errors
                print(f" - {error_msg}")
        elif labels is None:  # If labels is None, this part is skipped. This is for the else block.
            print("\n--- Overall Summary ---")
            print(f"Total {total_images_processed_successfully} images saved successfully across all labels.")


# # For class-conditional generation
# def save_images(images, path, idx_start, idx_end):
#     path = Path(path)
#     path.mkdir(parents=True, exist_ok=True)
#     for idx in tqdm(range(idx_start, idx_end)):
#         additional_name = f"{idx}.png"
#         filename = Path(path) / additional_name

#         img = images[idx - idx_start]
#         img = img.cpu().numpy().transpose(1, 2, 0)
#         pil_img = Image.fromarray(img)

#         pil_img.save(filename)

#     print(f"Images from {idx_start} to {idx_end} are saved in {path}")


def main():
    pass


if __name__ == "__main__":
    main()
