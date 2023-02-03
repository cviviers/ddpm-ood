import argparse

from src.trainers.reconstruct import Reconstruct


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--in_ids", help="Location of file with inlier ids.")
    parser.add_argument("--out_ids", help="List of location of file with outlier ids.")
    parser.add_argument(
        "--config_vqvae_file",
        default="None",
        help="Location of VQ-VAE config. None if not training a latent diffusion model.",
    )
    parser.add_argument("--vqvae_checkpoint", help="Path to checkpoint file.")
    parser.add_argument("--config_diffusion_file", help="Location of config.")
    parser.add_argument("--vqvae_uri", help="Path readable by load_model.")

    # inference param
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--augmentation",
        type=int,
        default=0,
        help="Use of augmentation, 1 (True) or 0 (False).",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--first_n_val",
        default=None,
        help="Only run on the first n samples from the val dataset.",
    )
    parser.add_argument(
        "--first_n",
        default=None,
        help="Only run on the first n samples from each dataset.",
    )
    parser.add_argument(
        "--eval_checkpoint",
        default=None,
        help="Select a specific checkpoint to evaluate on.",
    )
    parser.add_argument("--drop_last", default=False, help="Drop last non-complete batch..")
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")
    parser.add_argument("--run_val", type=int, default=1, help="Run reconstructions on val set.")
    parser.add_argument("--run_in", type=int, default=1, help="Run reconstructions on in set.")
    parser.add_argument("--run_out", type=int, default=1, help="Run reconstructions on out set.")

    # sampling options
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of inference steps to use with the PLMS sampler.",
    )
    parser.add_argument(
        "--inference_skip_factor",
        type=int,
        default=1,
        help="Perform fewer reconstructions by skipping some of the t-values as starting points.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    recon = Reconstruct(args)
    recon.reconstruct(args)
