"""Inference."""

import argparse
import io
import os
from typing import Any, Dict, Optional

import numpy as np
import ray
import ray.data
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

# Configuration constants
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetModel:
    """ResNet model for batch inference using Ray Data."""

    def __init__(self, model_name: str = "resnet50"):
        """Initialize ResNet model.

        Args:
            model_name: Name of the ResNet model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self) -> nn.Module:
        """Load and configure the pre-trained ResNet model."""
        if self.model_name == "resnet18":
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )
        elif self.model_name == "resnet34":
            model = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            )
        elif self.model_name == "resnet50":
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )
        elif self.model_name == "resnet101":
            model = torchvision.models.resnet101(
                weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model.eval()
        model.to(self.device)
        return model

    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform inference on a batch of images.

        Args:
            batch: Dictionary containing image data from Ray Data

        Returns:
            Dictionary with prediction results
        """
        batch_size = len(batch["bytes"])
        images = []

        # Process each image in the batch
        for i in range(batch_size):
            try:
                # Get image bytes
                img_bytes = batch["bytes"][i]

                # Convert bytes to BytesIO object for PIL
                img_buffer = io.BytesIO(img_bytes)

                # Convert BytesIO to PIL Image
                img = Image.open(img_buffer).convert("RGB")

                # Apply transforms
                img_tensor = self.transform(img)
                images.append(img_tensor)

            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Create a dummy tensor for failed images
                dummy_tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
                images.append(dummy_tensor)

        # Stack into batch tensor
        batch_tensor = torch.stack(images).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get predictions
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]

            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)

        # Return results as numpy arrays
        return {
            "predicted_class": predicted_classes.cpu().numpy(),
            "confidence_score": confidence_scores.cpu().numpy(),
            "top_5_classes": top5_indices.cpu().numpy(),
            "top_5_scores": top5_probs.cpu().numpy(),
            "file_path": batch.get("path", [None] * batch_size),
        }


def load_images_from_s3(
    s3_uri: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
) -> ray.data.Dataset:
    """Load images from S3 using Ray Data.

    Args:
        s3_uri: S3 URI (e.g., 's3://bucket/path/to/images/')
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token (for temporary credentials)

    Returns:
        Ray Dataset containing images from S3
    """
    # Set up AWS credentials if provided
    filesystem_kwargs = {}
    if aws_access_key_id:
        filesystem_kwargs = {
            "access_key_id": aws_access_key_id,
            "secret_access_key": aws_secret_access_key,
        }
        if aws_session_token:
            filesystem_kwargs["session_token"] = aws_session_token

    # Read binary files from S3
    # Ray Data will automatically handle image file formats
    ds = ray.data.read_binary_files(
        s3_uri,
        filesystem=filesystem_kwargs if filesystem_kwargs else None,
        include_paths=True,
    )

    return ds


def filter_image_files(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Filter to keep only image files based on file extension.

    Args:
        batch: Batch from Ray Data

    Returns:
        Filtered batch containing only image files
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Get indices of image files
    image_indices = []
    for i, path in enumerate(batch["path"]):
        if any(path.lower().endswith(ext) for ext in image_extensions):
            image_indices.append(i)

    # Filter batch to include only image files
    if image_indices:
        filtered_batch = {}
        for key, values in batch.items():
            filtered_batch[key] = [values[i] for i in image_indices]
        return filtered_batch
    else:
        # Return empty batch if no images found
        return {key: [] for key in batch.keys()}


def run_resnet_batch_prediction(
    s3_uri: str,
    model_name: str = "resnet50",
    batch_size: int = 32,
    num_gpus: float = 1.0,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
) -> ray.data.Dataset:
    """Run ResNet batch prediction on images from S3.

    Args:
        s3_uri: S3 URI containing images
        model_name: ResNet model name
        batch_size: Batch size for inference
        num_gpus: Number of GPUs to use
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token

    Returns:
        Ray Dataset with predictions
    """
    print(f"Loading images from S3: {s3_uri}")

    # Load images from S3
    ds = load_images_from_s3(
        s3_uri=s3_uri,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    print(f"Loaded {ds.count()} files from S3")

    # Filter to keep only image files
    ds = ds.map_batches(filter_image_files, batch_format="numpy")

    print(f"Found {ds.count()} image files")

    # Run inference using ResNet model
    print(f"Running batch inference with {model_name}")
    predictions = ds.map_batches(
        ResNetModel,
        fn_constructor_kwargs={"model_name": model_name},
        batch_size=batch_size,
        concurrency=1,
        num_gpus=num_gpus,
        batch_format="numpy",
    )

    return predictions


def save_predictions_locally(predictions: ray.data.Dataset, output_path: str) -> None:
    """Save predictions to local filesystem using Ray Data.

    Args:
        predictions: Ray Dataset containing predictions
        output_path: Local path to save results
    """
    print(f"Saving predictions locally to: {output_path}")

    # Determine output format based on file extension
    if output_path.endswith(".parquet") or output_path.endswith("/"):
        # Save as Parquet (directory of files)
        parquet_path = output_path if output_path.endswith("/") else output_path
        predictions.write_parquet(parquet_path)
        print(f"Predictions saved as Parquet to {parquet_path}")

    elif output_path.endswith(".json"):
        # Save as JSON Lines
        predictions.write_json(output_path)
        print(f"Predictions saved as JSON to {output_path}")

    elif output_path.endswith(".csv"):
        # Save as CSV
        predictions.write_csv(output_path)
        print(f"Predictions saved as CSV to {output_path}")

    else:
        # Default to Parquet directory
        predictions.write_parquet(output_path)
        print(f"Predictions saved as Parquet directory to {output_path}")


def show_sample_predictions(
    predictions: ray.data.Dataset, num_samples: int = 5
) -> None:
    """Display sample predictions from the dataset.

    Args:
        predictions: Ray Dataset containing predictions
        num_samples: Number of sample predictions to show
    """
    print(f"\nSample predictions (first {num_samples}):")

    # Take sample predictions
    samples = predictions.take(num_samples)

    for i, sample in enumerate(samples):
        print(f"\nPrediction {i + 1}:")
        if sample.get("file_path"):
            print(f"  File: {os.path.basename(sample['file_path'])}")
        print(f"  Predicted class: {sample['predicted_class']}")
        print(f"  Confidence: {sample['confidence_score']:.4f}")
        print(f"  Top 5 classes: {sample['top_5_classes'].tolist()}")
        print(f"  Top 5 scores: {[f'{score:.4f}' for score in sample['top_5_scores']]}")


def main() -> None:
    """Main function to run ResNet batch prediction on S3 data."""
    parser = argparse.ArgumentParser(
        description="Ray Data ResNet Batch Prediction from S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # S3 source arguments
    parser.add_argument(
        "--s3-uri",
        type=str,
        default="s3://anonymous@air-example-data-2/imagenette2/train/",
        help="S3 URI containing images (e.g., s3://bucket/path/to/images/)",
    )
    parser.add_argument(
        "--aws-access-key-id",
        type=str,
        help="AWS access key ID (if not using IAM role/profile)",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        type=str,
        help="AWS secret access key (if not using IAM role/profile)",
    )
    parser.add_argument(
        "--aws-session-token",
        type=str,
        help="AWS session token (for temporary credentials)",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="ResNet model variant to use",
    )

    # Processing arguments
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0.0,
        help="Number of GPUs to use for inference",
    )

    # Output arguments
    parser.add_argument(
        "--output-path",
        type=str,
        default="./predictions",
        help="Local path to save prediction results (.csv, .json, .parquet, or directory)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of sample predictions to display",
    )

    # Testing arguments
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick test with smaller batch size",
    )

    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    try:
        # Adjust parameters for smoke test
        batch_size = min(args.batch_size, 4) if args.smoke_test else args.batch_size

        # Run batch prediction
        predictions = run_resnet_batch_prediction(
            s3_uri=args.s3_uri,
            model_name=args.model_name,
            batch_size=batch_size,
            num_gpus=args.num_gpus,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_session_token=args.aws_session_token,
        )

        print("\nBatch prediction completed!")
        print(f"Total predictions: {predictions.count()}")

        # Show sample predictions
        show_sample_predictions(predictions, args.sample_size)

        # Save results locally
        save_predictions_locally(predictions=predictions, output_path=args.output_path)

        print("\nBatch prediction pipeline completed successfully!")

    except Exception as e:
        print(f"Error during batch prediction: {e}")
        raise
    finally:
        # Clean up Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
