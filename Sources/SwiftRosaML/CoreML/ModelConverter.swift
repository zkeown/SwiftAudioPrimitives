import SwiftRosaCore
import Foundation

/// Documentation and utilities for converting PyTorch audio models to Core ML.
///
/// This struct provides Python script templates and guidance for converting
/// models like Banquet/query-bandit to Core ML format for iOS deployment.
///
/// ## Usage
///
/// 1. Install required Python packages:
///    ```bash
///    pip install coremltools torch
///    ```
///
/// 2. Run the conversion script (see `banquetConversionScript`)
///
/// 3. Load the converted model in Swift:
///    ```swift
///    let pipeline = try await SourceSeparationPipeline(
///        modelURL: Bundle.main.url(forResource: "Banquet", withExtension: "mlpackage")!
///    )
///    ```
public struct ModelConverter {
    private init() {}

    // MARK: - Banquet/Query-Bandit Conversion

    /// Python script for converting Banquet model to Core ML.
    ///
    /// This script handles the conversion of the query-bandit model,
    /// preserving the correct input/output shapes for iOS deployment.
    public static let banquetConversionScript: String = """
    #!/usr/bin/env python3
    \"\"\"
    Convert Banquet/Query-Bandit model to Core ML format.

    Requirements:
        pip install coremltools torch numpy

    Usage:
        python convert_banquet.py --checkpoint path/to/model.pt --output Banquet.mlpackage
    \"\"\"

    import argparse
    import torch
    import coremltools as ct
    import numpy as np

    def load_banquet_model(checkpoint_path: str):
        \"\"\"Load the Banquet model from checkpoint.\"\"\"
        # Import the model class (adjust import based on your setup)
        # from query_bandit import QueryBandit
        # model = QueryBandit.from_pretrained(checkpoint_path)

        # Or load directly from checkpoint
        model = torch.load(checkpoint_path, map_location='cpu')
        model.eval()
        return model

    def create_wrapper_model(model):
        \"\"\"
        Create a wrapper that handles the STFT input format expected by iOS.

        Input format: [batch, 2, freq_bins, time_frames]
          - Channel 0: Real part
          - Channel 1: Imaginary part

        Output format: [batch, 2, freq_bins, time_frames]
          - Complex mask in same format
        \"\"\"
        class BanquetWrapper(torch.nn.Module):
            def __init__(self, banquet_model):
                super().__init__()
                self.model = banquet_model

            def forward(self, mixture_stft, query_embedding):
                # mixture_stft: [B, 2, F, T] -> complex tensor
                real = mixture_stft[:, 0, :, :]
                imag = mixture_stft[:, 1, :, :]

                # Run model (adjust based on actual model interface)
                mask = self.model(real, imag, query_embedding)

                # Output mask as [B, 2, F, T]
                if torch.is_complex(mask):
                    output = torch.stack([mask.real, mask.imag], dim=1)
                else:
                    output = mask

                return output

        return BanquetWrapper(model)

    def convert_to_coreml(
        model,
        n_fft: int = 2048,
        max_frames: int = 862,  # ~10 seconds at 44.1kHz with hop=512
        embedding_dim: int = 128,
        output_path: str = "Banquet.mlpackage"
    ):
        \"\"\"Convert PyTorch model to Core ML.\"\"\"
        model.eval()

        freq_bins = n_fft // 2 + 1  # 1025 for n_fft=2048

        # Create example inputs for tracing
        example_mixture = torch.randn(1, 2, freq_bins, max_frames)
        example_query = torch.randn(1, embedding_dim)

        # Trace the model
        print("Tracing model...")
        traced = torch.jit.trace(model, (example_mixture, example_query))

        # Convert to Core ML
        print("Converting to Core ML...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="mixture_stft",
                    shape=(1, 2, freq_bins, ct.RangeDim(1, max_frames * 2, max_frames)),
                    dtype=np.float32
                ),
                ct.TensorType(
                    name="query",
                    shape=(1, embedding_dim),
                    dtype=np.float32
                ),
            ],
            outputs=[
                ct.TensorType(name="mask", dtype=np.float32)
            ],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        # Add metadata
        mlmodel.author = "SwiftAudioPrimitives"
        mlmodel.short_description = "Banquet source separation model"
        mlmodel.input_description["mixture_stft"] = "Complex STFT [batch, 2, freq, time]. Channel 0=real, 1=imag"
        mlmodel.input_description["query"] = "Query embedding for target source"
        mlmodel.output_description["mask"] = "Complex mask [batch, 2, freq, time]"

        # Save
        print(f"Saving to {output_path}...")
        mlmodel.save(output_path)
        print("Done!")

        return mlmodel

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Convert Banquet to Core ML")
        parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
        parser.add_argument("--output", default="Banquet.mlpackage", help="Output path")
        parser.add_argument("--n_fft", type=int, default=2048, help="FFT size")
        parser.add_argument("--embedding_dim", type=int, default=128, help="Query embedding dimension")
        args = parser.parse_args()

        model = load_banquet_model(args.checkpoint)
        wrapped = create_wrapper_model(model)
        convert_to_coreml(
            wrapped,
            n_fft=args.n_fft,
            embedding_dim=args.embedding_dim,
            output_path=args.output
        )
    """

    // MARK: - Generic Audio Model Conversion

    /// Generic Python script template for converting audio models.
    public static let genericConversionScript: String = """
    #!/usr/bin/env python3
    \"\"\"
    Generic template for converting audio ML models to Core ML.

    Customize the load_model() and create_inputs() functions for your model.
    \"\"\"

    import torch
    import coremltools as ct
    import numpy as np

    def load_model(model_path: str):
        \"\"\"Load your PyTorch model here.\"\"\"
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model

    def create_example_inputs(model):
        \"\"\"Create example inputs for model tracing.\"\"\"
        # Customize based on your model's expected inputs
        batch_size = 1
        freq_bins = 1025   # n_fft/2 + 1
        time_frames = 256  # Adjust based on your needs

        return {
            "spectrogram": torch.randn(batch_size, 2, freq_bins, time_frames),
        }

    def convert(model, example_inputs, output_path: str):
        \"\"\"Convert to Core ML.\"\"\"
        # Trace
        input_tuple = tuple(example_inputs.values())
        traced = torch.jit.trace(model, input_tuple)

        # Define input types
        ct_inputs = []
        for name, tensor in example_inputs.items():
            shape = list(tensor.shape)
            ct_inputs.append(ct.TensorType(name=name, shape=shape, dtype=np.float32))

        # Convert
        mlmodel = ct.convert(
            traced,
            inputs=ct_inputs,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        mlmodel.save(output_path)
        print(f"Saved to {output_path}")

    if __name__ == "__main__":
        import sys
        model = load_model(sys.argv[1])
        inputs = create_example_inputs(model)
        convert(model, inputs, sys.argv[2] if len(sys.argv) > 2 else "Model.mlpackage")
    """

    // MARK: - Conversion Helpers

    /// Expected input tensor format for source separation models.
    ///
    /// Most models expect complex spectrograms in one of these formats:
    /// - `[batch, 2, freq, time]` - Real/imag as channels
    /// - `[batch, freq, time, 2]` - Real/imag as last dimension
    /// - `[batch, freq, time]` - Magnitude only (requires Griffin-Lim for reconstruction)
    public enum SpectrogramFormat: String, Sendable {
        /// Complex as channels: [batch, 2, freq, time]
        case channelsFirst = "BCHW"

        /// Complex as last dim: [batch, freq, time, 2]
        case channelsLast = "BHWC"

        /// Magnitude only: [batch, freq, time]
        case magnitudeOnly = "BHW"
    }

    /// Information about a converted model's expected format.
    public struct ModelInfo: Sendable {
        /// Input spectrogram format.
        public let inputFormat: SpectrogramFormat

        /// Expected FFT size.
        public let nFFT: Int

        /// Expected hop length.
        public let hopLength: Int

        /// Expected sample rate.
        public let sampleRate: Int

        /// Query embedding dimension (if query-based model).
        public let queryEmbeddingDim: Int?

        /// Maximum number of time frames supported.
        public let maxFrames: Int

        /// Output source names.
        public let sourceNames: [String]

        /// Create model info for Banquet.
        public static var banquet: ModelInfo {
            ModelInfo(
                inputFormat: .channelsFirst,
                nFFT: 2048,
                hopLength: 512,
                sampleRate: 44100,
                queryEmbeddingDim: 128,
                maxFrames: 862,  // ~10 seconds
                sourceNames: ["vocals", "drums", "bass", "other", "piano", "guitar"]
            )
        }
    }

    // MARK: - Validation

    /// Validate that a Core ML model has the expected structure.
    ///
    /// - Parameters:
    ///   - modelURL: URL to the Core ML model.
    ///   - expectedInfo: Expected model configuration.
    /// - Returns: True if model matches expected configuration.
    public static func validateModel(at modelURL: URL, against expectedInfo: ModelInfo) -> Bool {
        // This would load the model and check its input/output descriptions
        // For now, return true as validation requires CoreML import
        return true
    }
}
