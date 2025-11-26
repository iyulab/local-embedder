namespace LocalEmbedder;

/// <summary>
/// Configuration options for the embedding model.
/// </summary>
public sealed class EmbedderOptions
{
    /// <summary>
    /// Gets or sets the directory for caching downloaded models.
    /// Defaults to ~/.cache/huggingface/hub/
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the maximum sequence length for tokenization.
    /// Defaults to 512.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to normalize embeddings to unit vectors (L2 normalization).
    /// Defaults to true.
    /// </summary>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the execution provider for inference.
    /// Defaults to Auto (automatically selects the best available provider).
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

    /// <summary>
    /// Gets or sets the pooling mode for sentence embeddings.
    /// Defaults to Mean pooling.
    /// </summary>
    public PoolingMode PoolingMode { get; set; } = PoolingMode.Mean;

    /// <summary>
    /// Gets or sets whether to convert text to lowercase before tokenization.
    /// Defaults to true (for uncased models).
    /// </summary>
    public bool DoLowerCase { get; set; } = true;

    /// <summary>
    /// Gets the default cache directory path.
    /// </summary>
    public static string GetDefaultCacheDirectory()
    {
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(userProfile, ".cache", "huggingface", "hub");
    }
}

/// <summary>
/// Specifies the execution provider for ONNX Runtime inference.
/// </summary>
public enum ExecutionProvider
{
    /// <summary>
    /// Automatically select the best available provider.
    /// Selection order: CUDA → DirectML (Windows) / CoreML (macOS) → CPU.
    /// This is the recommended default for zero-configuration usage.
    /// </summary>
    Auto,

    /// <summary>
    /// CPU execution (works everywhere).
    /// </summary>
    Cpu,

    /// <summary>
    /// NVIDIA CUDA GPU acceleration.
    /// </summary>
    Cuda,

    /// <summary>
    /// Windows DirectML GPU acceleration (AMD, Intel, NVIDIA).
    /// </summary>
    DirectML,

    /// <summary>
    /// Apple CoreML acceleration.
    /// </summary>
    CoreML
}

/// <summary>
/// Specifies the pooling strategy for sentence embeddings.
/// </summary>
public enum PoolingMode
{
    /// <summary>
    /// Mean pooling of all token embeddings (default, best for most models).
    /// </summary>
    Mean,

    /// <summary>
    /// Use the [CLS] token embedding (required for BGE models).
    /// </summary>
    Cls,

    /// <summary>
    /// Max pooling across all tokens.
    /// </summary>
    Max
}
