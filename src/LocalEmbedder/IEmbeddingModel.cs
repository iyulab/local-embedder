namespace LocalEmbedder;

/// <summary>
/// Represents a loaded embedding model that can generate text embeddings.
/// </summary>
public interface IEmbeddingModel : IDisposable, IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the embedding vector dimension.
    /// </summary>
    int Dimensions { get; }

    /// <summary>
    /// Generates an embedding for a single text.
    /// </summary>
    ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates embeddings for multiple texts in batch.
    /// </summary>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);
}
