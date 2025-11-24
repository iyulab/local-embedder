using System.Buffers;
using LocalEmbedder.Inference;
using LocalEmbedder.Pooling;
using LocalEmbedder.Tokenization;
using LocalEmbedder.Utils;

namespace LocalEmbedder;

/// <summary>
/// Implementation of IEmbeddingModel with memory pooling and batch parallelization.
/// </summary>
internal sealed class EmbeddingModel : IEmbeddingModel
{
    private readonly OnnxInferenceEngine _engine;
    private readonly ITokenizer _tokenizer;
    private readonly IPoolingStrategy _poolingStrategy;
    private readonly EmbedderOptions _options;
    private bool _disposed;

    public string ModelId { get; }
    public int Dimensions => _engine.HiddenSize;

    internal EmbeddingModel(
        string modelId,
        OnnxInferenceEngine engine,
        ITokenizer tokenizer,
        IPoolingStrategy poolingStrategy,
        EmbedderOptions options)
    {
        ModelId = modelId;
        _engine = engine;
        _tokenizer = tokenizer;
        _poolingStrategy = poolingStrategy;
        _options = options;
    }

    public async ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Tokenize
        var (inputIds, attentionMask) = _tokenizer.Encode(text, _options.MaxSequenceLength);

        // Run inference
        var tokenEmbeddings = await Task.Run(
            () => _engine.RunInference(inputIds, attentionMask),
            cancellationToken);

        // Pool to sentence embedding using pooled buffer
        int seqLength = inputIds.Length;
        var result = new float[Dimensions];

        _poolingStrategy.Pool(
            tokenEmbeddings.AsSpan(),
            attentionMask.AsSpan(),
            result.AsSpan(),
            seqLength,
            Dimensions);

        // Normalize if requested
        if (_options.NormalizeEmbeddings)
        {
            VectorOperations.NormalizeL2(result.AsSpan());
        }

        return result;
    }

    public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (texts.Count == 0)
            return [];

        // Tokenize all texts
        var (allInputIds, allAttentionMasks) = _tokenizer.EncodeBatch(texts, _options.MaxSequenceLength);

        // Run batch inference with parallelization
        var allTokenEmbeddings = await Task.Run(
            () => _engine.RunBatchInferenceParallel(allInputIds, allAttentionMasks),
            cancellationToken);

        // Pool each to sentence embedding with parallel processing
        var results = new float[texts.Count][];
        int seqLength = _options.MaxSequenceLength;
        int hiddenDim = Dimensions;
        bool normalize = _options.NormalizeEmbeddings;

        Parallel.For(0, texts.Count, i =>
        {
            // Rent buffer from pool for intermediate work
            var resultBuffer = ArrayPool<float>.Shared.Rent(hiddenDim);
            try
            {
                var resultSpan = resultBuffer.AsSpan(0, hiddenDim);

                _poolingStrategy.Pool(
                    allTokenEmbeddings[i].AsSpan(),
                    allAttentionMasks[i].AsSpan(),
                    resultSpan,
                    seqLength,
                    hiddenDim);

                if (normalize)
                {
                    VectorOperations.NormalizeL2(resultSpan);
                }

                // Copy to final result array
                results[i] = resultSpan.ToArray();
            }
            finally
            {
                ArrayPool<float>.Shared.Return(resultBuffer);
            }
        });

        return results;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _engine.Dispose();
            _disposed = true;
        }
    }
}
