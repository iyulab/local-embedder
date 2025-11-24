# LocalEmbedder

[![CI](https://github.com/iyulab/local-embedder/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/local-embedder/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/LocalEmbedder.svg)](https://www.nuget.org/packages/LocalEmbedder)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A simple, high-performance .NET library for local text embeddings with automatic model downloading from HuggingFace.

```csharp
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");
var embedding = await model.EmbedAsync("Hello, world!");
```

## Features

- **Zero Configuration** - Just specify a model name, auto-downloads from HuggingFace
- **High Performance** - SIMD-optimized operations with TensorPrimitives
- **Minimal Dependencies** - Only ONNX Runtime and System.Numerics.Tensors
- **GPU Acceleration** - CUDA, DirectML, CoreML support
- **Pre-configured Models** - Popular embedding models ready to use
- **Resume Downloads** - Automatically resumes interrupted downloads

## Installation

```bash
dotnet add package LocalEmbedder
```

## Quick Start

```csharp
using LocalEmbedder;

// Load a model (auto-downloads if not cached)
using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

// Generate embedding
float[] embedding = await model.EmbedAsync("Your text here");

// Batch processing
float[][] embeddings = await model.EmbedAsync([
    "First sentence",
    "Second sentence",
    "Third sentence"
]);
```

## Available Models

| Model ID | Dimensions | Description |
|----------|------------|-------------|
| `all-MiniLM-L6-v2` | 384 | Fast, good quality general-purpose |
| `all-mpnet-base-v2` | 768 | High quality general-purpose |
| `bge-small-en-v1.5` | 384 | BAAI's efficient model |
| `bge-base-en-v1.5` | 768 | BAAI's high quality model |
| `multilingual-e5-small` | 384 | Multilingual support |
| `multilingual-e5-base` | 768 | Multilingual, high quality |

```csharp
// List all available models
foreach (var modelId in LocalEmbedder.GetAvailableModels())
{
    Console.WriteLine(modelId);
}
```

## Configuration

```csharp
var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2", new EmbedderOptions
{
    CacheDirectory = "./models",           // Model cache location
    MaxSequenceLength = 512,               // Max tokens
    NormalizeEmbeddings = true,            // L2 normalization
    Provider = ExecutionProvider.Cuda      // GPU acceleration
});
```

### Execution Providers

```csharp
ExecutionProvider.Cpu       // Default, works everywhere
ExecutionProvider.Cuda      // NVIDIA GPU
ExecutionProvider.DirectML  // Windows GPU (AMD, Intel, NVIDIA)
ExecutionProvider.CoreML    // Apple Silicon
```

## Advanced Usage

### Custom HuggingFace Model

```csharp
// Load any ONNX model from HuggingFace
var model = await LocalEmbedder.LoadAsync("intfloat/multilingual-e5-large");
```

### Local Model File

```csharp
// Load from local path
var model = await LocalEmbedder.LoadAsync("/path/to/model.onnx");
```

### Dependency Injection

```csharp
// Register as singleton
services.AddSingleton<IEmbeddingModel>(sp =>
    LocalEmbedder.LoadAsync("all-MiniLM-L6-v2").GetAwaiter().GetResult());

// Or use factory
services.AddSingleton<IEmbedderFactory, EmbedderFactory>();
```

### Similarity Search

```csharp
using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

var query = await model.EmbedAsync("What is machine learning?");
var documents = await model.EmbedAsync([
    "Machine learning is a subset of AI",
    "The weather is nice today",
    "Deep learning uses neural networks"
]);

// Find most similar
var similarities = documents.Select(doc => 
    LocalEmbedder.CosineSimilarity(query, doc));
```

## API Reference

### IEmbeddingModel

```csharp
public interface IEmbeddingModel : IDisposable
{
    string ModelId { get; }
    int Dimensions { get; }
    
    ValueTask<float[]> EmbedAsync(string text, CancellationToken ct = default);
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
}
```

### Utility Methods

```csharp
// Cosine similarity between two vectors
float similarity = LocalEmbedder.CosineSimilarity(vec1, vec2);

// Euclidean distance
float distance = LocalEmbedder.EuclideanDistance(vec1, vec2);

// Dot product
float dot = LocalEmbedder.DotProduct(vec1, vec2);
```

## Performance Tips

1. **Reuse model instances** - Loading is expensive, embed calls are cheap
2. **Batch your requests** - `EmbedAsync(string[])` is more efficient than multiple single calls
3. **Use GPU** - Significant speedup for large batches
4. **Choose the right model** - Smaller models (MiniLM) are much faster than large ones (E5-large)

## Requirements

- .NET 10.0 or later
- ONNX Runtime native libraries (included via NuGet)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
