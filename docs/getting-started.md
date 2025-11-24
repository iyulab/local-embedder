# Getting Started with LocalEmbedder

## Installation

Install LocalEmbedder via NuGet:

```bash
dotnet add package LocalEmbedder
```

Or via Package Manager Console:

```powershell
Install-Package LocalEmbedder
```

## Basic Usage

### 1. Load a Model

```csharp
using LocalEmbedder;

// Load a pre-configured model (auto-downloads from HuggingFace)
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");
```

### 2. Generate Embeddings

```csharp
// Single text
var embedding = await model.EmbedAsync("Hello, world!");
Console.WriteLine($"Dimensions: {embedding.Length}"); // 384

// Multiple texts (batch processing)
var texts = new[] { "First text", "Second text", "Third text" };
var embeddings = await model.EmbedAsync(texts);
```

### 3. Compute Similarity

```csharp
var text1 = await model.EmbedAsync("Machine learning is fascinating");
var text2 = await model.EmbedAsync("AI and ML are related topics");
var text3 = await model.EmbedAsync("The weather is nice today");

var sim1 = LocalEmbedder.CosineSimilarity(text1, text2); // ~0.7
var sim2 = LocalEmbedder.CosineSimilarity(text1, text3); // ~0.1
```

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| General English | `all-MiniLM-L6-v2` | Fast, good quality |
| High quality English | `all-mpnet-base-v2` | Better accuracy, slower |
| Multilingual | `multilingual-e5-small` | 100+ languages |
| Chinese/Korean/Japanese | `bge-small-en-v1.5` | Optimized for CJK |

## Configuration Options

```csharp
var options = new EmbedderOptions
{
    // Where to cache downloaded models
    CacheDirectory = "/path/to/cache",

    // Maximum tokens per text (longer texts are truncated)
    MaxSequenceLength = 256,

    // Normalize embeddings to unit length
    NormalizeEmbeddings = true,

    // Hardware acceleration
    Provider = ExecutionProvider.Cpu,

    // Pooling strategy
    PoolingMode = PoolingMode.Mean,

    // Lowercase input text
    DoLowerCase = true
};

var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2", options);
```

## GPU Acceleration

### NVIDIA CUDA

```csharp
var options = new EmbedderOptions
{
    Provider = ExecutionProvider.Cuda
};
```

Requirements:
- CUDA Toolkit 11.8 or 12.x
- cuDNN 8.x
- Install `Microsoft.ML.OnnxRuntime.Gpu` NuGet package

### Windows DirectML

```csharp
var options = new EmbedderOptions
{
    Provider = ExecutionProvider.DirectML
};
```

Works with AMD, Intel, and NVIDIA GPUs on Windows 10/11.

### Apple CoreML

```csharp
var options = new EmbedderOptions
{
    Provider = ExecutionProvider.CoreML
};
```

Optimized for Apple Silicon (M1/M2/M3) and Intel Macs.

## Download Progress

Track download progress for large models:

```csharp
var progress = new Progress<DownloadProgress>(p =>
{
    var percent = p.PercentComplete;
    var mb = p.BytesDownloaded / (1024.0 * 1024.0);
    Console.WriteLine($"{p.FileName}: {percent:F1}% ({mb:F1} MB)");
});

var model = await LocalEmbedder.LoadAsync("all-mpnet-base-v2", progress: progress);
```

## Custom Models

### From HuggingFace

```csharp
// Any ONNX model from HuggingFace Hub
var model = await LocalEmbedder.LoadAsync("sentence-transformers/paraphrase-MiniLM-L6-v2");
```

### From Local Path

```csharp
// Local ONNX file (vocab.txt must be in same directory)
var model = await LocalEmbedder.LoadAsync("/models/my-model.onnx");
```

## Cache Location

By default, models are cached at:
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub`
- **Linux/macOS**: `~/.cache/huggingface/hub`

This follows HuggingFace's standard cache structure for compatibility.

## Thread Safety

`IEmbeddingModel` is thread-safe. You can safely call `EmbedAsync` from multiple threads:

```csharp
var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

// Process in parallel
var tasks = texts.Select(text => model.EmbedAsync(text));
var embeddings = await Task.WhenAll(tasks);
```

## Memory Management

Always dispose the model when done:

```csharp
// Using statement (recommended)
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

// Or manual disposal
var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");
try
{
    // use model
}
finally
{
    model.Dispose();
}
```

## Next Steps

- [API Reference](api-reference.md)
- [Examples](examples.md)
- [Performance Tuning](performance.md)
