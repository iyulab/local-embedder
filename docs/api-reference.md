# API Reference

## LocalEmbedder (Static Class)

Main entry point for loading models and utility functions.

### Methods

#### LoadAsync

```csharp
public static Task<IEmbeddingModel> LoadAsync(
    string modelIdOrPath,
    EmbedderOptions? options = null,
    IProgress<DownloadProgress>? progress = null)
```

Loads an embedding model.

**Parameters:**
- `modelIdOrPath`: Model ID (e.g., "all-MiniLM-L6-v2"), HuggingFace repo (e.g., "sentence-transformers/all-MiniLM-L6-v2"), or local path to .onnx file
- `options`: Configuration options
- `progress`: Download progress reporter

**Returns:** An `IEmbeddingModel` instance

**Exceptions:**
- `ArgumentException`: Unknown model ID
- `FileNotFoundException`: Local model file not found

#### GetAvailableModels

```csharp
public static IEnumerable<string> GetAvailableModels()
```

Returns list of pre-configured model IDs.

#### CosineSimilarity

```csharp
public static float CosineSimilarity(
    ReadOnlySpan<float> embedding1,
    ReadOnlySpan<float> embedding2)
```

Computes cosine similarity between two vectors.

**Returns:** Value between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite)

**Exceptions:**
- `ArgumentException`: Vectors are empty or have different lengths

#### EuclideanDistance

```csharp
public static float EuclideanDistance(
    ReadOnlySpan<float> embedding1,
    ReadOnlySpan<float> embedding2)
```

Computes Euclidean distance between two vectors.

**Returns:** Distance value >= 0 (0 = identical)

#### DotProduct

```csharp
public static float DotProduct(
    ReadOnlySpan<float> embedding1,
    ReadOnlySpan<float> embedding2)
```

Computes dot product of two vectors.

---

## IEmbeddingModel

Interface for embedding model instances. Implements both `IDisposable` and `IAsyncDisposable` for flexible resource management.

```csharp
public interface IEmbeddingModel : IDisposable, IAsyncDisposable
{
    string ModelId { get; }
    int Dimensions { get; }
    ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default);
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);
}
```

### Properties

```csharp
string ModelId { get; }
int Dimensions { get; }
```

### Methods

#### EmbedAsync (Single)

```csharp
ValueTask<float[]> EmbedAsync(
    string text,
    CancellationToken cancellationToken = default)
```

Generates embedding for a single text.

**Parameters:**
- `text`: Input text
- `cancellationToken`: Cancellation token

**Returns:** Float array of length `Dimensions`

#### EmbedAsync (Batch)

```csharp
ValueTask<float[][]> EmbedAsync(
    IReadOnlyList<string> texts,
    CancellationToken cancellationToken = default)
```

Generates embeddings for multiple texts.

**Returns:** Array of float arrays

#### Dispose

```csharp
void Dispose()
```

Synchronously releases resources. Internally calls `DisposeAsync()` to ensure proper cleanup.

**Usage:** Compatible with traditional `using` statements for backward compatibility.

#### DisposeAsync

```csharp
ValueTask DisposeAsync()
```

Asynchronously releases resources. This is the preferred disposal method.

**Usage:** Use with `await using` statements for modern async patterns.

**Example:**
```csharp
// Recommended (async disposal)
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

// Also supported (sync disposal)
using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");
```

---

## EmbedderOptions

Configuration options for loading models.

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `CacheDirectory` | `string?` | null (uses default) | Model cache directory |
| `MaxSequenceLength` | `int` | 512 | Maximum tokens per input |
| `NormalizeEmbeddings` | `bool` | true | L2 normalize output |
| `Provider` | `ExecutionProvider` | Auto | Hardware acceleration (auto-selects best) |
| `PoolingMode` | `PoolingMode` | Mean | Embedding pooling strategy |
| `DoLowerCase` | `bool` | true | Lowercase input text |

### Static Methods

```csharp
public static string GetDefaultCacheDirectory()
```

Returns the default cache directory path.

---

## ExecutionProvider (Enum)

Hardware acceleration options.

| Value | Description |
|-------|-------------|
| `Auto` | Automatically select best available (default). Priority: CUDA → DirectML/CoreML → CPU |
| `Cpu` | CPU execution (works everywhere) |
| `Cuda` | NVIDIA CUDA acceleration |
| `DirectML` | Windows DirectML (AMD/Intel/NVIDIA) |
| `CoreML` | Apple CoreML (macOS/iOS) |

**Auto Selection Logic:**
1. Tries CUDA first (highest performance for NVIDIA GPUs)
2. Falls back to DirectML on Windows or CoreML on macOS
3. Uses CPU if no GPU acceleration is available

---

## PoolingMode (Enum)

Strategy for combining token embeddings.

| Value | Description |
|-------|-------------|
| `Mean` | Average of all token embeddings (most common) |
| `Cls` | Use [CLS] token embedding only |
| `Max` | Maximum value per dimension |

---

## DownloadProgress

Progress information for model downloads.

### Properties

```csharp
public string FileName { get; init; }
public long BytesDownloaded { get; init; }
public long TotalBytes { get; init; }
public double PercentComplete { get; }
```

---

## Usage Examples

### Basic Usage

```csharp
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");
var embedding = await model.EmbedAsync("Hello, world!");
```

### With Options

```csharp
var options = new EmbedderOptions
{
    MaxSequenceLength = 256,
    Provider = ExecutionProvider.Cuda,
    NormalizeEmbeddings = true
};

await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2", options);
```

### With Progress

```csharp
var progress = new Progress<DownloadProgress>(p =>
    Console.WriteLine($"{p.FileName}: {p.PercentComplete:F1}%"));

await using var model = await LocalEmbedder.LoadAsync("all-mpnet-base-v2", progress: progress);
```

### Batch Processing

```csharp
var texts = new[] { "Text 1", "Text 2", "Text 3" };
var embeddings = await model.EmbedAsync(texts);
```

### Similarity Computation

```csharp
var emb1 = await model.EmbedAsync("Machine learning");
var emb2 = await model.EmbedAsync("Artificial intelligence");
var similarity = LocalEmbedder.CosineSimilarity(emb1, emb2);
```
