# Changelog

All notable changes to LocalEmbedder will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-XX

### Added
- **ExecutionProvider.Auto** - Zero-configuration GPU acceleration
  - Automatically selects the best available execution provider
  - Selection priority: CUDA → DirectML (Windows) / CoreML (macOS) → CPU
  - Eliminates need for manual provider configuration
  - Maximizes performance out-of-the-box

### Changed
- **Default ExecutionProvider changed from Cpu to Auto**
  - Users now get automatic GPU acceleration when available
  - Existing code continues to work (backward compatible)
  - Explicit `Provider = ExecutionProvider.Cpu` still works for CPU-only scenarios

### Technical Details
- Added `TryAppendCudaProvider()`, `TryAppendDirectMLProvider()`, `TryAppendCoreMLProvider()` helper methods
- Auto provider uses graceful fallback chain for maximum compatibility
- Platform detection using `OperatingSystem.IsWindows()` and `OperatingSystem.IsMacOS()`

### Migration Guide
```csharp
// Before (v0.2.0) - CPU only by default
await using var model = await LocalEmbedder.LoadAsync("model-id");
// ^ Used CPU even if GPU was available

// After (v0.3.0) - Auto GPU acceleration
await using var model = await LocalEmbedder.LoadAsync("model-id");
// ^ Automatically uses GPU if available, falls back to CPU

// Force CPU-only if needed
await using var model = await LocalEmbedder.LoadAsync("model-id", new EmbedderOptions
{
    Provider = ExecutionProvider.Cpu
});
```

---

## [0.2.0] - 2025-01-XX

### Added
- **IAsyncDisposable Support** ([Issue #3](https://github.com/iyulab/local-embedder/issues/3))
  - `IEmbeddingModel` now implements `IAsyncDisposable` in addition to `IDisposable`
  - Enables modern `await using` pattern for resource disposal
  - Both `using` and `await using` patterns are fully supported
  - No breaking changes - existing code continues to work

- **DownloadProgress Namespace Export** ([Issue #1](https://github.com/iyulab/local-embedder/issues/1))
  - `DownloadProgress` type is now available in the main `LocalEmbedder` namespace
  - Users no longer need `using LocalEmbedder.Download;` for progress reporting
  - Improved API discoverability and developer experience

### Changed
- Recommended pattern for model disposal is now `await using` (backward compatible)
- Documentation updated to show `await using` as the preferred pattern

### Technical Details
- Added `GlobalUsings.cs` for type re-exports
- `EmbeddingModel.Dispose()` now calls `DisposeAsync()` internally
- Proper disposal guard to prevent double-disposal
- `GC.SuppressFinalize()` called in `DisposeAsync()`

### Migration Guide
```csharp
// Old style (still works)
using LocalEmbedder;
using LocalEmbedder.Download; // No longer needed
using var model = await LocalEmbedder.LoadAsync("model-id");

// New style (recommended)
using LocalEmbedder; // Single import
await using var model = await LocalEmbedder.LoadAsync("model-id");
```

---

## [0.1.1] - 2025-01-XX

### Added
- Unit tests for LocalEmbedder functionality
- Comprehensive test coverage for core operations

### Fixed
- Assembly information updates
- Build configuration improvements

---

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- ONNX-based embedding model support
- Automatic model downloading from HuggingFace
- SIMD-optimized vector operations
- GPU acceleration (CUDA, DirectML, CoreML)
- Pre-configured popular models
- Batch processing support
- Memory pooling for efficient inference
- Resume download capability

### Features
- Zero configuration setup
- Minimal dependencies (ONNX Runtime + System.Numerics.Tensors)
- High performance BERT tokenization
- Multiple pooling strategies (Mean, CLS, Max)
- L2 normalization support
- Vector similarity utilities (Cosine, Euclidean, Dot Product)

[0.2.0]: https://github.com/iyulab/local-embedder/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/iyulab/local-embedder/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/iyulab/local-embedder/releases/tag/v0.1.0
