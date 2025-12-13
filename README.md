# ⚠️ This Repository is Archived

> **This project has been superseded by [LocalAI.Embedder](https://github.com/iyulab/local-ai).**
>
> All future development, bug fixes, and new features will be in the LocalAI project.

---

## Migration Guide

### 1. Replace Package

```bash
# Remove old package
dotnet remove package LocalEmbedder

# Add new package
dotnet add package LocalAI.Embedder
```

### 2. Update Namespace

```diff
- using LocalEmbedder;
+ using LocalAI.Embedder;
```

### 3. Update Code

The API is nearly identical:

```csharp
// Before (LocalEmbedder)
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");
float[] embedding = await model.EmbedAsync("Hello, world!");

// After (LocalAI.Embedder)
await using var model = await LocalEmbedder.LoadAsync("default");
float[] embedding = await model.EmbedAsync("Hello, world!");
```

### Model Name Mapping

| LocalEmbedder | LocalAI.Embedder |
|---------------|------------------|
| `all-MiniLM-L6-v2` | `fast` |
| `bge-small-en-v1.5` | `default` |
| `bge-base-en-v1.5` | `quality` |
| `multilingual-e5-base` | `multilingual` |

Or continue using full model names directly.

---

## Why Migrate?

**LocalAI** provides a unified suite of local AI capabilities:

| Package | Description |
|---------|-------------|
| [LocalAI.Embedder](https://github.com/iyulab/local-ai) | Text embeddings (successor to this package) |
| [LocalAI.Reranker](https://github.com/iyulab/local-ai) | Semantic reranking for search |
| [LocalAI.Generator](https://github.com/iyulab/local-ai) | Text generation & chat |
| More coming... | OCR, Image captioning, Translation, etc. |

**Benefits:**
- Active development and maintenance
- More models and capabilities
- Unified API across all local AI tasks
- Better documentation and examples

---

## Links

- **New Repository**: https://github.com/iyulab/local-ai
- **NuGet Package**: https://www.nuget.org/packages/LocalAI.Embedder
- **Documentation**: https://github.com/iyulab/local-ai/blob/main/docs/embedder.md

---

## License

MIT License - see [LICENSE](LICENSE) for details.
