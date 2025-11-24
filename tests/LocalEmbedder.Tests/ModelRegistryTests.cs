using LocalEmbedder.Utils;

namespace LocalEmbedder.Tests;

public class ModelRegistryTests
{
    [Fact]
    public void TryGetModel_ReturnsTrue_ForKnownModel()
    {
        var result = ModelRegistry.TryGetModel("all-MiniLM-L6-v2", out var info);

        Assert.True(result);
        Assert.NotNull(info);
        Assert.Equal("sentence-transformers/all-MiniLM-L6-v2", info.RepoId);
        Assert.Equal(384, info.Dimensions);
    }

    [Fact]
    public void TryGetModel_ReturnsFalse_ForUnknownModel()
    {
        var result = ModelRegistry.TryGetModel("unknown-model", out var info);

        Assert.False(result);
        Assert.Null(info);
    }

    [Fact]
    public void TryGetModel_IsCaseInsensitive()
    {
        var result = ModelRegistry.TryGetModel("ALL-MINILM-L6-V2", out var info);

        Assert.True(result);
        Assert.NotNull(info);
    }

    [Fact]
    public void GetAvailableModels_ReturnsNonEmptyList()
    {
        var models = ModelRegistry.GetAvailableModels().ToList();

        Assert.NotEmpty(models);
        Assert.Contains("all-MiniLM-L6-v2", models);
        Assert.Contains("bge-small-en-v1.5", models);
    }

    [Theory]
    [InlineData("all-MiniLM-L6-v2", 384, PoolingMode.Mean)]
    [InlineData("all-mpnet-base-v2", 768, PoolingMode.Mean)]
    [InlineData("bge-small-en-v1.5", 384, PoolingMode.Cls)]
    [InlineData("multilingual-e5-small", 384, PoolingMode.Mean)]
    public void KnownModels_HaveCorrectConfiguration(string modelId, int dimensions, PoolingMode pooling)
    {
        ModelRegistry.TryGetModel(modelId, out var info);

        Assert.NotNull(info);
        Assert.Equal(dimensions, info.Dimensions);
        Assert.Equal(pooling, info.PoolingMode);
    }
}
