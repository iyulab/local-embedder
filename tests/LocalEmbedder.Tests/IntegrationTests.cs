using LocalEmbedder.Download;

namespace LocalEmbedder.Tests;

/// <summary>
/// Integration tests that require actual HuggingFace downloads.
/// These tests are skipped by default in CI but can be run manually.
/// </summary>
[Trait("Category", "Integration")]
public class IntegrationTests : IDisposable
{
    private readonly string _testCacheDir;

    public IntegrationTests()
    {
        _testCacheDir = Path.Combine(Path.GetTempPath(), $"localembedder_integration_{Guid.NewGuid()}");
        Directory.CreateDirectory(_testCacheDir);
    }

    [Fact(Skip = "Integration test - requires internet. Run manually with: dotnet test --filter Category=Integration")]
    public async Task LoadAsync_SentenceTransformers_DownloadsFromOnnxSubfolder()
    {
        // Arrange
        var options = new EmbedderOptions { CacheDirectory = _testCacheDir };
        var progressReports = new List<DownloadProgress>();
        var progress = new Progress<DownloadProgress>(p => progressReports.Add(p));

        // Act
        await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2", options, progress);

        // Assert
        Assert.NotNull(model);
        Assert.Equal("all-MiniLM-L6-v2", model.ModelId);
        Assert.Equal(384, model.Dimensions);

        // Verify model works
        var embedding = await model.EmbedAsync("This is a test sentence.");
        Assert.Equal(384, embedding.Length);

        // Verify embedding values are reasonable (not all zeros)
        Assert.NotEqual(0f, embedding.Max(Math.Abs));
    }

    [Fact(Skip = "Integration test - requires internet. Run manually with: dotnet test --filter Category=Integration")]
    public async Task LoadAsync_BaaiModel_DownloadsFromRoot()
    {
        // Arrange
        var options = new EmbedderOptions { CacheDirectory = _testCacheDir };

        // Act
        await using var model = await LocalEmbedder.LoadAsync("bge-small-en-v1.5", options);

        // Assert
        Assert.NotNull(model);
        Assert.Equal("bge-small-en-v1.5", model.ModelId);
        Assert.Equal(384, model.Dimensions);

        // Verify model works
        var embedding = await model.EmbedAsync("Test embedding generation.");
        Assert.Equal(384, embedding.Length);
    }

    [Fact(Skip = "Integration test - requires internet. Run manually with: dotnet test --filter Category=Integration")]
    public async Task LoadAsync_MultilingualE5_DownloadsSuccessfully()
    {
        // Arrange
        var options = new EmbedderOptions { CacheDirectory = _testCacheDir };

        // Act
        await using var model = await LocalEmbedder.LoadAsync("multilingual-e5-small", options);

        // Assert
        Assert.NotNull(model);
        Assert.Equal(384, model.Dimensions);

        // Test with non-English text
        var embedding = await model.EmbedAsync("이것은 테스트 문장입니다.");
        Assert.Equal(384, embedding.Length);
    }

    [Fact(Skip = "Integration test - requires internet. Run manually with: dotnet test --filter Category=Integration")]
    public async Task LoadAsync_AllMpnetBase_DownloadsFromOnnxSubfolder()
    {
        // Arrange
        var options = new EmbedderOptions { CacheDirectory = _testCacheDir };

        // Act
        await using var model = await LocalEmbedder.LoadAsync("all-mpnet-base-v2", options);

        // Assert
        Assert.NotNull(model);
        Assert.Equal("all-mpnet-base-v2", model.ModelId);
        Assert.Equal(768, model.Dimensions);

        // Verify model works
        var embedding = await model.EmbedAsync("Test sentence for mpnet model.");
        Assert.Equal(768, embedding.Length);
    }

    [Fact(Skip = "Integration test - requires internet. Run manually with: dotnet test --filter Category=Integration")]
    public async Task CosineSimilarity_SimilarSentences_ReturnsHighScore()
    {
        // Arrange
        var options = new EmbedderOptions { CacheDirectory = _testCacheDir };
        await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2", options);

        // Act
        var embedding1 = await model.EmbedAsync("The quick brown fox jumps over the lazy dog.");
        var embedding2 = await model.EmbedAsync("A fast brown fox leaps over a sleepy dog.");
        var embedding3 = await model.EmbedAsync("The weather forecast predicts rain tomorrow.");

        var similaritySameContext = LocalEmbedder.CosineSimilarity(embedding1, embedding2);
        var similarityDifferentContext = LocalEmbedder.CosineSimilarity(embedding1, embedding3);

        // Assert
        Assert.True(similaritySameContext > 0.7f, $"Similar sentences should have high similarity, got {similaritySameContext}");
        Assert.True(similarityDifferentContext < similaritySameContext, "Different context should have lower similarity");
    }

    public void Dispose()
    {
        if (Directory.Exists(_testCacheDir))
        {
            try
            {
                Directory.Delete(_testCacheDir, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }
}
