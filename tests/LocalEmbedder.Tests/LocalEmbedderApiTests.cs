namespace LocalEmbedder.Tests;

public class LocalEmbedderApiTests
{
    [Fact]
    public void GetAvailableModels_ReturnsKnownModels()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        Assert.NotEmpty(models);
        Assert.Contains("all-MiniLM-L6-v2", models);
    }

    [Fact]
    public void CosineSimilarity_ComputesCorrectly()
    {
        var vec1 = new float[] { 1, 0, 0 };
        var vec2 = new float[] { 1, 0, 0 };

        var result = LocalEmbedder.CosineSimilarity(vec1, vec2);
        Assert.Equal(1.0f, result, precision: 5);
    }

    [Fact]
    public void EuclideanDistance_ComputesCorrectly()
    {
        var vec1 = new float[] { 0, 0 };
        var vec2 = new float[] { 3, 4 };

        var result = LocalEmbedder.EuclideanDistance(vec1, vec2);
        Assert.Equal(5.0f, result, precision: 5);
    }

    [Fact]
    public void DotProduct_ComputesCorrectly()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 4, 5, 6 };

        var result = LocalEmbedder.DotProduct(vec1, vec2);
        Assert.Equal(32.0f, result, precision: 5);
    }

    [Fact]
    public async Task LoadAsync_ThrowsForUnknownModel()
    {
        await Assert.ThrowsAsync<ArgumentException>(
            () => LocalEmbedder.LoadAsync("completely-unknown-model-xyz"));
    }

    [Fact]
    public async Task LoadAsync_ThrowsForMissingLocalFile()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => LocalEmbedder.LoadAsync("/nonexistent/path/model.onnx"));
    }

    [Fact]
    public async Task LoadAsync_AcceptsCustomOptions()
    {
        var options = new EmbedderOptions
        {
            MaxSequenceLength = 128,
            Provider = ExecutionProvider.Cpu
        };

        // This will fail because model doesn't exist, but options should be accepted
        await Assert.ThrowsAsync<ArgumentException>(
            () => LocalEmbedder.LoadAsync("unknown", options));
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnEmptyVectors()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        Assert.Throws<ArgumentException>(() => LocalEmbedder.CosineSimilarity(empty, vec));
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnMismatchedLengths()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2 };

        Assert.Throws<ArgumentException>(() => LocalEmbedder.CosineSimilarity(vec1, vec2));
    }

    [Fact]
    public void EuclideanDistance_ThrowsOnMismatchedLengths()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2, 3, 4 };

        Assert.Throws<ArgumentException>(() => LocalEmbedder.EuclideanDistance(vec1, vec2));
    }

    [Fact]
    public void DotProduct_ThrowsOnEmptyVectors()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        Assert.Throws<ArgumentException>(() => LocalEmbedder.DotProduct(empty, vec));
    }

    [Fact]
    public void CosineSimilarity_WorksWithLargeVectors()
    {
        var vec1 = Enumerable.Range(0, 384).Select(i => (float)i).ToArray();
        var vec2 = Enumerable.Range(0, 384).Select(i => (float)i).ToArray();

        var result = LocalEmbedder.CosineSimilarity(vec1, vec2);
        Assert.Equal(1.0f, result, precision: 5);
    }

    [Theory]
    [InlineData("all-MiniLM-L6-v2")]
    [InlineData("all-mpnet-base-v2")]
    [InlineData("bge-small-en-v1.5")]
    public void GetAvailableModels_ContainsExpectedModels(string expectedModel)
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();
        Assert.Contains(expectedModel, models);
    }
}
