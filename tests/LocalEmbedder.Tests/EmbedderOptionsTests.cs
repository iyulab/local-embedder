namespace LocalEmbedder.Tests;

public class EmbedderOptionsTests
{
    [Fact]
    public void DefaultValues_AreCorrect()
    {
        var options = new EmbedderOptions();

        Assert.Null(options.CacheDirectory);
        Assert.Equal(512, options.MaxSequenceLength);
        Assert.True(options.NormalizeEmbeddings);
        Assert.Equal(ExecutionProvider.Cpu, options.Provider);
        Assert.Equal(PoolingMode.Mean, options.PoolingMode);
        Assert.True(options.DoLowerCase);
    }

    [Fact]
    public void GetDefaultCacheDirectory_ReturnsValidPath()
    {
        var path = EmbedderOptions.GetDefaultCacheDirectory();

        Assert.NotNull(path);
        Assert.NotEmpty(path);
        Assert.Contains(".cache", path);
        Assert.Contains("huggingface", path);
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        var options = new EmbedderOptions
        {
            CacheDirectory = "/custom/path",
            MaxSequenceLength = 256,
            NormalizeEmbeddings = false,
            Provider = ExecutionProvider.Cuda,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false
        };

        Assert.Equal("/custom/path", options.CacheDirectory);
        Assert.Equal(256, options.MaxSequenceLength);
        Assert.False(options.NormalizeEmbeddings);
        Assert.Equal(ExecutionProvider.Cuda, options.Provider);
        Assert.Equal(PoolingMode.Cls, options.PoolingMode);
        Assert.False(options.DoLowerCase);
    }
}
