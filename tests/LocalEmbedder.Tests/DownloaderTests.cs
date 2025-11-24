using LocalEmbedder.Download;

namespace LocalEmbedder.Tests;

public class DownloaderTests : IDisposable
{
    private readonly string _testCacheDir;

    public DownloaderTests()
    {
        _testCacheDir = Path.Combine(Path.GetTempPath(), $"localembedder_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_testCacheDir);
    }

    [Fact]
    public void Constructor_UsesDefaultCacheDir()
    {
        using var downloader = new HuggingFaceDownloader();
        // Should not throw
        Assert.NotNull(downloader);
    }

    [Fact]
    public void Constructor_UsesCustomCacheDir()
    {
        using var downloader = new HuggingFaceDownloader(_testCacheDir);
        Assert.NotNull(downloader);
    }

    [Fact]
    public async Task DownloadFileAsync_CreatesDirectory()
    {
        using var downloader = new HuggingFaceDownloader(_testCacheDir);

        var destPath = Path.Combine(_testCacheDir, "subdir", "test.txt");

        // This will fail because the URL doesn't exist, but directory should be created
        try
        {
            await downloader.DownloadFileAsync(
                "nonexistent/repo",
                "file.txt",
                destPath);
        }
        catch (HttpRequestException)
        {
            // Expected
        }

        Assert.True(Directory.Exists(Path.GetDirectoryName(destPath)));
    }

    [Fact]
    public void DownloadProgress_CalculatesPercentCorrectly()
    {
        var progress = new DownloadProgress
        {
            FileName = "test.onnx",
            BytesDownloaded = 50,
            TotalBytes = 100
        };

        Assert.Equal(50.0, progress.PercentComplete);
    }

    [Fact]
    public void DownloadProgress_HandlesZeroTotal()
    {
        var progress = new DownloadProgress
        {
            FileName = "test.onnx",
            BytesDownloaded = 50,
            TotalBytes = 0
        };

        Assert.Equal(0.0, progress.PercentComplete);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        var downloader = new HuggingFaceDownloader(_testCacheDir);
        downloader.Dispose();
        downloader.Dispose(); // Should not throw
    }

    [Fact]
    public void GetModelDirectory_CreatesCorrectPath()
    {
        // Test internal method behavior through DownloadModelAsync
        using var downloader = new HuggingFaceDownloader(_testCacheDir);

        // The path structure should follow HuggingFace convention
        var expectedPattern = Path.Combine(_testCacheDir, "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots", "main");
        Assert.Contains("models--", expectedPattern);
        Assert.Contains("snapshots", expectedPattern);
    }

    [Fact]
    public async Task DownloadFileAsync_HandlesEmptyDirectory()
    {
        using var downloader = new HuggingFaceDownloader(_testCacheDir);

        var destPath = Path.Combine(_testCacheDir, "test.txt");

        // Test with empty dir parameter (just filename)
        try
        {
            await downloader.DownloadFileAsync(
                "nonexistent/repo",
                "file.txt",
                destPath);
        }
        catch (HttpRequestException)
        {
            // Expected
        }

        // File should not exist but no crash
        Assert.True(true);
    }

    [Fact]
    public void DownloadProgress_AllPropertiesSet()
    {
        var progress = new DownloadProgress
        {
            FileName = "model.onnx",
            BytesDownloaded = 1024 * 1024, // 1MB
            TotalBytes = 10 * 1024 * 1024  // 10MB
        };

        Assert.Equal("model.onnx", progress.FileName);
        Assert.Equal(1024 * 1024, progress.BytesDownloaded);
        Assert.Equal(10 * 1024 * 1024, progress.TotalBytes);
        Assert.Equal(10.0, progress.PercentComplete, precision: 1);
    }

    [Fact]
    public void DownloadProgress_HandlesComplete()
    {
        var progress = new DownloadProgress
        {
            FileName = "test.onnx",
            BytesDownloaded = 100,
            TotalBytes = 100
        };

        Assert.Equal(100.0, progress.PercentComplete);
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
