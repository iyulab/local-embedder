using System.Net;
using System.Net.Http.Headers;

namespace LocalEmbedder.Download;

/// <summary>
/// Downloads models from HuggingFace Hub with resume support.
/// </summary>
internal sealed class HuggingFaceDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _cacheDir;

    private const string HuggingFaceBaseUrl = "https://huggingface.co";

    public HuggingFaceDownloader(string? cacheDir = null)
    {
        _cacheDir = cacheDir ?? EmbedderOptions.GetDefaultCacheDirectory();

        var handler = new HttpClientHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10
        };

        _httpClient = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromMinutes(30)
        };

        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("LocalEmbedder/1.0");
    }

    /// <summary>
    /// Downloads a model from HuggingFace and returns the local path.
    /// </summary>
    public async Task<string> DownloadModelAsync(
        string repoId,
        string revision = "main",
        string? subfolder = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Create model directory
        var modelDir = GetModelDirectory(repoId, revision);
        Directory.CreateDirectory(modelDir);

        // Required files for embedding models
        var requiredFiles = new[]
        {
            "model.onnx",
            "config.json"
        };

        // Optional files (different models use different tokenizer formats)
        var optionalFiles = new[]
        {
            "vocab.txt",  // Used by some models (e.g., BERT-based)
            "tokenizer.json",  // Used by sentence-transformers models
            "tokenizer_config.json",
            "special_tokens_map.json",
            "1_Pooling/config.json"
        };

        // Download required files (may be in subfolder)
        foreach (var file in requiredFiles)
        {
            var localPath = Path.Combine(modelDir, file);
            if (!File.Exists(localPath))
            {
                await DownloadFileAsync(repoId, file, localPath, revision, subfolder, progress, cancellationToken);
            }
        }

        // Download optional files (always at root level, no subfolder)
        foreach (var file in optionalFiles)
        {
            var localPath = Path.Combine(modelDir, file);
            if (!File.Exists(localPath))
            {
                try
                {
                    await DownloadFileAsync(repoId, file, localPath, revision, null, progress, cancellationToken);
                }
                catch (HttpRequestException)
                {
                    // Optional file not found, skip
                }
            }
        }

        return modelDir;
    }

    /// <summary>
    /// Downloads a single file with resume support.
    /// </summary>
    public async Task DownloadFileAsync(
        string repoId,
        string filename,
        string destinationPath,
        string revision = "main",
        string? subfolder = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Ensure directory exists
        var dir = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        // Build URL using resolve endpoint (handles LFS automatically)
        var url = $"{HuggingFaceBaseUrl}/{repoId}/resolve/{revision}/{(string.IsNullOrEmpty(subfolder) ? "" : subfolder + "/")}{filename}";

        var tempPath = destinationPath + ".part";
        long startPosition = 0;

        // Check for partial download
        if (File.Exists(tempPath))
        {
            startPosition = new FileInfo(tempPath).Length;
        }

        // Create request
        var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (startPosition > 0)
        {
            request.Headers.Range = new RangeHeaderValue(startPosition, null);
        }

        // Send request
        using var response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        // Handle 416 (Range Not Satisfiable) - file already complete
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            if (File.Exists(tempPath))
            {
                File.Move(tempPath, destinationPath, overwrite: true);
            }
            return;
        }

        response.EnsureSuccessStatusCode();

        // Check if this is an LFS pointer file
        var contentLength = response.Content.Headers.ContentLength ?? 0;
        if (contentLength < 1024 && filename.EndsWith(".onnx"))
        {
            // Might be LFS pointer, read and check
            var content = await response.Content.ReadAsStringAsync(cancellationToken);
            if (IsLfsPointer(content))
            {
                throw new InvalidOperationException(
                    $"Received LFS pointer for {filename}. The file should be downloaded via resolve URL. " +
                    "This may indicate a network or redirect issue.");
            }
        }

        // Determine total size
        long totalBytes = response.Content.Headers.ContentLength ?? 0;
        if (response.StatusCode == HttpStatusCode.PartialContent)
        {
            // Get total from Content-Range header
            var contentRange = response.Content.Headers.ContentRange;
            if (contentRange?.Length.HasValue == true)
            {
                totalBytes = contentRange.Length.Value;
            }
            else
            {
                totalBytes = startPosition + (response.Content.Headers.ContentLength ?? 0);
            }
        }
        else
        {
            // Full download, reset position
            startPosition = 0;
        }

        // Open file for writing
        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var fileMode = startPosition > 0 ? FileMode.Append : FileMode.Create;
        await using var fileStream = new FileStream(tempPath, fileMode, FileAccess.Write, FileShare.None, 81920, true);

        // Download with progress
        var buffer = new byte[81920];
        long bytesDownloaded = startPosition;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            bytesDownloaded += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = filename,
                BytesDownloaded = bytesDownloaded,
                TotalBytes = totalBytes
            });
        }

        // Move to final location atomically
        fileStream.Close();
        File.Move(tempPath, destinationPath, overwrite: true);
    }

    private string GetModelDirectory(string repoId, string revision)
    {
        // Use HuggingFace cache structure: models--{org}--{model}/snapshots/{revision}
        var sanitizedRepoId = repoId.Replace("/", "--");
        return Path.Combine(_cacheDir, $"models--{sanitizedRepoId}", "snapshots", revision);
    }

    private static bool IsLfsPointer(string content)
    {
        return content.StartsWith("version https://git-lfs.github.com/spec/v1");
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}

/// <summary>
/// Progress information for model downloads.
/// </summary>
public record DownloadProgress
{
    public required string FileName { get; init; }
    public long BytesDownloaded { get; init; }
    public long TotalBytes { get; init; }

    public double PercentComplete => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;
}