// Global usings for LocalEmbedder library

// Re-export DownloadProgress from LocalEmbedder.Download namespace for better API discoverability.
// This allows users to reference DownloadProgress with only "using LocalEmbedder;"
// instead of requiring "using LocalEmbedder.Download;".
// See Issue #1: https://github.com/iyulab/local-embedder/issues/1
global using DownloadProgress = LocalEmbedder.Download.DownloadProgress;
