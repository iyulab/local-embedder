using LocalEmbedder.Tokenization;

namespace LocalEmbedder.Tests;

public class TokenizerTests : IDisposable
{
    private readonly string _testVocabPath;

    public TokenizerTests()
    {
        // Create a minimal test vocabulary
        _testVocabPath = Path.GetTempFileName();
        var vocab = new[]
        {
            "[PAD]",    // 0
            "[UNK]",    // 1
            "[CLS]",    // 2
            "[SEP]",    // 3
            "[MASK]",   // 4
            "hello",    // 5
            "world",    // 6
            "test",     // 7
            "##ing",    // 8
            "##ed",     // 9
            "the",      // 10
            "a",        // 11
            ".",        // 12
            ",",        // 13
        };
        File.WriteAllLines(_testVocabPath, vocab);
    }

    [Fact]
    public async Task CreateFromVocabAsync_LoadsVocabulary()
    {
        var tokenizer = await CreateTokenizerAsync();
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public async Task CreateFromVocabAsync_ThrowsOnMissingFile()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => BertTokenizer.CreateFromVocabAsync("/nonexistent/path"));
    }

    [Fact]
    public async Task Encode_AddsClsAndSepTokens()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, attentionMask) = tokenizer.Encode("hello", maxLength: 10);

        // [CLS] hello [SEP] [PAD]...
        Assert.Equal(2, inputIds[0]); // [CLS]
        Assert.Equal(5, inputIds[1]); // hello
        Assert.Equal(3, inputIds[2]); // [SEP]
    }

    [Fact]
    public async Task Encode_SetsCorrectAttentionMask()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, attentionMask) = tokenizer.Encode("hello", maxLength: 10);

        // First 3 tokens should have mask=1, rest should be 0
        Assert.Equal(1, attentionMask[0]); // [CLS]
        Assert.Equal(1, attentionMask[1]); // hello
        Assert.Equal(1, attentionMask[2]); // [SEP]
        Assert.Equal(0, attentionMask[3]); // [PAD]
    }

    [Fact]
    public async Task Encode_PadsToMaxLength()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, attentionMask) = tokenizer.Encode("hello", maxLength: 20);

        Assert.Equal(20, inputIds.Length);
        Assert.Equal(20, attentionMask.Length);
    }

    [Fact]
    public async Task Encode_TruncatesToMaxLength()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, attentionMask) = tokenizer.Encode("hello world test hello world", maxLength: 5);

        Assert.Equal(5, inputIds.Length);
        Assert.Equal(2, inputIds[0]); // [CLS]
        Assert.Equal(3, inputIds[^1]); // Last should be padding or [SEP]
    }

    [Fact]
    public async Task Encode_HandlesUnknownTokens()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("xyz", maxLength: 10);

        // xyz is not in vocab, should be [UNK]
        Assert.Equal(1, inputIds[1]); // [UNK]
    }

    [Fact]
    public async Task Encode_SplitsPunctuation()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("hello.", maxLength: 10);

        // [CLS] hello . [SEP]
        Assert.Equal(2, inputIds[0]);  // [CLS]
        Assert.Equal(5, inputIds[1]);  // hello
        Assert.Equal(12, inputIds[2]); // .
        Assert.Equal(3, inputIds[3]);  // [SEP]
    }

    [Fact]
    public async Task Encode_LowercasesText()
    {
        var tokenizer = await CreateTokenizerAsync(doLowerCase: true);
        var (inputIds1, _) = tokenizer.Encode("Hello", maxLength: 10);
        var (inputIds2, _) = tokenizer.Encode("hello", maxLength: 10);

        Assert.Equal(inputIds1[1], inputIds2[1]);
    }

    [Fact]
    public async Task EncodeBatch_ProcessesMultipleTexts()
    {
        var tokenizer = await CreateTokenizerAsync();
        var texts = new[] { "hello", "world" };
        var (inputIds, attentionMasks) = tokenizer.EncodeBatch(texts, maxLength: 10);

        Assert.Equal(2, inputIds.Length);
        Assert.Equal(2, attentionMasks.Length);
        Assert.Equal(10, inputIds[0].Length);
        Assert.Equal(10, inputIds[1].Length);
    }

    [Fact]
    public async Task Encode_HandlesEmptyString()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, attentionMask) = tokenizer.Encode("", maxLength: 10);

        // Should still have [CLS] and [SEP]
        Assert.Equal(2, inputIds[0]); // [CLS]
        Assert.Equal(3, inputIds[1]); // [SEP]
        Assert.Equal(1, attentionMask[0]);
        Assert.Equal(1, attentionMask[1]);
        Assert.Equal(0, attentionMask[2]);
    }

    [Fact]
    public async Task Encode_HandlesWhitespaceOnly()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("   ", maxLength: 10);

        // Only [CLS] and [SEP]
        Assert.Equal(2, inputIds[0]);
        Assert.Equal(3, inputIds[1]);
    }

    [Fact]
    public async Task Encode_HandlesSpecialCharacters()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("hello, world.", maxLength: 10);

        // [CLS] hello , world . [SEP]
        Assert.Equal(2, inputIds[0]);  // [CLS]
        Assert.Equal(5, inputIds[1]);  // hello
        Assert.Equal(13, inputIds[2]); // ,
        Assert.Equal(6, inputIds[3]);  // world
        Assert.Equal(12, inputIds[4]); // .
        Assert.Equal(3, inputIds[5]);  // [SEP]
    }

    [Fact]
    public async Task Encode_PreservesCase_WhenNotLowerCasing()
    {
        var tokenizer = await CreateTokenizerAsync(doLowerCase: false);
        var (inputIds1, _) = tokenizer.Encode("Hello", maxLength: 10);
        var (inputIds2, _) = tokenizer.Encode("hello", maxLength: 10);

        // Should be different when not lowercasing (Hello -> [UNK], hello -> 5)
        Assert.Equal(1, inputIds1[1]); // [UNK] - "Hello" not in vocab
        Assert.Equal(5, inputIds2[1]); // hello
    }

    [Fact]
    public async Task Encode_HandlesControlCharacters()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("hello\x00world", maxLength: 10);

        // Control characters should be stripped
        Assert.Equal(2, inputIds[0]); // [CLS]
    }

    [Fact]
    public async Task Encode_HandlesVeryLongWord()
    {
        var tokenizer = await CreateTokenizerAsync();
        var longWord = new string('a', 250);
        var (inputIds, _) = tokenizer.Encode(longWord, maxLength: 10);

        // Very long word exceeds max_chars_per_word (200), should become [UNK]
        Assert.Equal(2, inputIds[0]); // [CLS]
        Assert.Equal(1, inputIds[1]); // [UNK]
        Assert.Equal(3, inputIds[2]); // [SEP]
    }

    [Fact]
    public async Task EncodeBatch_HandlesEmptyList()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, attentionMasks) = tokenizer.EncodeBatch(Array.Empty<string>(), maxLength: 10);

        Assert.Empty(inputIds);
        Assert.Empty(attentionMasks);
    }

    [Fact]
    public async Task EncodeBatch_HandlesMixedContent()
    {
        var tokenizer = await CreateTokenizerAsync();
        var texts = new[] { "hello", "", "world" };
        var (inputIds, attentionMasks) = tokenizer.EncodeBatch(texts, maxLength: 10);

        Assert.Equal(3, inputIds.Length);
        Assert.Equal(3, attentionMasks.Length);

        // All should have [CLS] at start
        Assert.All(inputIds, ids => Assert.Equal(2, ids[0]));
    }

    [Fact]
    public async Task Encode_HandlesTabsAndNewlines()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("hello\tworld\ntest", maxLength: 10);

        // Tabs and newlines become spaces, text is split
        // Tokens that aren't in vocab become [UNK]
        Assert.Equal(2, inputIds[0]); // [CLS]
        Assert.Equal(5, inputIds[1]); // hello
        // world with tab prefix might be [UNK] depending on tokenization
        Assert.True(inputIds[2] == 6 || inputIds[2] == 1); // world or [UNK]
    }

    [Fact]
    public async Task Encode_MaxLength_RespectsMinimum()
    {
        var tokenizer = await CreateTokenizerAsync();
        var (inputIds, _) = tokenizer.Encode("hello world test", maxLength: 3);

        // Minimum should still include [CLS] and [SEP]
        Assert.Equal(3, inputIds.Length);
        Assert.Equal(2, inputIds[0]); // [CLS]
        Assert.Equal(3, inputIds[^1]); // [SEP]
    }

    private async Task<BertTokenizer> CreateTokenizerAsync(bool doLowerCase = true)
    {
        return await BertTokenizer.CreateFromVocabAsync(_testVocabPath, doLowerCase);
    }

    public void Dispose()
    {
        if (File.Exists(_testVocabPath))
        {
            File.Delete(_testVocabPath);
        }
    }
}
