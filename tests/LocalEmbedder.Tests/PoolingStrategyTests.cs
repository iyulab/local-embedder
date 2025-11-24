using LocalEmbedder.Pooling;

namespace LocalEmbedder.Tests;

public class PoolingStrategyTests
{
    private const int HiddenDim = 4;
    private const int SeqLength = 3;

    [Fact]
    public void MeanPooling_CalculatesCorrectAverage()
    {
        var strategy = new MeanPoolingStrategy();

        // Token embeddings: 3 tokens x 4 dims
        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,     // Token 0
            5, 6, 7, 8,     // Token 1
            9, 10, 11, 12   // Token 2
        };
        var attentionMask = new long[] { 1, 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        // Expected: (1+5+9)/3, (2+6+10)/3, (3+7+11)/3, (4+8+12)/3
        Assert.Equal(5.0f, result[0], precision: 5);
        Assert.Equal(6.0f, result[1], precision: 5);
        Assert.Equal(7.0f, result[2], precision: 5);
        Assert.Equal(8.0f, result[3], precision: 5);
    }

    [Fact]
    public void MeanPooling_IgnoresPaddingTokens()
    {
        var strategy = new MeanPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,     // Token 0 (real)
            5, 6, 7, 8,     // Token 1 (real)
            100, 100, 100, 100  // Token 2 (padding - should be ignored)
        };
        var attentionMask = new long[] { 1, 1, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        // Expected: (1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2
        Assert.Equal(3.0f, result[0], precision: 5);
        Assert.Equal(4.0f, result[1], precision: 5);
        Assert.Equal(5.0f, result[2], precision: 5);
        Assert.Equal(6.0f, result[3], precision: 5);
    }

    [Fact]
    public void MeanPooling_AllPadding_ReturnsZero()
    {
        var strategy = new MeanPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        var attentionMask = new long[] { 0, 0, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        Assert.All(result, v => Assert.Equal(0.0f, v, precision: 5));
    }

    [Fact]
    public void ClsPooling_ReturnsFirstTokenEmbedding()
    {
        var strategy = new ClsPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,     // CLS token
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        var attentionMask = new long[] { 1, 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        Assert.Equal(1.0f, result[0]);
        Assert.Equal(2.0f, result[1]);
        Assert.Equal(3.0f, result[2]);
        Assert.Equal(4.0f, result[3]);
    }

    [Fact]
    public void MaxPooling_ReturnsMaxPerDimension()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 10, 3, 4,    // Token 0
            5, 2, 11, 8,    // Token 1
            9, 6, 7, 12     // Token 2
        };
        var attentionMask = new long[] { 1, 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        Assert.Equal(9.0f, result[0]);   // max(1,5,9)
        Assert.Equal(10.0f, result[1]);  // max(10,2,6)
        Assert.Equal(11.0f, result[2]);  // max(3,11,7)
        Assert.Equal(12.0f, result[3]);  // max(4,8,12)
    }

    [Fact]
    public void MaxPooling_IgnoresPaddingTokens()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            100, 100, 100, 100  // Padding
        };
        var attentionMask = new long[] { 1, 1, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, SeqLength, HiddenDim);

        Assert.Equal(5.0f, result[0]);
        Assert.Equal(6.0f, result[1]);
        Assert.Equal(7.0f, result[2]);
        Assert.Equal(8.0f, result[3]);
    }

    [Fact]
    public void PoolingFactory_CreatesMeanStrategy()
    {
        var strategy = PoolingFactory.Create(PoolingMode.Mean);
        Assert.IsType<MeanPoolingStrategy>(strategy);
    }

    [Fact]
    public void PoolingFactory_CreatesClsStrategy()
    {
        var strategy = PoolingFactory.Create(PoolingMode.Cls);
        Assert.IsType<ClsPoolingStrategy>(strategy);
    }

    [Fact]
    public void PoolingFactory_CreatesMaxStrategy()
    {
        var strategy = PoolingFactory.Create(PoolingMode.Max);
        Assert.IsType<MaxPoolingStrategy>(strategy);
    }

    [Fact]
    public void MeanPooling_SingleToken()
    {
        var strategy = new MeanPoolingStrategy();
        var tokenEmbeddings = new float[] { 1, 2, 3, 4 };
        var attentionMask = new long[] { 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 1, HiddenDim);

        Assert.Equal(1.0f, result[0]);
        Assert.Equal(2.0f, result[1]);
        Assert.Equal(3.0f, result[2]);
        Assert.Equal(4.0f, result[3]);
    }

    [Fact]
    public void MaxPooling_WithNegativeValues()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            -5, -3, -1, -2,
            -1, -2, -3, -4
        };
        var attentionMask = new long[] { 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 2, HiddenDim);

        Assert.Equal(-1.0f, result[0]); // max(-5, -1)
        Assert.Equal(-2.0f, result[1]); // max(-3, -2)
        Assert.Equal(-1.0f, result[2]); // max(-1, -3)
        Assert.Equal(-2.0f, result[3]); // max(-2, -4)
    }

    [Fact]
    public void MaxPooling_AllPadding_ReturnsZero()
    {
        var strategy = new MaxPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8
        };
        var attentionMask = new long[] { 0, 0 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 2, HiddenDim);

        // When all tokens are padding, result remains zero-initialized
        Assert.All(result, v => Assert.Equal(0.0f, v));
    }

    [Fact]
    public void ClsPooling_IgnoresOtherTokens()
    {
        var strategy = new ClsPoolingStrategy();

        var tokenEmbeddings = new float[]
        {
            1, 2, 3, 4,      // CLS token
            100, 200, 300, 400
        };
        var attentionMask = new long[] { 1, 1 };
        var result = new float[HiddenDim];

        strategy.Pool(tokenEmbeddings, attentionMask, result, 2, HiddenDim);

        // Should only return the first token regardless of attention mask
        Assert.Equal(1.0f, result[0]);
        Assert.Equal(2.0f, result[1]);
        Assert.Equal(3.0f, result[2]);
        Assert.Equal(4.0f, result[3]);
    }

    [Fact]
    public void PoolingFactory_ThrowsForInvalidMode()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PoolingFactory.Create((PoolingMode)999));
    }
}
