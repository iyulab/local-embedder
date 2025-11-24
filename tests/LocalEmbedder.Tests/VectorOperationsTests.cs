using LocalEmbedder.Utils;

namespace LocalEmbedder.Tests;

public class VectorOperationsTests
{
    [Fact]
    public void CosineSimilarity_IdenticalVectors_ReturnsOne()
    {
        var vec = new float[] { 1, 2, 3, 4, 5 };
        var result = VectorOperations.CosineSimilarity(vec, vec);
        Assert.Equal(1.0f, result, precision: 5);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_ReturnsZero()
    {
        var vec1 = new float[] { 1, 0, 0 };
        var vec2 = new float[] { 0, 1, 0 };
        var result = VectorOperations.CosineSimilarity(vec1, vec2);
        Assert.Equal(0.0f, result, precision: 5);
    }

    [Fact]
    public void CosineSimilarity_OppositeVectors_ReturnsNegativeOne()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { -1, -2, -3 };
        var result = VectorOperations.CosineSimilarity(vec1, vec2);
        Assert.Equal(-1.0f, result, precision: 5);
    }

    [Fact]
    public void EuclideanDistance_IdenticalVectors_ReturnsZero()
    {
        var vec = new float[] { 1, 2, 3 };
        var result = VectorOperations.EuclideanDistance(vec, vec);
        Assert.Equal(0.0f, result, precision: 5);
    }

    [Fact]
    public void EuclideanDistance_KnownVectors_ReturnsCorrectValue()
    {
        var vec1 = new float[] { 0, 0, 0 };
        var vec2 = new float[] { 3, 4, 0 };
        var result = VectorOperations.EuclideanDistance(vec1, vec2);
        Assert.Equal(5.0f, result, precision: 5);
    }

    [Fact]
    public void DotProduct_KnownVectors_ReturnsCorrectValue()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 4, 5, 6 };
        var result = VectorOperations.DotProduct(vec1, vec2);
        Assert.Equal(32.0f, result, precision: 5); // 1*4 + 2*5 + 3*6
    }

    [Fact]
    public void NormalizeL2_NormalizesVectorToUnitLength()
    {
        var vec = new float[] { 3, 4 };
        VectorOperations.NormalizeL2(vec);

        var norm = VectorOperations.Norm(vec);
        Assert.Equal(1.0f, norm, precision: 5);
        Assert.Equal(0.6f, vec[0], precision: 5);
        Assert.Equal(0.8f, vec[1], precision: 5);
    }

    [Fact]
    public void NormalizeL2_ZeroVector_RemainsZero()
    {
        var vec = new float[] { 0, 0, 0 };
        VectorOperations.NormalizeL2(vec);

        Assert.All(vec, v => Assert.Equal(0.0f, v));
    }

    [Fact]
    public void Norm_KnownVector_ReturnsCorrectValue()
    {
        var vec = new float[] { 3, 4 };
        var result = VectorOperations.Norm(vec);
        Assert.Equal(5.0f, result, precision: 5);
    }

    [Fact]
    public void Add_AddsVectorsInPlace()
    {
        var dest = new float[] { 1, 2, 3 };
        var source = new float[] { 4, 5, 6 };
        VectorOperations.Add(dest, source);

        Assert.Equal(5.0f, dest[0]);
        Assert.Equal(7.0f, dest[1]);
        Assert.Equal(9.0f, dest[2]);
    }

    [Fact]
    public void Divide_DividesVectorByScalar()
    {
        var vec = new float[] { 10, 20, 30 };
        VectorOperations.Divide(vec, 10);

        Assert.Equal(1.0f, vec[0]);
        Assert.Equal(2.0f, vec[1]);
        Assert.Equal(3.0f, vec[2]);
    }

    [Fact]
    public void CosineSimilarity_LargeVectors()
    {
        var vec1 = Enumerable.Range(0, 384).Select(i => (float)i).ToArray();
        var vec2 = Enumerable.Range(0, 384).Select(i => (float)i * 2).ToArray();

        var result = VectorOperations.CosineSimilarity(vec1, vec2);
        Assert.InRange(result, 0.99f, 1.0f);
    }

    [Fact]
    public void EuclideanDistance_SingleDimension()
    {
        var vec1 = new float[] { 0 };
        var vec2 = new float[] { 5 };
        var result = VectorOperations.EuclideanDistance(vec1, vec2);
        Assert.Equal(5.0f, result, precision: 5);
    }

    [Fact]
    public void DotProduct_ZeroVectors_ReturnsZero()
    {
        var vec1 = new float[] { 0, 0, 0 };
        var vec2 = new float[] { 1, 2, 3 };
        var result = VectorOperations.DotProduct(vec1, vec2);
        Assert.Equal(0.0f, result, precision: 5);
    }

    [Fact]
    public void Norm_ZeroVector_ReturnsZero()
    {
        var vec = new float[] { 0, 0, 0 };
        var result = VectorOperations.Norm(vec);
        Assert.Equal(0.0f, result, precision: 5);
    }

    [Fact]
    public void Norm_UnitVector_ReturnsOne()
    {
        var vec = new float[] { 1, 0, 0 };
        var result = VectorOperations.Norm(vec);
        Assert.Equal(1.0f, result, precision: 5);
    }

    [Fact]
    public void Add_WithNegativeValues()
    {
        var dest = new float[] { -1, -2, -3 };
        var source = new float[] { 1, 2, 3 };
        VectorOperations.Add(dest, source);

        Assert.Equal(0.0f, dest[0]);
        Assert.Equal(0.0f, dest[1]);
        Assert.Equal(0.0f, dest[2]);
    }

    [Fact]
    public void NormalizeL2_AlreadyNormalized()
    {
        var vec = new float[] { 0.6f, 0.8f };
        VectorOperations.NormalizeL2(vec);

        var norm = VectorOperations.Norm(vec);
        Assert.Equal(1.0f, norm, precision: 4);
    }

    [Fact]
    public void Divide_ByFractionalScalar()
    {
        var vec = new float[] { 1, 2, 3 };
        VectorOperations.Divide(vec, 0.5f);

        Assert.Equal(2.0f, vec[0], precision: 5);
        Assert.Equal(4.0f, vec[1], precision: 5);
        Assert.Equal(6.0f, vec[2], precision: 5);
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnEmptyVector()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        Assert.Throws<ArgumentException>(() => VectorOperations.CosineSimilarity(empty, vec));
        Assert.Throws<ArgumentException>(() => VectorOperations.CosineSimilarity(vec, empty));
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnLengthMismatch()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2 };

        Assert.Throws<ArgumentException>(() => VectorOperations.CosineSimilarity(vec1, vec2));
    }

    [Fact]
    public void EuclideanDistance_ThrowsOnEmptyVector()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        Assert.Throws<ArgumentException>(() => VectorOperations.EuclideanDistance(empty, vec));
    }

    [Fact]
    public void DotProduct_ThrowsOnLengthMismatch()
    {
        var vec1 = new float[] { 1, 2, 3, 4 };
        var vec2 = new float[] { 1, 2 };

        var ex = Assert.Throws<ArgumentException>(() => VectorOperations.DotProduct(vec1, vec2));
        Assert.Contains("lengths must match", ex.Message);
    }
}
