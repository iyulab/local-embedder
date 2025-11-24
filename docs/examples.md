# Examples

## Semantic Search

Find the most similar documents to a query:

```csharp
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

// Documents to search
var documents = new[]
{
    "The cat sat on the mat",
    "Dogs are loyal companions",
    "Machine learning transforms data into insights",
    "Natural language processing enables text understanding",
    "The quick brown fox jumps over the lazy dog"
};

// Generate embeddings for all documents
var docEmbeddings = await model.EmbedAsync(documents);

// Search query
var query = "AI and NLP applications";
var queryEmbedding = await model.EmbedAsync(query);

// Rank by similarity
var results = documents
    .Select((doc, i) => new
    {
        Document = doc,
        Score = LocalEmbedder.CosineSimilarity(queryEmbedding, docEmbeddings[i])
    })
    .OrderByDescending(x => x.Score)
    .ToList();

Console.WriteLine($"Query: {query}\n");
foreach (var result in results)
{
    Console.WriteLine($"{result.Score:F4}: {result.Document}");
}
```

## Clustering

Group similar texts together:

```csharp
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

var texts = new[]
{
    "Apple released new iPhone",
    "Google announces Android update",
    "Microsoft launches Windows 12",
    "Tesla unveils new electric car",
    "Ford reveals electric truck",
    "BMW shows electric SUV"
};

var embeddings = await model.EmbedAsync(texts);

// Simple clustering by similarity threshold
var threshold = 0.5f;
var clusters = new List<List<int>>();
var assigned = new HashSet<int>();

for (int i = 0; i < texts.Length; i++)
{
    if (assigned.Contains(i)) continue;

    var cluster = new List<int> { i };
    assigned.Add(i);

    for (int j = i + 1; j < texts.Length; j++)
    {
        if (assigned.Contains(j)) continue;

        var similarity = LocalEmbedder.CosineSimilarity(embeddings[i], embeddings[j]);
        if (similarity >= threshold)
        {
            cluster.Add(j);
            assigned.Add(j);
        }
    }

    clusters.Add(cluster);
}

// Print clusters
for (int i = 0; i < clusters.Count; i++)
{
    Console.WriteLine($"Cluster {i + 1}:");
    foreach (var idx in clusters[i])
    {
        Console.WriteLine($"  - {texts[idx]}");
    }
}
```

## Duplicate Detection

Find near-duplicate content:

```csharp
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

var documents = new[]
{
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleepy dog",
    "Machine learning is a subset of artificial intelligence",
    "ML is part of AI",
    "The weather is nice today"
};

var embeddings = await model.EmbedAsync(documents);

// Find duplicates (similarity > 0.8)
var duplicates = new List<(int, int, float)>();

for (int i = 0; i < documents.Length; i++)
{
    for (int j = i + 1; j < documents.Length; j++)
    {
        var similarity = LocalEmbedder.CosineSimilarity(embeddings[i], embeddings[j]);
        if (similarity > 0.8f)
        {
            duplicates.Add((i, j, similarity));
        }
    }
}

Console.WriteLine("Potential duplicates:");
foreach (var (i, j, sim) in duplicates)
{
    Console.WriteLine($"\nSimilarity: {sim:F4}");
    Console.WriteLine($"  1: {documents[i]}");
    Console.WriteLine($"  2: {documents[j]}");
}
```

## Question Answering (Retrieval)

Simple RAG-style retrieval:

```csharp
await using var model = await LocalEmbedder.LoadAsync("all-MiniLM-L6-v2");

// Knowledge base
var knowledge = new Dictionary<string, string>
{
    ["Paris is the capital of France"] = "France's capital city",
    ["Tokyo is the capital of Japan"] = "Japan's capital city",
    ["The speed of light is 299,792,458 m/s"] = "Physics constant",
    ["Water boils at 100 degrees Celsius"] = "Physical property",
    ["Python was created by Guido van Rossum"] = "Programming language"
};

var facts = knowledge.Keys.ToArray();
var factEmbeddings = await model.EmbedAsync(facts);

// Answer questions
var questions = new[]
{
    "What is the capital of France?",
    "Who created Python?",
    "At what temperature does water boil?"
};

foreach (var question in questions)
{
    var queryEmb = await model.EmbedAsync(question);

    var bestMatch = facts
        .Select((fact, i) => new
        {
            Fact = fact,
            Score = LocalEmbedder.CosineSimilarity(queryEmb, factEmbeddings[i])
        })
        .OrderByDescending(x => x.Score)
        .First();

    Console.WriteLine($"Q: {question}");
    Console.WriteLine($"A: {bestMatch.Fact} (score: {bestMatch.Score:F4})\n");
}
```

## Multi-language Support

Use multilingual models for cross-language search:

```csharp
await using var model = await LocalEmbedder.LoadAsync("multilingual-e5-small");

// Mixed language documents
var documents = new[]
{
    "Machine learning is transforming industries",    // English
    "L'apprentissage automatique transforme les industries",  // French
    "기계 학습이 산업을 변화시키고 있습니다",          // Korean
    "机器学习正在改变各行各业",                        // Chinese
    "機械学習は産業を変革しています"                   // Japanese
};

var embeddings = await model.EmbedAsync(documents);

// Query in English
var query = "AI in industry";
var queryEmb = await model.EmbedAsync(query);

// Find similar across languages
var results = documents
    .Select((doc, i) => new
    {
        Text = doc,
        Similarity = LocalEmbedder.CosineSimilarity(queryEmb, embeddings[i])
    })
    .OrderByDescending(x => x.Similarity);

foreach (var r in results)
{
    Console.WriteLine($"{r.Similarity:F4}: {r.Text}");
}
```

## Dependency Injection

### ASP.NET Core

```csharp
// Program.cs
builder.Services.AddSingleton<IEmbeddingModel>(sp =>
{
    var options = new EmbedderOptions
    {
        MaxSequenceLength = 256,
        Provider = ExecutionProvider.Cpu
    };

    return LocalEmbedder.LoadAsync("all-MiniLM-L6-v2", options)
        .GetAwaiter()
        .GetResult();
});

// Controller
public class SearchController : ControllerBase
{
    private readonly IEmbeddingModel _model;

    public SearchController(IEmbeddingModel model)
    {
        _model = model;
    }

    [HttpPost("embed")]
    public async Task<float[]> Embed([FromBody] string text)
    {
        return await _model.EmbedAsync(text);
    }
}
```

### Factory Pattern

```csharp
public interface IEmbeddingModelFactory
{
    Task<IEmbeddingModel> CreateAsync(string modelId);
}

public class EmbeddingModelFactory : IEmbeddingModelFactory
{
    private readonly EmbedderOptions _options;

    public EmbeddingModelFactory(IOptions<EmbedderOptions> options)
    {
        _options = options.Value;
    }

    public async Task<IEmbeddingModel> CreateAsync(string modelId)
    {
        return await LocalEmbedder.LoadAsync(modelId, _options);
    }
}

// Registration
builder.Services.Configure<EmbedderOptions>(config.GetSection("Embedder"));
builder.Services.AddSingleton<IEmbeddingModelFactory, EmbeddingModelFactory>();
```

## Caching Results

Cache embeddings to avoid recomputation:

```csharp
public class EmbeddingCache
{
    private readonly IEmbeddingModel _model;
    private readonly Dictionary<string, float[]> _cache = new();

    public EmbeddingCache(IEmbeddingModel model)
    {
        _model = model;
    }

    public async ValueTask<float[]> GetEmbeddingAsync(string text)
    {
        if (_cache.TryGetValue(text, out var cached))
        {
            return cached;
        }

        var embedding = await _model.EmbedAsync(text);
        _cache[text] = embedding;
        return embedding;
    }
}
```
