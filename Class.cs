using Microsoft.ML;
using Microsoft.ML.Data;

public class TestModel
{
    private readonly PredictionEngine<ImageData, ImagePrediction> _predictionEngine;

    public TestModel()
    {
        var mlContext = new MLContext();
        ITransformer loadedModel;

       
        string modelPath = @"C:\Users\purr\source\repos\BlazorApp1\BlazorApp1\bin\Debug\net8.0\test.mlnet";

        try
        {
            using (var stream = File.OpenRead(modelPath))
            {
                loadedModel = mlContext.Model.Load(stream, out var modelInputSchema);

               
                Console.WriteLine("Model Input Schema:");
                foreach (var column in modelInputSchema.AsEnumerable())
                {
                    Console.WriteLine($"Column Name: {column.Name}, Type: {column.Type}");
                }
            }

            _predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка загрузки модели: {ex.Message}");
            throw;
        }
    }

    public ImagePrediction Predict(byte[] imageBytes)
    {
        try
        {
          
            var data = new ImageData { Feature = imageBytes, Label = string.Empty };

           
            return _predictionEngine.Predict(data);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка при выполнении предсказания: {ex.Message}");
            throw;
        }
    }
}

public class ImageData
{
    [LoadColumn(0)] 
    public byte[] Feature { get; set; }

    [LoadColumn(1)] 
    public string Label { get; set; } = string.Empty;
}

public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; }
}