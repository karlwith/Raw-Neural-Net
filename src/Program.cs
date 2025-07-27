using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

namespace FontColourPredictor
{
    // Unified data class for both training and testing data
    public class ColorData
    {
        public float red { get; set; }
        public float green { get; set; }
        public float blue { get; set; }
        public float black { get; set; }
        public float white { get; set; }
    }
    
    class Program
    {
        static void Main(string[] args) 
        {
            Console.Clear();
            PrintHeader();
            
            // Set network variables
            float iterations = 3500;
            float learningRate = 0.1f;
            
            // Network architecture
            int[] networkStructure = { 3, 20, 20, 20, 2 };
            
            Console.WriteLine($"  Training parameters:");
            Console.WriteLine($"  • Iterations: {iterations}");
            Console.WriteLine($"  • Learning Rate: {learningRate}");
            Console.WriteLine($"  • Network: {networkStructure[0]} inputs → {string.Join(" → ", networkStructure.Skip(1).Take(networkStructure.Length - 2))} → {networkStructure[networkStructure.Length - 1]} outputs");
            Console.WriteLine();

            // Load the training data
            PrintStage("LOADING TRAINING DATA");
            List<ColorData> trainingIteration = new List<ColorData>();
            
            try
            {
                List<string> trainingData = File.ReadAllLines("../data/input/train.csv").ToList();
                Console.WriteLine($"  Loaded {trainingData.Count} training samples");

                foreach (var line in trainingData)
                {
                    string[] entries = line.Split(',');
                    ColorData newIteration = new ColorData();

                    newIteration.red = float.Parse(entries[0]);
                    newIteration.green = float.Parse(entries[1]);
                    newIteration.blue = float.Parse(entries[2]);
                    newIteration.black = float.Parse(entries[3]);
                    newIteration.white = float.Parse(entries[4]);

                    trainingIteration.Add(newIteration);               
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"  Error loading training data: {ex.Message}");
                Console.ResetColor();
                return;
            }

            // Instantiate neural network
            Console.WriteLine("\n  Creating neural network architecture...");
            NeuralNetwork net = new NeuralNetwork(networkStructure);

            // Train the network
            PrintStage("TRAINING NEURAL NETWORK");
            Console.WriteLine("  Training network with backpropagation...");
            
            // Show a progress bar during training
            Console.Write("  [");
            Console.CursorVisible = false;
            
            int progressBarWidth = 50;
            for (int i = 0; i < iterations; i++)
            {               
                foreach (var trainingItem in trainingIteration)
                {
                    net.FeedForward(new float[] { trainingItem.red, trainingItem.green, trainingItem.blue });
                    net.BackPropagation(new float[] { trainingItem.black, trainingItem.white }, learningRate);
                }
                
                // Update progress bar every 1%
                if (i % (iterations / progressBarWidth) == 0 || i == iterations - 1)
                {
                    int progress = (int)Math.Ceiling((i + 1) * 100.0 / iterations);
                    int position = (int)Math.Ceiling((i + 1) * (double)progressBarWidth / iterations);
                    Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write("█");
                    Console.ResetColor();
                }
            }
            Console.Write("]");
            Console.CursorVisible = true;
            Console.WriteLine(" Complete!");

            // To test the networks accuracy
            PrintStage("TESTING MODEL ACCURACY");
            
            // Load the test data
            List<ColorData> testingIteration = new List<ColorData>();
            
            try
            {
                List<string> testingData = File.ReadAllLines("../data/input/test.csv").ToList();
                Console.WriteLine($"  Loaded {testingData.Count} testing samples");

                foreach (var line in testingData)
                {
                    string[] entries = line.Split(',');
                    ColorData newIteration = new ColorData();

                    newIteration.red = float.Parse(entries[0]);
                    newIteration.green = float.Parse(entries[1]);
                    newIteration.blue = float.Parse(entries[2]);
                    newIteration.black = float.Parse(entries[3]);
                    newIteration.white = float.Parse(entries[4]);

                    testingIteration.Add(newIteration);
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"  Error loading testing data: {ex.Message}");
                Console.ResetColor();
                return;
            }

            // Display the predictions and errors from the test split
            Console.WriteLine("\n  Model evaluation on test data:");
            Console.WriteLine("\n  Input RGB             Prediction       Error");
            Console.WriteLine("  ------------------------------------------------");
            
            float predictionBlack, predictionWhite, errorBlack, errorWhite;
            float totalError = 0;
            int correctPredictions = 0;
            
            // Display only first 5 test results
            int resultsToShow = 5;
            
            for (int i = 0; i < testingIteration.Count; i++)
            {
                var testItem = testingIteration[i];
                
                // Convert normalized RGB back to 0-255 for display
                int r = (int)((testItem.red * 255) - 1);
                int g = (int)((testItem.green * 255) - 1);
                int b = (int)((testItem.blue * 255) - 1);
                
                predictionBlack = net.FeedForward(new float[] { testItem.red, testItem.green, testItem.blue })[0];
                predictionWhite = net.FeedForward(new float[] { testItem.red, testItem.green, testItem.blue })[1];
                
                errorBlack = 100 * Math.Abs(predictionBlack - testItem.black);
                errorWhite = 100 * Math.Abs(predictionWhite - testItem.white);
                
                bool isBlackExpected = testItem.black > testItem.white;
                bool isBlackPredicted = predictionBlack > predictionWhite;
                
                // Check if prediction matches expected
                bool isCorrect = (isBlackExpected == isBlackPredicted);
                if (isCorrect) correctPredictions++;
                
                // Add to total error
                totalError += isBlackExpected ? errorBlack : errorWhite;
                
                // Only print the first few results
                if (i < resultsToShow)
                {
                    // Print RGB values
                    Console.Write("  R:{0,3} G:{1,3} B:{2,3}    ", r, g, b);
                    
                    // Print prediction
                    if (isBlackPredicted)
                    {
                        Console.Write("BLACK {0:P1}    ", predictionBlack);
                    }
                    else 
                    {
                        Console.Write("WHITE {0:P1}    ", predictionWhite);
                    }
                    
                    // Print error
                    float error = isBlackPredicted ? errorBlack : errorWhite;
                    Console.WriteLine("{0:F2}%", error);
                }
                else if (i == resultsToShow)
                {
                    // Show truncation indicator
                    Console.WriteLine("  ... and {0} more results ...", testingIteration.Count - resultsToShow);
                }
            }
            
            Console.WriteLine("  ------------------------------------------------");
            
            float avgError = totalError / testingIteration.Count;
            float accuracy = (float)correctPredictions / testingIteration.Count * 100;
            
            Console.WriteLine($"\n  Model accuracy: {accuracy:F1}% ({correctPredictions}/{testingIteration.Count} correct)");
            Console.WriteLine($"  Average error: {avgError:F2}%");

            // Make predictions with user input
            MakePredictions(net);
        }
        
        static void MakePredictions(NeuralNetwork net)
        {
            // Check if the console supports ANSI escape codes
            bool supportsAnsiColors = true;

            PrintStage("MAKE A PREDICTION");
            Console.WriteLine("  Enter RGB values or type 'exit' to quit\n");
            
            while (true)
            {
                try
                {
                    // Get red value
                    Console.Write("  Red (0-255): ");
                    string input = Console.ReadLine();
                    if (input.ToLower() == "exit") break;
                    int r = int.Parse(input);
                    if (r < 0 || r > 255) throw new Exception("Value must be between 0 and 255");
                    
                    // Get green value
                    Console.Write("  Green (0-255): ");
                    input = Console.ReadLine();
                    if (input.ToLower() == "exit") break;
                    int g = int.Parse(input);
                    if (g < 0 || g > 255) throw new Exception("Value must be between 0 and 255");
                    
                    // Get blue value
                    Console.Write("  Blue (0-255): ");
                    input = Console.ReadLine();
                    if (input.ToLower() == "exit") break;
                    int b = int.Parse(input);
                    if (b < 0 || r > 255) throw new Exception("Value must be between 0 and 255");
                    
                    // Normalize values for the neural network
                    float redNorm = (r + 1) / 255f;
                    float greenNorm = (g + 1) / 255f;
                    float blueNorm = (b + 1) / 255f;
                    
                    // Get prediction
                    float[] predictions = net.FeedForward(new float[] { redNorm, greenNorm, blueNorm });
                    float predictionBlack = predictions[0];
                    float predictionWhite = predictions[1];
                    
                    Console.WriteLine("\n  Background: R:{0} G:{1} B:{2}", r, g, b);
                    
                    // Try to display the color using ANSI escape codes
                    if (supportsAnsiColors)
                    {
                        try 
                        {
                            Console.Write("  Color: ");
                            
                            // Use ANSI escape code for background color
                            Console.Write($"\u001b[48;2;{r};{g};{b}m          \u001b[0m");
                            Console.WriteLine();
                        }
                        catch 
                        {
                            // If it fails, disable ANSI colors for future iterations
                            supportsAnsiColors = false;
                        }
                    }
                    
                    Console.Write("  Prediction: ");
                    if (predictionBlack > predictionWhite)
                    {
                        Console.WriteLine($"BLACK TEXT ({predictionBlack:P1} confidence)");
                    }
                    else
                    {
                        Console.WriteLine($"WHITE TEXT ({predictionWhite:P1} confidence)");
                    }
                    
                    Console.WriteLine("\n  Enter another RGB value or type 'exit' to quit\n");
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"  Error: {ex.Message}");
                    Console.ResetColor();
                }
            }
        }

        static void PrintHeader()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(@"  ┌────────────────────────────────────────┐");
            Console.WriteLine(@"  │   ░█▀▀░█▀█░█▀█░▀█▀░░░█▀▀░█▀█░█░░░█▀█░█▀▄ │");
            Console.WriteLine(@"  │   ░█▀▀░█░█░█░█░░█░░░░█░░░█░█░█░░░█░█░█▀▄ │");
            Console.WriteLine(@"  │   ░▀░░░▀▀▀░▀░▀░░▀░░░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀░▀ │");
            Console.WriteLine(@"  │                                        │");
            Console.WriteLine(@"  │   ░█▀█░█▀▄░█▀▀░█▀▄░▀█▀░█▀▀░▀█▀░█▀█░█▀▄ │");
            Console.WriteLine(@"  │   ░█▀▀░█▀▄░█▀▀░█░█░░█░░█░░░░█░░█░█░█▀▄ │");
            Console.WriteLine(@"  │   ░▀░░░▀░▀░▀▀▀░▀▀░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀ │");
            Console.WriteLine(@"  └────────────────────────────────────────┘");
            Console.WriteLine();
            Console.WriteLine("  A neural network to predict optimal font color (BLACK/WHITE)");
            Console.WriteLine("  for any background color based on RGB values");
            Console.WriteLine();
            Console.ResetColor();
        }
        
        static void PrintStage(string title)
        {
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"  ■ {title}");
            Console.WriteLine("  " + new string('─', title.Length + 4));
            Console.ResetColor();
        }
    }
}