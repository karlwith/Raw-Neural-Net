using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FontColourPredictor
{
    class Tester
    {
        static void Main(string[] args) 
        {
            Console.WriteLine("FONT COLOUR PREDICTOR\n");
            // Set network variables
            float iterations = 3500;
            float learningRate = 0.1f;
            Console.WriteLine("Iterations - {0}\nLearning Rate - {1}\n", iterations, learningRate);

            // Load the training data
            Console.WriteLine("Loading training data...");
            List<TrainingData> trainingIteration = new List<TrainingData>();
            List<string> trainingData = File.ReadAllLines("trainingDataNormalized.csv").ToList();

            foreach (var line in trainingData)
            {
                string[] entries = line.Split(',');
                TrainingData newIteration = new TrainingData();

                newIteration.red = float.Parse(entries[0]);
                newIteration.green = float.Parse(entries[1]);
                newIteration.blue = float.Parse(entries[2]);
                newIteration.black = float.Parse(entries[3]);
                newIteration.white = float.Parse(entries[4]);

                trainingIteration.Add(newIteration);               
            }

            // Instantiate neural network - number of neurons in each layer {input, hl1, hl2, output}
            NeuralNetwork net = new NeuralNetwork(new int[] { 3, 20, 20, 20, 2 });

            // Train the network
            Console.WriteLine("Training network weights...");
            for (int i = 0; i < iterations; i++)
            {               
                foreach (var TrainingData in trainingIteration)
                {
                    net.FeedForward(new float[] { TrainingData.red,TrainingData.green,TrainingData.blue });
                    net.BackPropagation(new float[] { TrainingData.black, TrainingData.white }, learningRate);
                }             
            }

            // To test the networks accuracy
            // Load the test data
            Console.WriteLine("Loading testing data...");
            List<TestingData> testingIteration = new List<TestingData>();
            List<string> testingData = File.ReadAllLines("testDataNormalized.csv").ToList();

            foreach (var line in testingData)
            {
                string[] entries = line.Split(',');
                TestingData newIteration = new TestingData();

                newIteration.red = float.Parse(entries[0]);
                newIteration.green = float.Parse(entries[1]);
                newIteration.blue = float.Parse(entries[2]);
                newIteration.black = float.Parse(entries[3]);
                newIteration.white = float.Parse(entries[4]);

                testingIteration.Add(newIteration);
            }

            // Display the predictions and errors
            Console.WriteLine("\nInput\t\t\t\t\t\t\t\tPrediction\t\n");           
            Console.WriteLine("Red\t\tGreen\t\tBlue\t\tColour\t\tBlack\t\tWhite\t\tError (%)");
            Console.WriteLine("--------------------------------------------------------------------------------------------------------------");
            float predictionBlack, predictionWhite, errorBlack, errorWhite;
            foreach (var TestingData in testingIteration)
            {
                predictionBlack = net.FeedForward(new float[] { TestingData.red, TestingData.green, TestingData.blue })[0];
                predictionWhite = net.FeedForward(new float[] { TestingData.red, TestingData.green, TestingData.blue })[1];
                errorBlack = 100 * Math.Abs(predictionBlack - TestingData.black);
                errorWhite = 100 * Math.Abs(predictionWhite - TestingData.white);
                Console.Write("{0}\t{1}\t{2}\t", TestingData.red, TestingData.green, TestingData.blue);                   
                if (TestingData.black > TestingData.white)
                {
                    Console.Write("BLACK\t\t");
                }
                else
                {
                    Console.Write("WHITE\t\t");
                }
                Console.Write(predictionBlack); Console.Write("\t"); Console.Write(predictionWhite); Console.Write("\t");        
                if (predictionBlack < predictionWhite)
                {
                    Console.WriteLine(errorWhite);
                }
                else
                {
                    Console.WriteLine(errorBlack);
                }
            }

            // To make further predictions
            Console.WriteLine("\nMake a prediction:\n");
            float redvalue, greenvalue, bluevalue;
            while (1 > 0)
            {
                Console.WriteLine("Enter a red value between 0 and 255:");
                redvalue = (float.Parse(Console.ReadLine()) + 1) / 255;
                Console.WriteLine("Enter a green value between 0 and 255:");
                greenvalue = (float.Parse(Console.ReadLine()) + 1) / 255;
                Console.WriteLine("Enter a blue value between 0 and 255:");
                bluevalue = (float.Parse(Console.ReadLine()) + 1) / 255;               
                predictionBlack = 100 * net.FeedForward(new float[] { redvalue, greenvalue, bluevalue })[0];
                predictionWhite = 100 * net.FeedForward(new float[] { redvalue, greenvalue, bluevalue })[1];

                if (predictionBlack < predictionWhite)
                {
                    Console.WriteLine("\nWhite font is predicted - {0}% confidence\n", Math.Abs(predictionWhite));
                }
                else
                {
                    Console.WriteLine("\nBlack font is predicted - {0}% confidence\n", Math.Abs(predictionBlack));
                }
            }
        }
    }
}