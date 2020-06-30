using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace FontColourPredictor
{
    public class NeuralNetwork
    {
        int[] layer;
        Layer[] layers;
        public NeuralNetwork(int[] layer)
        {
            // Create layer array (this.layer) to match the constructor call from tester eg {3, 25, 25, 1}
            this.layer = new int[layer.Length];
            for (int i = 0; i < layer.Length; i++)
            {
                this.layer[i] = layer[i];
            }
            // Create object for each Layer (except input layer hence -1)      
            layers = new Layer[layer.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                // Call class constructor for each layer specifying the number of inputs and the number of neurons in the current layer
                layers[i] = new Layer(layer[i], layer[i + 1]);
            }
        }

        // To process the feed forward function (defined in the Layer class) for each layer
        public float[] FeedForward(float[] inputs)
        {
            layers[0].FeedForward(inputs);
            // Starting at i = 1 as i = 0 is the input layer
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward(layers[i - 1].outputs);
            }
            // Last layers output values
            return layers[layers.Length - 1].outputs;
        }

        // To process the back propagation functions (defined in Layer class) for each layer
        public void BackPropagation(float[] expected, float learningRate)
        {
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                // For the output layer
                if (i == layers.Length - 1)
                {
                    layers[i].BackPropagationOutput(expected);
                }
                // For the hidden layer(s)
                else
                {
                    // Pass the forward layer gamma and weights
                    layers[i].BackPropagationHidden(layers[i + 1].gamma, layers[i + 1].weights);
                }
            }
            // Update the weights
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].UpdateWeights(learningRate);
            }
        }

        public class Layer
        {
            // Number of neurons in the previous layer
            int neuronsInPrevLayer;
            // Number of neurons in the current layer 
            int neuronsInCurrentLayer;
            public float[] outputs;
            public float[] inputs;
            public float[,] weights;
            // Factor the weights are changed by (weights = weights - (weightsDelta * learningRate)
            public float[,] weightsDelta;
            // For the output layer only
            public float[] gamma;
            public float[] error;
            // Random class initialized to facilitate array of random weights
            public static Random random = new Random();

            public Layer(int neuronsInPrevLayer, int neuronsInCurrentLayer)
            {
                this.neuronsInPrevLayer = neuronsInPrevLayer;
                this.neuronsInCurrentLayer = neuronsInCurrentLayer;

                inputs = new float[neuronsInPrevLayer];
                outputs = new float[neuronsInCurrentLayer];
                weights = new float[neuronsInCurrentLayer, neuronsInPrevLayer];
                weightsDelta = new float[neuronsInCurrentLayer, neuronsInPrevLayer];
                gamma = new float[neuronsInCurrentLayer];
                error = new float[neuronsInCurrentLayer];

                InitializeWeights();
            }

            // Initialize weights randomly so there is no symmetry breaking
            public void InitializeWeights()
            {
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    for (int j = 0; j < neuronsInPrevLayer; j++)
                    {
                        // Between -0.5 and 0.5
                        weights[i, j] = (float)random.NextDouble() - 0.5f;
                    }
                }
            }

            // To feed forward to this specific layer
            public float[] FeedForward(float[] inputs)
            {
                this.inputs = inputs;
                // Iterate over the neurons in the current layer
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    // To compute from scratch for each test run
                    outputs[i] = 0;
                    // Iterate over the neurons in the previous layer
                    for (int j = 0; j < neuronsInPrevLayer; j++)
                    {
                        outputs[i] += inputs[j] * weights[i, j];
                    }
                    // Activation function of Tanh
                    outputs[i] = (float)Math.Tanh(outputs[i]);
                }
                return outputs;
            }

            // Sigmoid
            public double Sigmoid(float x)
            {
                return 1 / (1 + Math.Exp(-x));
            }

            // To derive Tanh as used in back propagation
            public float TanHDerivative(float value)
            {
                return 1 - (value * value);
            }

            // Back propagation if this.layer is the output layer
            public void BackPropagationOutput(float[] expected)
            {
                // For mean squared error
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    error[i] = outputs[i] - expected[i];
                }
                // For gamma aka bias?
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    gamma[i] = error[i] * TanHDerivative(outputs[i]);
                }
                // For weightsDelta
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    for (int j = 0; j < neuronsInPrevLayer; j++)
                    {
                        weightsDelta[i, j] = gamma[i] * inputs[j];
                    }
                }
            }

            // Back propagation if this.layer is a hidden layer
            public void BackPropagationHidden(float[] gammaForward, float[,] weightsForward)
            {
                // For gamma values
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    gamma[i] = 0;
                    for (int j = 0; j < gammaForward.Length; j++)
                    {
                        gamma[i] += gammaForward[j] * weightsForward[j, i];
                    }
                    gamma[i] *= TanHDerivative(outputs[i]);
                }
                // For weightsDelta
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    for (int j = 0; j < neuronsInPrevLayer; j++)
                    {
                        weightsDelta[i, j] = gamma[i] * inputs[j];
                    }
                }
            }

            // To update the weights as part of back propagation
            public void UpdateWeights(float learningRate)
            {           
                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    for (int j = 0; j < neuronsInPrevLayer; j++)
                    {
                        weights[i, j] = weights[i, j] - (weightsDelta[i, j] * learningRate);
                    }
                }
            }
        }
    }
}