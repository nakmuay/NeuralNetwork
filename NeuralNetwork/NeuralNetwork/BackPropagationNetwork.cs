using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class BackPropagationNetwork
    {

        #region Private data

        private double[][] layerInput;
        private double[][] layerOutput;
        private double[][] delta;

        private Layer inputLayer;
        private int layerCount;
        private Layer[] layers;
        private LayerConnection[] layerConnections;

        #endregion

        public BackPropagationNetwork(int[] layerSizes, ActivationFunction[] transferFunctions)
        {
            Console.WriteLine("Creating layers...");

            // Initialize variables
            layerCount = layerSizes.Length - 1;
            layerInput = new double[layerCount][];
            layerOutput = new double[layerCount][];
            delta = new double[layerCount][];

            // Treat input layer separately
            inputLayer = new Layer(layerSizes[0], transferFunctions[0]);

            layers = new Layer[layerCount];
            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                // Add one element to hold the bias value
                layerInput[layerIndex] = new double[layerSizes[layerIndex + 1]];
                layerOutput[layerIndex] = new double[layerSizes[layerIndex + 1]];
                delta[layerIndex] = new double[layerSizes[layerIndex + 1]];

                layers[layerIndex] = new Layer(layerSizes[layerIndex + 1], transferFunctions[layerIndex + 1]);
            }

            Console.WriteLine("Creating layer connections ...");

            layerConnections = new LayerConnection[layerCount];
            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                // Create connections between input layer and the first hidden layer
                layerConnections[layerIndex] = new LayerConnection(layerIndex == 0 ? inputLayer : layers[layerIndex - 1], layers[layerIndex]);
            }
        }

        public void Write()
        {
            // Print weight Matrix
            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                layerConnections[layerIndex].Write();
            }
        }

        public void Run(double[] input, out double[] output)
        {
            for (int l = 0; l < layerCount; l++)
            {
                layerConnections[l].ForwardPropagate((l == 0 ? input : layerOutput[l - 1]), out layerInput[l]);
                layers[l].Run(layerInput[l], out layerOutput[l]);
            }

            output = layerOutput[layerCount - 1];
        }

        public void Train(double[] input, double[] wantedOutput, double learningRate, out double error)
        {
            // Run the network
            double[] output;
            Run(input, out output);

            error = 0.0;

            for (int layerIndex = layerCount - 1; layerIndex > 0; layerIndex--)
            {
                // Handle the output layer case
                if (layerIndex == layerCount - 1)
                {
                    for (int nodeIndex = 0; nodeIndex < layers[layerIndex].Size; nodeIndex++)
                    {
                        delta[layerIndex][nodeIndex] = (output[nodeIndex] - wantedOutput[nodeIndex]);
                        error += Math.Pow(delta[layerIndex][nodeIndex], 2);
                    }
                }

                layers[layerIndex].CalculateDeltas(layerInput[layerIndex], delta[layerIndex], out delta[layerIndex]);
                layerConnections[layerIndex].Backpropagate(delta[layerIndex], out delta[layerIndex - 1]);
            }

            // Update weights
            for (int layerIndex = 0; layerIndex < layerConnections.GetLength(0); layerIndex++)
            {
                double[] connectionInput = (layerIndex == 0 ? input : layerOutput[layerIndex - 1]);
                layerConnections[layerIndex].UpdateWeights(connectionInput, delta[layerIndex], learningRate);
            }
        }
    }
}