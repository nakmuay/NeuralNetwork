﻿using System;
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

        private int layerCount;
        private Layer[] layers;
        private LayerConnection[] layerConnections;

        #endregion

        public BackPropagationNetwork(int[] layerSizes, ActivationFunction[] transferFunctions)
        {
            Console.WriteLine("Creating layers...");
            this.layerCount = layerSizes.Length;

            // Initialize variables
            layerInput = new double[layerCount][];
            layerOutput = new double[layerCount][];
            delta = new double[layerCount][];

            layers = new Layer[layerCount];
            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                // Add one element to hold the bias value
                layerInput[layerIndex] = new double[layerSizes[layerIndex] + 1];
                layerOutput[layerIndex] = new double[layerSizes[layerIndex] + 1];
                delta[layerIndex] = new double[layerSizes[layerIndex]];

                layers[layerIndex] = new Layer(layerSizes[layerIndex], transferFunctions[layerIndex]);
            }

            // Create connections between input layer and the first hidden layer
            Console.WriteLine("Creating layer connections ...");

            // Create connections between remaining layers
            layerConnections = new LayerConnection[layerCount - 1];
            for (int connectionIndex = 0; connectionIndex < layerCount - 1; connectionIndex++)
            {
                layerConnections[connectionIndex] = new LayerConnection(layers[connectionIndex], layers[connectionIndex + 1]);
            }
        }

        public void Write()
        {
            // Print weight Matrix
            for (int layerIndex = 0; layerIndex < layerCount - 1; layerIndex++)
            {
                layerConnections[layerIndex].Write();
            }
        }

        public void Run(double[] input, out double[] output)
        {
            layerInput[0] = input;
            layers[0].Run(input, out layerOutput[0]);

            for (int l = 0; l < layerCount - 1; l++)
            {
                layerConnections[l].Run(layerOutput[l], out layerInput[l + 1]);
                layers[l + 1].Run(layerInput[l + 1], out layerOutput[l + 1]);
            }

            output = layerOutput[layerCount - 1];
        }

        public void Train(double[] input, double[] wantedOutput, double learningRate, out double error)
        {
            // Run the network
            double[] output;
            Run(input, out output);

            error = 0.0;

            // Calculate deltas
            for (int layerIndex = layerCount - 1; layerIndex > 0; layerIndex--)
            {
                for (int nodeIndex = 0; nodeIndex < layers[layerIndex].Size; nodeIndex++)
                {
                    // Handle the output layer case
                    if (layerIndex == layerCount - 1)
                    {
                        delta[layerIndex][nodeIndex] = (wantedOutput[nodeIndex] - output[nodeIndex]);
                        error += Math.Pow(delta[layerIndex][nodeIndex], 2);
                    }
                }

                if (layerIndex == layerCount - 1)
                {
                    layers[layerIndex].CalculateDeltas(delta[layerIndex], out delta[layerIndex]);
                    continue;
                }

                layerConnections[layerIndex].Backpropagate(delta[layerIndex + 1], out delta[layerIndex]);
                layers[layerIndex].CalculateDeltas(delta[layerIndex], out delta[layerIndex]);
            }

            // Update weights
            for (int connectionIndex = 0; connectionIndex < layerConnections.GetLength(0) - 1; connectionIndex++)
            {
                layerConnections[connectionIndex].UpdateWeights(layerOutput[connectionIndex], delta[connectionIndex], delta[connectionIndex + 1], learningRate);
                layerConnections[connectionIndex].UpdateDeltas(delta[connectionIndex + 1], learningRate);
            }
        }
    }
}