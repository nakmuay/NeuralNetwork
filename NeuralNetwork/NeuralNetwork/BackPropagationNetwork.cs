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
            // Validate input
            if (transferFunctions.Length != layerSizes.Length)
            {
                throw new ArgumentException("Cannot create a network with the provided parameters.");
            }

            Console.WriteLine("Creating layers...");

            // Initialize variables
            layerCount = layerSizes.Length - 1;
            layerInput = new double[layerCount][];
            layerOutput = new double[layerCount][];
            delta = new double[layerCount][];

            // Treat input layer separately
            inputLayer = new Layer(layerSizes[0], transferFunctions[0]);

            // Create hidden layers and output layer
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

            // Create connections between layers
            layerConnections = new LayerConnection[layerCount];
            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                layerConnections[layerIndex] = new LayerConnection(layerIndex == 0 ? inputLayer : layers[layerIndex - 1], layers[layerIndex]);
            }
        }

        #region properties

        private int lastLayerIndex
        {
            get { return layerCount - 1; }
        }

        #endregion

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
            // Validate input
            if (input.Length != inputLayer.Size)
            {
                throw new ArgumentException("The input data must have the same size as the network input layer.");
            }

            // Run the network
            for (int l = 0; l < layerCount; l++)
            {
                layerConnections[l].ForwardPropagate((l == 0 ? input : layerOutput[l - 1]), out layerInput[l]);
                layers[l].Run(layerInput[l], out layerOutput[l]);
            }

            output = layerOutput[lastLayerIndex];
        }

        public void Train(double[] input, double[] wantedOutput, double learningRate, out double error)
        {
            // Validate input
            if (input.Length != inputLayer.Size)
            {
                throw new ArgumentException("The input data must have the same size as the network input layer.");
            }

            if (wantedOutput.Length != layers[lastLayerIndex].Size)
            {
                throw new ArgumentException("The training output data must have the same size as the network output layer.");
            }

            // Run the network
            double[] output;
            Run(input, out output);

            error = 0.0;
            for (int layerIndex = lastLayerIndex; layerIndex > 0; layerIndex--)
            {
                // Handle the output layer case
                if (layerIndex == lastLayerIndex)
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
            for (int layerIndex = 0; layerIndex < layerConnections.Length; layerIndex++)
            {
                double[] connectionInput = (layerIndex == 0 ? input : layerOutput[layerIndex - 1]);
                layerConnections[layerIndex].UpdateWeights(connectionInput, delta[layerIndex], learningRate);
            }
        }
    }
}