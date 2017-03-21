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

        public BackPropagationNetwork(int[] layerSizes, IDoubleEvaluatable[] transferFunctions)
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
            for (int i = 0; i < layerCount; i++)
            {
                // Add one element to hold the bias value
                layerInput[i]  = new double[layerSizes[i + 1]];
                layerOutput[i] = new double[layerSizes[i + 1]];
                delta[i]       = new double[layerSizes[i + 1]];

                layers[i]      = new Layer(layerSizes[i + 1], transferFunctions[i + 1]);
            }

            Console.WriteLine("Creating layer connections ...");

            // Create connections between layers
            layerConnections = new LayerConnection[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                layerConnections[i] = new LayerConnection(i == 0 ? inputLayer : layers[i - 1], layers[i]);
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
            for (int i = 0; i < layerCount; i++)
            {
                layerConnections[i].ForwardPropagate((i == 0 ? input : layerOutput[i - 1]), out layerInput[i]);
                layers[i].Run(layerInput[i], out layerOutput[i]);
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

            // Calculate error for output layer
            error = 0.0;
            for (int i = 0; i < layers[lastLayerIndex].Size; i++)
            {
                delta[lastLayerIndex][i] = (output[i] - wantedOutput[i]);
                error += Math.Pow(delta[lastLayerIndex][i], 2);
            }

            // Backpropagate error through hidden layers
            for (int i = lastLayerIndex; i > 0; i--)
            {
                layers[i].CalculateDeltas(layerInput[i], delta[i], out delta[i]);
                layerConnections[i].Backpropagate(delta[i], out delta[i - 1]);
            }

            // Update weights
            for (int i = 0; i < layerConnections.Length; i++)
            {
                double[] connectionInput = (i == 0 ? input : layerOutput[i - 1]);
                layerConnections[i].UpdateWeights(connectionInput, delta[i], learningRate);
            }
        }
    }
}