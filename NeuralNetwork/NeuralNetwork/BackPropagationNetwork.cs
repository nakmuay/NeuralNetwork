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

        private Layer inputLayer;
        private int layerCount;
        private HiddenLayer[] layers;
        private LayerConnection[] layerConnections;

        #endregion

        public BackPropagationNetwork(int inputSize, int[] layerSizes, ActivationFunction[] transferFunctions)
        {
            Console.WriteLine("Creating layers...");
            this.inputLayer = new Layer(inputSize);
            this.layerCount = layerSizes.Length;
            layers = new HiddenLayer[layerCount];
            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                layers[layerIndex] = new HiddenLayer(layerSizes[layerIndex], transferFunctions[layerIndex]);
            }

            // Create connections between input layer and the first hidden layer
            Console.WriteLine("Creating layer connections ...");
            layerConnections = new LayerConnection[layerCount];
            layerConnections[0] = new LayerConnection(inputLayer, layers[0]);

            // Create connections between remaining layers
            for (int layerIndex = 0; layerIndex < layerCount - 1; layerIndex++)
            {
                layerConnections[layerIndex + 1] = new LayerConnection(layers[layerIndex], layers[layerIndex + 1] );
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

        public void Run(ref double[] input, out double[] output)
        {
            int outputSize = layers[layerCount - 1].Size;
            output = new double[outputSize];

            double[] connectionInput;
            double[] connectionOutput;
            double[] layerOutput;
            double[] prevLayerOutput = { 0 };
            for (int l = 0; l < layerCount; l++)
            {
                if (l == 0)
                {
                    connectionInput = input;
                }
                else
                {
                    connectionInput = prevLayerOutput;
                }

                layerConnections[l].Run(ref connectionInput, out connectionOutput);
                layers[l].Run(ref connectionOutput, out layerOutput);

                prevLayerOutput = layerOutput;
            }

            output = prevLayerOutput;
        }

    }
}