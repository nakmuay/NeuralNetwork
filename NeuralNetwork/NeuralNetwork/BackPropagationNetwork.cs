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
        private IDoubleCostFunction costFunction;


        private NeuronLayer inputLayer;
        private int layerCount;
        private NeuronLayer[] hiddenLayers;
        private SynapseLayer[] synapses;

        #endregion

        public BackPropagationNetwork(int[] layerSizes, IDoubleEvaluatable[] transferFunctions, IDoubleCostFunction costFunction)
        {
            this.costFunction = costFunction;

            // Validate input
            if (transferFunctions.Length != layerSizes.Length)
            {
                throw new ArgumentException("Cannot create a network with the provided parameters.");
            }

            Console.WriteLine("Creating layers ...");

            // Initialize variables
            layerCount  = layerSizes.Length - 1;
            layerInput  = new double[layerCount][];
            layerOutput = new double[layerCount][];
            delta       = new double[layerCount][];

            // Treat input layer separately
            inputLayer = new NeuronLayer(layerSizes[0], transferFunctions[0]);

            // Create hidden layers and output layer
            hiddenLayers = new NeuronLayer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                // Add one element to hold the bias value
                layerInput[i]  = new double[layerSizes[i + 1]];
                layerOutput[i] = new double[layerSizes[i + 1]];
                delta[i]       = new double[layerSizes[i + 1]];

                hiddenLayers[i] = new NeuronLayer(layerSizes[i + 1], transferFunctions[i + 1]);
            }

            Console.WriteLine("Creating synapses ...");

            InitializeSynapses();
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
                synapses[layerIndex].Write();
            }
        }

        public void InitializeSynapses()
        {
            // Create synapses (connections) between layers
            synapses = new SynapseLayer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                synapses[i] = new SynapseLayer(i == 0 ? inputLayer : hiddenLayers[i - 1], hiddenLayers[i]);
            }
        }

        public void NudgeSynapses()
        {
            // Create synapses (connections) between layers
            for (int i = 0; i < synapses.Length; i++)
            {
                synapses[i].Nudge();
            }
        }

        public void Run(IdentificationData idInput, out IdentificationData idOutput)
        {
            double[][] output = new double[idInput.NumSamples][];
            for (int i = 0; i < idInput.NumSamples; i++)
            {
                Run(idInput.InputData[i], out output[i]);
            }

            idOutput = new IdentificationData(idInput.InputData, output);
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
                synapses[i].ForwardPropagate((i == 0 ? input : layerOutput[i - 1]), out layerInput[i]);
                hiddenLayers[i].Run(layerInput[i], out layerOutput[i]);
            }

            output = layerOutput[lastLayerIndex];
        }

        public double Train(double[] input, double[] targetOutput, double learningRate, double momentum)
        {
            // Validate input
            if (input.Length != inputLayer.Size)
            {
                throw new ArgumentException("The input data must have the same size as the network input layer.");
            }

            if (targetOutput.Length != hiddenLayers[lastLayerIndex].Size)
            {
                throw new ArgumentException("The training output data must have the same size as the network output layer.");
            }

            // Run the network
            double[] output;
            Run(input, out output);

            // Calculate cost for output layer
            double cost = costFunction.Evaluate(output, targetOutput, out delta[lastLayerIndex]);

            // Backpropagate error through hidden layers
            for (int i = lastLayerIndex; i > 0; i--)
            {
                hiddenLayers[i].CalculateDeltas(layerInput[i], delta[i], out delta[i]);
                synapses[i].Backpropagate(delta[i], out delta[i - 1]);
            }

            // Update weights
            for (int i = 0; i < synapses.Length; i++)
            {
                double[] synapseInput = (i == 0 ? input : layerOutput[i - 1]);
                synapses[i].Update(synapseInput, delta[i], learningRate, momentum);
            }

            return cost;
        }
    }
}