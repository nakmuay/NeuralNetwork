using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    class SynapseLayer
    {
        private NeuronLayer firstLayer;
        private NeuronLayer secondLayer;

        private double[,] weightMatrix;
        private double[,] previousWeightDelta;

        private double[] bias;
        private double[] previousBiasDelta;

        private Random rng;

        public SynapseLayer(NeuronLayer firstLayer, NeuronLayer secondLayer)
        {
            this.firstLayer = firstLayer;
            this.secondLayer = secondLayer;
            this.rng = new Random();

            // Allocate weight matrix
            weightMatrix = new double[secondLayer.Size, firstLayer.Size];
            bias = new double[secondLayer.Size];

            // Allocate delta matrices
            previousWeightDelta = new double[secondLayer.Size, firstLayer.Size];
            previousBiasDelta = new double[secondLayer.Size];

            // Initialize weights and biases
            initialize(firstLayer.Size);
        }

        public void Write()
        {
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size; j++)
                {
                    Console.Write(String.Format("{0:E2} ", weightMatrix[i, j]));
                }
                Console.WriteLine("\n");
            }
        }

        public void ForwardPropagate(double[] input, out double[] output)
        {
            output = new double[secondLayer.Size];

            for (int i = 0; i < secondLayer.Size; i++)
            {
                double dotProduct = 0.0f;
                for (int j = 0; j < firstLayer.Size; j++)
                {
                    dotProduct += weightMatrix[i, j] * input[j];
                }

                // Add bias term
                output[i] = dotProduct + bias[i];
            }
        }

        public void Backpropagate(double[] deltas, out double[] backPropagatedDeltas)
        {
            backPropagatedDeltas = new double[firstLayer.Size];
            double[,] transposedWeightMatrix = getTransposedWeightMatrix();

            double dotProduct = 0.0f;
            for (int i = 0; i < transposedWeightMatrix.GetLength(0); i++)
            {
                dotProduct = 0.0f;
                for (int j = 0; j < transposedWeightMatrix.GetLength(1); j++)
                {
                    dotProduct += transposedWeightMatrix[i, j] * deltas[j];
                }

                backPropagatedDeltas[i] = dotProduct;
            }
        }

        public void Update(double[] prevLayerOutput, double[] deltas, double learningRate, double momentum)
        {
            // Declare local variables
            double weightDelta = 0.0;
            double biasDelta = 0.0;

            // Update weights
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size; j++)
                {
                    weightDelta = learningRate * deltas[i] * prevLayerOutput[j]; // + momentum * previousWeightDelta[i, j];
                    weightMatrix[i, j] -= weightDelta;
                    previousWeightDelta[i, j] = weightDelta;
                }

                // Update deltas
                biasDelta = learningRate * deltas[i]; // + momentum * previousBiasDelta[i];
                previousBiasDelta[i] = biasDelta;
                bias[i] -= biasDelta;
            }
        }

        public void Nudge()
        {
            nudgeWeights();
            nudgeBiases();
        }

        private void nudgeWeights()
        {
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size; j++)
                {
                    double std = 0.01 * weightMatrix[i, j];
                    double mean = weightMatrix[i, j];
                    GaussianGenerator gaussianRand = new GaussianGenerator(mean, std, rng);
                    weightMatrix[i, j] = gaussianRand.NextDouble();
                }
            }
        }

        private void nudgeBiases()
        {
            for (int i = 0; i < secondLayer.Size; i++)
            {
                double std = 0.01 * bias[i];
                double mean = bias[i];
                GaussianGenerator gaussianRand = new GaussianGenerator(mean, std, rng);
                bias[i] = gaussianRand.NextDouble();
            }
        }

        private void initialize(int numberOfInputs)
        {
            initializeWeights(numberOfInputs);
            initializeBiases(numberOfInputs);
        }

        private void initializeWeights(int numberOfInputs)
        {
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size; j++)
                {
                    weightMatrix[i, j] = initializeWeight(firstLayer.Size);
                }
            }
        }

        private void initializeBiases(int numberOfInputs)
        {
            for (int i = 0; i < secondLayer.Size; i++)
            {
                bias[i] = initializeBias(firstLayer.Size);
            }
        }

        private double initializeWeight(int numberOfInputs)
        {
            // TODO [martin, 2017-04-01]:   This initialization assumes a symmteric activation function, e.g. tanh, see: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf.
            //                              More ellaborate methods can be used depending on which type of activation function is used.
            
            // In case of tanh activation function
            double std = Math.Sqrt(1.0 / numberOfInputs);
            GaussianGenerator gaussianRand = new GaussianGenerator(0.0, std, rng);
            return gaussianRand.NextDouble();

            /*
            // In case of sigmoid activation function
            double std = Math.Sqrt(1.0 / numberOfInputs);
            GaussianGenerator gaussianRand = new GaussianGenerator(0.5, std, rng);
            return gaussianRand.NextDouble();
            */
        }

        private double initializeBias(int numberOfInputs)
        {
            // In case of tanh activation function

            GaussianGenerator gaussianRand = new GaussianGenerator(0.0, 1.0, rng);
            return gaussianRand.NextDouble();

            /*
            // In case of sigmoid activation function
            GaussianGenerator gaussianRand = new GaussianGenerator(0.5, 1.0, rng);
            return gaussianRand.NextDouble();
            */
        }

        private double[,] getTransposedWeightMatrix()
        {
            double[,] transposedWeightMatrix = new double[firstLayer.Size, secondLayer.Size];

            for (int i = 0; i < transposedWeightMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < transposedWeightMatrix.GetLength(1); j++)
                {
                    transposedWeightMatrix[i, j] = weightMatrix[j, i];
                }
            }

            return transposedWeightMatrix;
        }

    }
}
