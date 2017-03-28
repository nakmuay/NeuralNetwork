using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    class Synapse
    {
        private Layer firstLayer;
        private Layer secondLayer;
        private double[,] weightMatrix;
        private double[,] previousWeightDelta;
        private Random rnd;

        public Synapse(Layer firstLayer, Layer secondLayer)
        {
            this.firstLayer = firstLayer;
            this.secondLayer = secondLayer;
            this.rnd = new Random();

            // Allocate weight matrix
            weightMatrix = new double[secondLayer.Size, firstLayer.Size + 1];
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size + 1; j++)
                {
                    weightMatrix[i, j] = initializeWeight(firstLayer.ActivationFunction.MeanActivation());
                }
            }

            // Allocate weight delta matrices
            previousWeightDelta = new double[secondLayer.Size, firstLayer.Size];
        }

        public void Write()
        {
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size + 1; j++)
                {
                    Console.Write(String.Format("{0} ", weightMatrix[i, j]));
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

                // TODO: [martin, 2017-03-20] Extract bias to separate field. There is no point to keep it as part of the weight matrix until the code is vectorized.
                // Add bias term
                dotProduct += weightMatrix[i, firstLayer.Size];
                output[i] = dotProduct;
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

        public void UpdateWeights(double[] prevLayerOutput, double[] deltas, double learningRate, double momentum)
        {
            // Declare local variables
            double weightDelta = 0.0;

            // Update weights
            for (int i = 0; i < secondLayer.Size; i++)
            {
                for (int j = 0; j < firstLayer.Size; j++)
                {
                    weightDelta = learningRate * deltas[i] * prevLayerOutput[j] + momentum * previousWeightDelta[i, j];
                    weightMatrix[i, j] -= weightDelta;

                    previousWeightDelta[i, j] = weightDelta;
                }

                // Update deltas
                weightMatrix[i, weightMatrix.GetLength(1) - 1] -= learningRate * deltas[i];
            }
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

        private double initializeWeight(double mean)
        {
            return rnd.NextDouble() - mean;
        }
    }
}
