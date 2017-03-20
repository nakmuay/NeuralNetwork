using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class LayerConnection
    {
        private Layer firstLayer;
        private Layer secondLayer;
        private double[,] weightMatrix;

        public LayerConnection(Layer firstLayer, Layer secondLayer)
        {
            this.firstLayer = firstLayer;
            this.secondLayer = secondLayer;

            // Allocate matrix size
            weightMatrix = new double[secondLayer.Size, firstLayer.Size + 1];

            Random rnd = new Random();

            for (int rowIndex = 0; rowIndex < secondLayer.Size; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < firstLayer.Size + 1; columnIndex++)
                {
                    weightMatrix[rowIndex, columnIndex] = rnd.NextDouble();
                }
            }
        }

        public void Write()
        {
            for (int row = 0; row < secondLayer.Size; row++)
            {
                for (int columnIndex = 0; columnIndex < firstLayer.Size + 1; columnIndex++)
                {
                    Console.Write(String.Format("{0} ", weightMatrix[row, columnIndex]));
                }
                Console.WriteLine("\n");
            }
        }

        public void ForwardPropagate(double[] input, out double[] output)
        {
            output = new double[secondLayer.Size];

            for (int rowIndex = 0; rowIndex < secondLayer.Size; rowIndex++)
            {
                double dotProduct = 0.0f;
                for (int columnIndex = 0; columnIndex < firstLayer.Size; columnIndex++)
                {
                    dotProduct += weightMatrix[rowIndex, columnIndex] * input[columnIndex];
                }

                // TODO: [martin, 2017-03-20] Extract bias to separate field. There is no point to keep it as part of the weight matrix until the code is vectorized.
                // Add bias term
                dotProduct += weightMatrix[rowIndex, firstLayer.Size];
                output[rowIndex] = dotProduct;
            }
        }

        public void Backpropagate(double[] deltas, out double[] backPropagatedDeltas)
        {
            backPropagatedDeltas = new double[firstLayer.Size];
            double[,] transposedWeightMatrix = getTransposedWeightMatrix();

            double dotProduct = 0.0f;
            for (int rowIndex = 0; rowIndex < transposedWeightMatrix.GetLength(0); rowIndex++)
            {
                dotProduct = 0.0f;
                for (int columnIndex = 0; columnIndex < transposedWeightMatrix.GetLength(1); columnIndex++)
                {
                    dotProduct += transposedWeightMatrix[rowIndex, columnIndex] * deltas[columnIndex];
                }

                backPropagatedDeltas[rowIndex] = dotProduct;
            }
        }

        public void UpdateWeights(double[] prevLayerOutput, double[] deltas, double learningRate)
        {
            // Update weights
            for (int rowIndex = 0; rowIndex < secondLayer.Size; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < firstLayer.Size; columnIndex++)
                {
                    weightMatrix[rowIndex, columnIndex] -=  learningRate * deltas[rowIndex] * prevLayerOutput[columnIndex];
                }

                // Update deltas
                weightMatrix[rowIndex, weightMatrix.GetLength(1) - 1] -= learningRate * deltas[rowIndex];
            }
        }

        private double[,] getTransposedWeightMatrix()
        {
            double[,] transposedWeightMatrix = new double[firstLayer.Size, secondLayer.Size];

            for (int rowIndex = 0; rowIndex < transposedWeightMatrix.GetLength(0); rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < transposedWeightMatrix.GetLength(1); columnIndex++)
                {
                    transposedWeightMatrix[rowIndex, columnIndex] = weightMatrix[columnIndex, rowIndex];
                }
            }

            return transposedWeightMatrix;
        }
    }
}
