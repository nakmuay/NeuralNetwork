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
        private double[][] weightMatrix;

        public LayerConnection(Layer firstLayer, Layer secondLayer)
        {
            this.firstLayer = firstLayer;
            this.secondLayer = secondLayer;

            // Allocate matrix size
            weightMatrix = new double[firstLayer.Size + 1][];

            Random rnd = new Random();

            for (int row = 0; row < firstLayer.Size + 1; row++)
            {
                weightMatrix[row] = new double[secondLayer.Size + 1];
                for (int column = 0; column < secondLayer.Size + 1; column++)
                {
                    weightMatrix[row][column] = rnd.NextDouble();
                }
            }
        }

        public void Write()
        {
            int numberOfRows = weightMatrix.Length;
            int numberOfColumns = weightMatrix[0].Length;

            for (int row = 0; row < numberOfRows; row++)
            {
                for (int column = 0; column < numberOfColumns; column++)
                {
                    Console.Write(String.Format("{0} ", weightMatrix[row][column]));
                }
                Console.WriteLine("\n");
            }
        }

        public void Run(ref double[] input, out double[] output)
        {
            output = new double[secondLayer.Size];

            for (int column = 0; column < secondLayer.Size; column++)
            {
                double dotProduct = 0.0f;
                for (int row = 0; row < firstLayer.Size; row++)
                {
                    dotProduct += weightMatrix[row][column] * input[row];
                }
                
                // Add bias term
                dotProduct += weightMatrix[firstLayer.Size][column];
                output[column] = dotProduct;
            }
        }

    }
}
