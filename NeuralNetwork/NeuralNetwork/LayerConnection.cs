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

            for (int row = 0; row < secondLayer.Size; row++)
            {
                for (int column = 0; column < firstLayer.Size + 1; column++)
                {
                    weightMatrix[row, column] = rnd.NextDouble();
                }
            }
        }

        public void Write()
        {
            for (int row = 0; row < secondLayer.Size; row++)
            {
                for (int column = 0; column < firstLayer.Size + 1; column++)
                {
                    Console.Write(String.Format("{0} ", weightMatrix[row, column]));
                }
                Console.WriteLine("\n");
            }
        }

        public void Run(ref double[] input, out double[] output)
        {
            output = new double[secondLayer.Size];


            for (int row = 0; row < secondLayer.Size; row++)
            {
                double dotProduct = 0.0f;
                for (int column = 0; column < firstLayer.Size; column++)
                {
                    dotProduct += weightMatrix[row, column] * input[column];
                }

                // Add bias term
                dotProduct += weightMatrix[row, firstLayer.Size];
                output[row] = dotProduct;
            }
        }
    }
}
