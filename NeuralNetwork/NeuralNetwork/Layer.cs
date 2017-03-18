using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Layer
    {

        protected readonly int size;
        private readonly ActivationFunction activationFunction;

        public Layer(int size, ActivationFunction activationFunction)
        {
            this.size = size;
            this.activationFunction = activationFunction;
        }

        #region Properties

        public int Size
        {
            get
            {
                return this.size;
            }
        }

        public ActivationFunction ActivationFunction
        {
            get
            {
                return this.activationFunction;
            }
        }

        #endregion


        public void Run(double[] input, out double[] output)
        {
            Console.WriteLine("Running layer!");
            RunCore(input, out output);
        }

        public virtual void RunCore(double[] input, out double[] output)
        {
            int size = input.Length;
            output = new double[size];
            for (int i = 0; i < size; i++)
            {
                output[i] = activationFunction.Evaluate(input[i]);
            }
        }

        public void Train()
        {
            Console.WriteLine("Training layer!");
        }
    }
}
