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

        public double[] input;

        public Layer(int size)
        {
            this.size = size;

            // Allocate input and output
            input = new double[size];
            for (int i = 0; i < size; i++)
            {
                input[i] = 0.0f;
            }
        }

        #region Properties

        public int Size
        {
            get
            {
                return this.size;
            }
        }

        #endregion

        public void Run(ref double[] input, out double[] output)
        {
            Console.WriteLine("Running layer!");
            this.input = input;
            RunCore(ref input, out output);
        }

        public virtual void RunCore(ref double[] input, out double[] output)
        {
            int size = input.Length;
            output = new double[size];
            for (int i = 0; i < size; i++)
            {
                output[i] = input[i];
            }
        }

        public void Train()
        {
            Console.WriteLine("Training layer!");
        }
    }


    class HiddenLayer : Layer
    {

        private readonly ActivationFunction transferFunction;

        public HiddenLayer(int size, ActivationFunction transferFunction) : base(size)
        {
            this.transferFunction = transferFunction;
        }

        public ActivationFunction TransferFunction
        {
            get
            {
                return this.transferFunction;
            }
        }

        public override void RunCore(ref double[] input, out double[] output)
        {
            int size = input.Length;
            output = new double[size];
            for (int i = 0; i < size; i++)
            {
                output[i] = TransferFunction.Evaluate(input[i]);
            }
        }

    }
}
