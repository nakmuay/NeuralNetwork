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
        private readonly IDoubleEvaluatable activationFunction;

        public Layer(int size, IDoubleEvaluatable activationFunction)
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

        public IDoubleEvaluatable ActivationFunction
        {
            get
            {
                return this.activationFunction;
            }
        }

        #endregion


        public void Run(double[] input, out double[] output)
        {
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

        public void CalculateDeltas(double[] input, double[] backPropDeltas, out double[] thisLayerDeltas)
        {
            thisLayerDeltas = new double[size];

            for(int i = 0; i < size; i++)
            {
                thisLayerDeltas[i] = activationFunction.EvaluateDerivative(input[i]) * backPropDeltas[i];
            }
        }
    }
}
