using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    // TODO [martin, 2017-04-05]: This should be extracted into a common interface
    //                            for both activation functions and cost functions
    public interface ICostFunction<T>
    {
        T Evaluate(T[] value, T[] targetValue);

        T[] EvaluatePartialDerivatives(T[] value, T[] targetValue);
    }


    public interface IDoubleCostFunction : ICostFunction<double>
    {
    }


    public class SquaredErrorFunction : IDoubleCostFunction
    {

        public double Evaluate(double[] outputs, double[] targetOutputs)
        {
            // Validate input
            if (outputs.Length != targetOutputs.Length)
            {
                throw new ArgumentException("The number of output values must correspond to the number of target output values.");
            }

            // Calculate cost
            double cost = 0.0;
            int outputLength = outputs.Length;
            for (int i = 0; i < outputLength; i++)
            {
                cost += Math.Pow(targetOutputs[i] - outputs[i], 2);
            }

            return 1.0 / 2.0 * cost;
        }

        public double[] EvaluatePartialDerivatives(double[] outputs, double[] targetOutputs)
        {
            // Validate input
            if (outputs.Length != targetOutputs.Length)
            {
                throw new ArgumentException("The number of output values must correspond to the number of target output values.");
            }

            // Calculate partial derivatives
            int outputLength = outputs.Length;
            double[] partialDerivative = new double[outputLength];
            for (int i = 0; i < outputLength; i++)
            {
                partialDerivative[i] = -(targetOutputs[i] - outputs[i]);
            }

            return partialDerivative;
        }

    }
}
