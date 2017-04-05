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
        T Evaluate(T[] value, T[] targetValue, out T[] partialDerivatives);
    }


    public interface IDoubleCostFunction : ICostFunction<double>
    {
    }


    public class SquaredErrorFunction : IDoubleCostFunction
    {

        public double Evaluate(double[] outputs, double[] targetOutputs, out double[] partialDerivatives)
        {
            // Validate input
            if (outputs.Length != targetOutputs.Length)
            {
                throw new ArgumentException("The number of output values must correspond to the number of target output values.");
            }

            // Calculate cost and partial derivatives
            double cost = 0.0;
            double partialDerivative = 0.0;

            int outputLength = outputs.Length;
            partialDerivatives = new double[outputLength];
            for (int i = 0; i < outputLength; i++)
            {
                partialDerivative = -(targetOutputs[i] - outputs[i]);
                cost += Math.Pow(partialDerivative, 2);
                partialDerivatives[i] = partialDerivative;
            }

            return 1.0 / 2.0 * cost;
        }
        
    }
}
