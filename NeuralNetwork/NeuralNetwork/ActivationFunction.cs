using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    public interface IEvaluatable<T>
    {
        T Evaluate(T value);

        T EvaluateDerivative(T value);
    }

    public interface IDoubleEvaluatable : IEvaluatable<double>
    {
    }

    public class None : IDoubleEvaluatable
    {
        public double Evaluate(double value)
        {
            return value;
        }

        public double EvaluateDerivative(double value)
        {
            return 1.0;
        }
    }

    public class SigmoidActivationFunction : IDoubleEvaluatable
    {
        public double Evaluate(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public double EvaluateDerivative(double value)
        {
            return Evaluate(value) * (1 - Evaluate(value));
        }
    }

    public class TanhActivationFunction : IDoubleEvaluatable
    {
        public double Evaluate(double value)
        {
            return Math.Tanh(value);
        }

        public double EvaluateDerivative(double value)
        {
            return 1 - Math.Pow(Evaluate(value), 2);
        }
    }

}
