using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    interface IEvaluatable<T>
    {
        double Evaluate(double value);

        double EvaluateDerivative(double value);
    }

    public abstract class ActivationFunction : IEvaluatable<ActivationFunction>
    {
        abstract public double Evaluate(double value);

        abstract public double EvaluateDerivative(double value);
    }

    public class None : ActivationFunction
    {
        override public double Evaluate(double value)
        {
            return value;
        }

        override public double EvaluateDerivative(double value)
        {
            return 1.0;
        }
    }

    public class SigmoidTransferFunction : ActivationFunction
    {
        override public double Evaluate(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        override public double EvaluateDerivative(double value)
        {
            return Evaluate(value) * (1-Evaluate(value));
        }
    }

}
