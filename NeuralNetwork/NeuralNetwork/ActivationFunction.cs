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
            return Evaluate(value) * (1.0 - Evaluate(value));
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
            return 1 - Math.Pow(Evaluate(value), 2.0);
        }

    }

    public class ModifiedTanhActivationFunction : IDoubleEvaluatable
    {

        private double scale = 1.7159;
        private double periodScale = 2.0/3.0;

        public double Evaluate(double value)
        {
            return scale * Math.Tanh(periodScale * value);
        }

        public double EvaluateDerivative(double value)
        {
            return scale * periodScale * (1.0 - Math.Pow(Evaluate(value), 2.0));
        }

    }

    public class RectifiedLinearUnit : IDoubleEvaluatable
    {

        public double Evaluate(double value)
        {
            return Math.Max(0.0, value);
        }

        public double EvaluateDerivative(double value)
        {
            return value > 0 ? 1.0 : 0.0;
        }

    }


    public class LeakyRectifiedLinearUnit : IDoubleEvaluatable
    {

        private double positiveHalfPlaneSlope = 1.0;
        private double negativeHalfPlaneSlope = 0.01;

        public double Evaluate(double value)
        {
            return Math.Max(negativeHalfPlaneSlope * value, value);
        }

        public double EvaluateDerivative(double value)
        {
            return value > 0.0 ? positiveHalfPlaneSlope : negativeHalfPlaneSlope;
        }

    }


    public class Linear : IDoubleEvaluatable
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

}
