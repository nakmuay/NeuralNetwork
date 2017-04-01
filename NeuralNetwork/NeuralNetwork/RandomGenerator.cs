using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class GaussianGenerator
    {

        private Random rand;
        private double mean;
        private double standardDeviation;

        public GaussianGenerator(double mean, double standardDeviation)
        {
            this.mean = mean;
            this.standardDeviation = standardDeviation;

            this.rand = new Random();
        }

        public double NextDouble()
        {
            // We need to make sure that we do not generate a zero here since we are evaluating a log function later.
            var u1 = 1.0 - rand.NextDouble();
            var u2 = 1.0 - rand.NextDouble();

            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + standardDeviation * randStdNormal;
        }

    }
}
